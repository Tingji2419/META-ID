import os
import transformers
import argparse
import torch
import logging
from torch.utils.data import ConcatDataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from TaskAlternateTrainer import TaskAlternateTrainer
import re

from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    AutoTokenizer,
    LlamaForCausalLM
)

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from utils import utils
from utils import indexing


def load_pickle(filename):
    import pickle
    with open(filename, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description='OpenP5')
    parser = utils.parse_args(parser)
    args, extras = parser.parse_known_args()
    # if args.gpu != '':
    #     torch.cuda.set_device(int(args.gpu))
    # setup
    utils.setup_logging(args)
    utils.set_seed(args.seed)
    
    # determine whether distributed
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        args.gradient_accumulation_steps = args.gradient_accumulation_steps // world_size
        
    # use wandb
    wandb_project = ""
    wandb_run_name = ""
    wandb_watch = ""  # options: false | gradients | all
    wandb_log_model = ""
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
        
    # load model, tokenizer
    if 't5' in args.backbone.lower():
        config = T5Config.from_pretrained(args.backbone)
        model = T5ForConditionalGeneration.from_pretrained(args.backbone, config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    elif 'llama' in args.backbone.lower():
        model = LlamaForCausalLM.from_pretrained(
            'meta-llama/' + args.backbone,
            # load_in_8bit=True,
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/' + args.backbone)
    else:
        raise NotImplementedError
        
        
    
        
    datasets = args.datasets.split(',')
    if len(datasets) == 1:
        dataset = datasets[0]
        train_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_train_llama.json')
        valid_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_validation_{args.valid_prompt}_llama.json')
        train_data = load_dataset("json", data_files=train_data_file, field='data')
        valid_data = load_dataset("json", data_files=valid_data_file, field='data')
    else:
        train_data_list, valid_data_list = [], []
        for dataset in datasets:
            train_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_train_llama.json')
            valid_data_file = os.path.join(args.data_path, dataset, f'{dataset}_{args.tasks}_{args.item_indexing}_validation_{args.valid_prompt}_llama.json')
            t_data = load_dataset("json", data_files=train_data_file, field='data')
            v_data = load_dataset("json", data_files=valid_data_file, field='data')
            train_data_list.append(t_data)
            valid_data_list.append(v_data)
        train_data = concatenate_datasets(train_data_list)
        valid_data = concatenate_datasets(valid_data_list)
    
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt, truncation=True, max_length=args.cutoff, padding=False, return_tensors=None,
        )
        # result = tokenizer(dataset['input'], padding="longest", truncation=True, max_length=args.cutoff)
                           
        if (isinstance(result["input_ids"][-1], int) and result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff
            and add_eos_token
           ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        elif isinstance(result["input_ids"][-1], list) and add_eos_token:
            for i in range(len(result['input_ids'])):
                if result["input_ids"][i][-1] != tokenizer.eos_token_id and len(result["input_ids"][i]) < args.cutoff:
                    result["input_ids"][i].append(tokenizer.eos_token_id)
                    result["attention_mask"][i].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    def generate_prompt(data_point, output=True):
        if isinstance(data_point['input'], list):
            if output:
                return [f'{data_point["input"][i]} {data_point["output"][i]}' for i in range(len(data_point['input']))]
            else:
                return data_point["input"]
        if output:
            return f'{data_point["input"]} {data_point["output"]}'
        else:
            return data_point["input"]
    
    def process_func(datapoint):
        if 't5' in args.backbone.lower():
            encoding = tokenize(datapoint['input'], add_eos_token=True)
            labels = tokenize(datapoint['output'], add_eos_token=True)
            encoding['labels'] = labels['input_ids'].copy()
        elif 'llama' in args.backbone.lower():
            user_prompt = generate_prompt(datapoint, output=False)
            encoding_input = tokenize(user_prompt, add_eos_token=False)
            if isinstance(user_prompt, list):
                input_len = [len(encoding_input["input_ids"][i]) for i in range(len(encoding_input["input_ids"]))]
            else:
                input_len = len(encoding_input["input_ids"])
            full_prompt = generate_prompt(datapoint)
            # print(full_prompt)
            encoding = tokenize(full_prompt)

            if isinstance(user_prompt, list):
                encoding["labels"] = [
                    [-100] * input_len[i]
                    + encoding["labels"][i][input_len[i]:] for i in range(len(encoding["labels"]))
                ]
            else:
                encoding["labels"] = (
                    [-100] * input_len
                    + encoding["labels"][input_len:]
                )

        # return encoding
        return {**datapoint,**encoding}
    
    # add token and resize embedding for collaborative indexing
    if args.item_indexing == 'collaborative':
        new_tokens = []
        for dataset in datasets:
            item_index_file = os.path.join(args.data_path, dataset, f'item_collaborative_indexing_{args.collaborative_token_size}_{args.collaborative_cluster}_{args.collaborative_last_token}.txt')
            item_info = utils.ReadLineFromFile(item_index_file)
            item_map = indexing.get_dict_from_lines(item_info)
            for idx in list(item_map.values()):
                new_tokens.extend(re.findall(r'\<.*?\>', idx))
        tokenizer.add_tokens(sorted(new_tokens))
        model.resize_token_embeddings(len(tokenizer))

    # add token and resize embedding for collaborative indexing
    elif args.item_indexing == 'metapath':
        new_tokens = []
        for dataset in datasets:
            item_index_file = os.path.join(args.data_path, dataset, f'item_metapath_indexing_kmcos_100_leakage=True_ag.txt')
            item_info = utils.ReadLineFromFile(item_index_file)
            item_map = indexing.get_dict_from_lines(item_info)
            for idx in list(item_map.values()):
                new_tokens.extend(re.findall(r'\<.*?\>', idx))
        # print(set(new_tokens))
        tokenizer.add_tokens(sorted(new_tokens))
        model.resize_token_embeddings(len(tokenizer))
        print('Add metapath Tokens, num:', len(tokenizer))



    # # Assign pre-defined embeddings to the added tokens
    if args.pad_zero_emb in ['zero', 'repli']:
        with torch.no_grad():
            for dataset in datasets:
                new_token_embeddings = load_pickle(f'{args.data_path}/{dataset}/centroid_metapath_indexing_kmcos_100_leakage=True_ag.pkl')
                for token, emb in new_token_embeddings.items():
                    if 'CT' not in token:
                        continue
                    # emb = torch.tensor(emb).to(device)
                    emb = torch.tensor(emb, dtype=torch.float16)
                    token_id = tokenizer.convert_tokens_to_ids(token)

                    # Pad the embeddings if necessary
                    padding_size = 4096 - emb.size(0)
                    if args.pad_zero_emb == 'zero':
                        padded_emb = torch.cat([emb, torch.zeros(padding_size, dtype=torch.float16)], dim=0)
                    else:
                        # Use torch.cat to replicate emb and pad if needed
                        num_replications = (padding_size // emb.size(0)) + 1
                        padded_emb = torch.cat([emb] * num_replications)
                    model.base_model.embed_tokens.weight.data[token_id] = padded_emb
                    model.lm_head.weight.data[token_id] = padded_emb
                    # model.weight.data[token_id] = padded_emb


    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left" 
        
    # no task alternating optimization if only one task in the data
    if len(set(train_data['train']['task'])) == 1:
        args.task_alternating_optim = 0
    
    if args.task_alternating_optim == 1:
        TrainSet = dict()
        for task in set(train_data['train']['task']):
            TrainSet[task] = train_data['train'].filter(lambda example: example["task"]==task)
        for task in TrainSet:
            TrainSet[task] = TrainSet[task].shuffle().map(process_func, batched=True)
        
    else:
        TrainSet = train_data['train'].shuffle().map(process_func, batched=True)

    ValidSet = valid_data['train'].shuffle().map(process_func, batched=True)

    
    # randomly initialize number related tokens
    if args.random_initialize == 1:
        # logging.info("Random initialize number related tokens")
        utils.random_initialization(model, tokenizer, args.backbone)
        
    # apply lora
    if args.lora > 0:
        # model = prepare_model_for_int8_training(model)

        config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules.split(','),
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    
    # decide output dir
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
    output_dir = os.path.join(args.model_dir, folder_name, args.item_indexing, '-'.join([args.backbone, str(args.lr), str(args.lora_r), str(args.lora_alpha), str(args.gradient_accumulation_steps)]))
    
    if args.task_alternating_optim == 1:
        trainer = TaskAlternateTrainer(model=model,
            train_dataset=TrainSet,
            eval_dataset=ValidSet if args.valid_select > 0 else None,
            args= transformers.TrainingArguments(
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_steps=args.warmup_steps,
                num_train_epochs=args.epochs,
                learning_rate=args.lr,
                fp16=True,
                logging_dir=args.log_dir,
                logging_steps=args.logging_steps,
                optim=args.optim,
                evaluation_strategy="steps" if args.valid_select > 0 else "no",
                save_strategy="steps",
                eval_steps=200 if args.valid_select > 0 else None,
                save_steps=200,
                output_dir=output_dir,
                save_total_limit=3,
                load_best_model_at_end=True if args.valid_select > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=False,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
    else:
        trainer = transformers.Trainer(
            model=model,
            train_dataset=TrainSet,
            eval_dataset=ValidSet if args.valid_select > 0 else None,
            args= transformers.TrainingArguments(
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_steps=args.warmup_steps,
                num_train_epochs=args.epochs,
                learning_rate=args.lr,
                fp16=True,
                logging_steps=args.logging_steps,
                optim=args.optim,
                evaluation_strategy="steps" if args.valid_select > 0 else "no",
                save_strategy="steps",
                eval_steps=200 if args.valid_select > 0 else None,
                save_steps=200,
                output_dir=output_dir,
                save_total_limit=3,
                load_best_model_at_end=True if args.valid_select > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=False,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
    
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    


if __name__ == "__main__":
    
    main()