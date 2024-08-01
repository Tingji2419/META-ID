import os
import transformers
import argparse
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from utils import utils
from peft import PeftModel
import utils.generation_trie as gt
from tqdm import tqdm
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    AutoTokenizer,
    LlamaForCausalLM,
    DataCollatorWithPadding
)

import pdb

from utils import indexing
from utils import evaluate

def main():
    parser = argparse.ArgumentParser(description='OpenP5Testing')
    # global arguments
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--log_dir", type=str, default='../log', help='The log directory')
    parser.add_argument('--logging_level', type=int, default=logging.INFO,help='Logging Level, 0, 10, ..., 50')
    parser.add_argument("--checkpoint_path", type=str, default='../model/Beauty/sequential/t5-small', help='The prompt used for evaluation, seen/unseen: id')
    parser.add_argument("--backbone", type=str, default='t5-small', help='backbone model name')
    parser.add_argument("--lora", type=int, default=0, help='whether user lora.')
    # arguments related to dataset
    parser.add_argument("--data_path", type=str, default='../data', help="data directory")
    parser.add_argument("--item_indexing", type=str, default='sequential', help="item indexing method, including random, sequential and collaborative")
    parser.add_argument("--tasks", type=str, default='sequential,straightforward', help="Downstream tasks, separate by comma")
    parser.add_argument("--dataset", type=str, default='Beauty', help="Dataset names, separate by comma")
    parser.add_argument("--cutoff", type=int, default=512, help='cutoff length for data')
    parser.add_argument("--pad_zero_emb", type=str, default='zero', help="initialize new token with zero")
    parser.add_argument("--linear", action='store_true')
    # arguments related to item indexing methods
    parser.add_argument("--sequential_order", type=str, default='original', help='The rank of user history during ')
    parser.add_argument("--collaborative_token_size", type=int, default=200, help='the number of tokens used for indexing')
    parser.add_argument("--collaborative_cluster", type=int, default=20, help='the number of clusters in each level for collaborative indexing.')
    parser.add_argument("--collaborative_last_token", type=str, default='sequential', help='how to assign the last token to items within the same clusters, random or sequential')
    parser.add_argument("--collaborative_float32", type=int, default=0, help='1 for use float32 during indexing, 0 for float64.')
    # arguments related for evaluations
    parser.add_argument("--valid_prompt", type=str, default='seen:0', help='The prompt used for evaluation, seen/unseen: id')
    parser.add_argument("--test_prompt", type=str, default='seen:0', help='The prompt used for evaluation, seen/unseen: id')
    parser.add_argument("--metrics", type=str, default='hit@5,hit@10,ndcg@5,ndcg@10', help='Metrics used for evaluation')
    parser.add_argument("--eval_batch_size", type=int, default=32, help="the batch size for evaluation")
    args, extras = parser.parse_known_args()
    
    # setup
    log_file = os.path.join(args.log_dir, args.dataset, args.checkpoint_path.replace('.','').replace('/', '_') + '.log')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=args.logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    utils.set_seed(args.seed)
    
    # get all items
    if args.item_indexing == 'sequential':
        item_index_file = os.path.join(args.data_path, args.dataset, f'item_sequential_indexing_{args.sequential_order}.txt')
        
    elif args.item_indexing == 'random':
        
        item_index_file = os.path.join(args.data_path, args.dataset, 'item_random_indexing.txt')
        
    elif args.item_indexing == 'collaborative':
        item_index_file = os.path.join(args.data_path, args.dataset, f'item_collaborative_indexing_{args.collaborative_token_size}_{args.collaborative_cluster}_{args.collaborative_last_token}.txt')
    elif args.item_indexing == 'metapath':
        item_index_file = os.path.join(args.data_path, args.dataset, f'item_metapath_indexing_kmcos_100_leakage=True_ag.txt')
    else:
        raise NotImplementedError
        
    item_info = utils.ReadLineFromFile(item_index_file)
    item_map = indexing.get_dict_from_lines(item_info)
    all_items = list(item_map.values())

    # load checkpoint
    if 't5' in args.backbone.lower():
        if args.linear:
            from T5_Linear import CustomT5ForConditionalGeneration
            model = CustomT5ForConditionalGeneration.from_pretrained(args.checkpoint_path)
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path)

        # model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
        print(f'Model: {model}')
        print(f'Model Input embd size:{model.get_input_embeddings()}')
        print(f'Model Output embd size:{model.lm_head}')
        # if args.item_indexing == 'metapath' or args.item_indexing == 'collaborative':
        #     new_tokens = []
        #     import re
        #     for idx in list(item_map.values()):
        #         new_tokens.extend(re.findall(r'\<.*?\>', idx))
        #     tokenizer.add_tokens(sorted(new_tokens))
            
        #     print('Add new Tokens, num:', len(tokenizer))
        #     model.resize_token_embeddings(len(tokenizer))

    elif 'llama' in args.backbone.lower():
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/' + args.backbone)
        if args.lora > 0:
            model = LlamaForCausalLM.from_pretrained(
                'meta-llama/' + args.backbone,
                # load_in_8bit=True,
                torch_dtype=torch.float16
            )

            if args.item_indexing == 'metapath' or args.item_indexing == 'collaborative':
                new_tokens = []
                import re
                for idx in list(item_map.values()):
                    new_tokens.extend(re.findall(r'\<.*?\>', idx))
                tokenizer.add_tokens(sorted(new_tokens))
                
                print('Add new Tokens, num:', len(tokenizer))
                model.resize_token_embeddings(len(tokenizer))
            

            model = PeftModel.from_pretrained(
                model,
                args.checkpoint_path,
                torch_dtype=torch.float16
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                args.checkpoint_path,
                load_in_8bit=True,
                torch_dtype=torch.float16
            )

        embeddings = model.get_input_embeddings()
        print(embeddings(torch.LongTensor([31000])))
        print(embeddings(torch.LongTensor([0])))
        assert 0
                    
        # if args.item_indexing == 'metapath' or args.item_indexing == 'collaborative':
        #     new_tokens = []
        #     import re
        #     for idx in list(item_map.values()):
        #         new_tokens.extend(re.findall(r'\<.*?\>', idx))
        #     tokenizer.add_tokens(sorted(new_tokens))
            
        #     print('Add new Tokens, num:', len(tokenizer))
        #     model.resize_token_embeddings(len(tokenizer))

    else:
        raise NotImplementedError




    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left" 
        
    # load test data
    test_data_file = os.path.join(args.data_path, args.dataset, f'{args.dataset}_{args.tasks}_{args.item_indexing}_test_{args.test_prompt}.json')
    test_data = load_dataset("json", data_files=test_data_file, field='data')
    # test_data = load_dataset("json", data_files="/data/yehj/huangtj/OpenP5-old/data/Beauty/Beauty_sequential,straightforward_metapath_test_seen:0.json", field='data')
    
    model.eval()
    
    candidates = all_items
    candidate_trie = gt.Trie(
        [
            [0] + tokenizer.encode(f"{args.dataset} item_{candidate}")
            for candidate in candidates
        ]
    )
    prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
    
    task_list = np.unique(test_data['train']['task'])
    metrics = args.metrics.split(',')
    generate_num = max([int(m.split('@')[1]) for m in metrics])
    for task in task_list:
        logging.info(f'testing on {task}')
        subset_data = test_data.filter(lambda example: example['task'] == task)
        dataset = EvaluationDataset(subset_data['train'], tokenizer, args.cutoff)
        dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False)
        test_single_task(model, tokenizer, dataloader, args, metrics, generate_num, prefix_allowed_tokens)
        
class EvaluationDataset(Dataset):
    def __init__(self, dataset, tokenizer, cutoff):
        super().__init__()
        self.data = dataset
        self.max_length = cutoff
        self.input = tokenizer(
            dataset['input'], padding="longest", truncation=True, max_length=cutoff
        )
        self.output = tokenizer(
            dataset['output'], padding="longest", truncation=True, max_length=cutoff
        )

    def __len__(self):
        return len(self.input["input_ids"])

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.input["input_ids"][index]),
            "attention_mask": torch.tensor(self.input["attention_mask"][index]),
            'label': torch.tensor(self.output["input_ids"][index])
        }    
    
def test_single_task(model, tokenizer, dataloader, args, metrics, generate_num, prefix_allowed_tokens):

    test_total = 0
    metrics_res = np.array([0.0] * len(metrics))
    model.cuda()
    for batch_i, batch in tqdm(enumerate(dataloader)):

        input_ids = batch["input_ids"].cuda()
        attention_mask = batch.get("attention_mask").cuda() if "attention_mask" in batch else None

        # Generate output sequence and capture cross-attention weights
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                output_attentions=True, return_dict_in_generate=True)

        # Get the cross-attention weights of the last layer in the decoder
        cross_attention = output.cross_attentions[-1]  # Last layer's cross-attention
        cross_attention = torch.cat(output.cross_attentions[-1], dim=2)
        cross_attention_mean = cross_attention.mean(dim=1)  # Average over the heads

        input_text = dataloader.dataset.data[batch_i]["input"]
        input_text = tokenizer.tokenize(input_text)
        input_text.append(tokenizer.eos_token)
        output_text = tokenizer.tokenize(tokenizer.decode(output.sequences[0], skip_special_tokens=True))
        output_text.append(tokenizer.eos_token)

        
        for i in range(cross_attention_mean.shape[0]):
            plt.figure(figsize=(20, 4))
            non_pad_positions = input_ids[i] != tokenizer.pad_token_id
            ax = sns.heatmap(cross_attention_mean[i][:, non_pad_positions].detach().cpu().numpy(), annot=False) 
            ax.set_xticklabels(input_text, rotation=45, ha="right")
            ax.set_yticklabels(output_text, rotation=0)
            plt.xlabel("Input Sequence")
            plt.ylabel("Output Sequence")
            plt.title("Attention Weights of the Last Encoder Layer (First Head)")
            plt.savefig(f'/data/yehj/huangtj/OpenP5-old/plt/tmp_{batch_i}.png', dpi=300)
            plt.clf()
        continue



        prediction = model.generate(
            input_ids=batch["input_ids"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
            max_length=30,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=generate_num,
            num_return_sequences=generate_num,
            output_scores=True,
            return_dict_in_generate=True,
        )
        
        output_ids = batch['label'].cuda()  # [B, one_token_id]
        prediction_ids = prediction["sequences"]    #[B*10, one_token_id]
        prediction_scores = prediction["sequences_scores"]

        gold_sents = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        generated_sents = tokenizer.batch_decode(
            prediction_ids, skip_special_tokens=True
        )

        # print(generated_sents)
        # exit()
        rel_results = evaluate.rel_results(generated_sents, gold_sents, prediction_scores, generate_num)

        test_total += len(rel_results)

        metrics_res += evaluate.get_metrics_results(rel_results, metrics)

    metrics_res = torch.tensor(metrics_res)
    test_total = torch.tensor(test_total)

    metrics_res /= test_total

    for i in range(len(metrics)):
        logging.info(f'{metrics[i]}: {metrics_res[i]}')
        
    
if __name__ == "__main__":
    
    main()