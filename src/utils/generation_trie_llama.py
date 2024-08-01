class Llama_Trie(object):
    def __init__(self, dataset, item_map):
        super().__init__()

        self.allowed_tokens = None
        self.item_map = item_map
        self.dataset = dataset

    def get_prefix_allowed_tokens_fn(self, tokenizer):
        
        self.allowed_tokens = {}
        prefix = tokenizer(f"{self.dataset} item")["input_ids"][1:]
        for i, token in enumerate(prefix):
            self.allowed_tokens[i] = set([token])

        for index in self.item_map.values():
            index = index.replace("><", "> <").split()
            for i, token in enumerate(index):
                token_id = tokenizer(token)["input_ids"][1]
                if i + len(prefix) not in self.allowed_tokens.keys():
                    self.allowed_tokens[i + len(prefix)] = set()
                self.allowed_tokens[i + len(prefix)].add(token_id)
        self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        # self.allowed_tokens[len(self.allowed_tokens.keys()) + 1] = set([tokenizer.eos_token_id])
        sep = tokenizer("?")["input_ids"][1:]

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn