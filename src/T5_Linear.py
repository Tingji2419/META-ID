import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config

class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.d_model = config.d_model
        self.alpha = None
        self.min_extra_token_id = None
        

    def init_for_linear(self, fixed_matrix, extra_token_ids, linear_alpha=0.01, tokenizer=None):
        self.min_extra_token_id = min(extra_token_ids)
        self.alpha = linear_alpha
        self.tokenizer = tokenizer
        
        fixed_matrix_tensor = torch.tensor([fixed_matrix[token_id] for token_id in extra_token_ids], dtype=torch.float32)
        self.register_buffer('fixed_matrix', fixed_matrix_tensor)
        self.register_buffer('extra_token_ids', torch.tensor(extra_token_ids, dtype=torch.int))
        
        self.projection_layer = nn.Linear(self.fixed_matrix.size(1), self.d_model)

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
        if input_ids is not None:
            # Identify the extra tokens in the input
            # input_ids: [batch_size, input_length]
            adjusted_input_ids = input_ids - self.min_extra_token_id

            # Create a mask for extra tokens
            extra_tokens_mask = (adjusted_input_ids >= 0) & (adjusted_input_ids < len(self.extra_token_ids))
            extra_token_indices = adjusted_input_ids[extra_tokens_mask]

            # Standard embedding for regular tokens
            input_embeddings = self.encoder.embed_tokens(input_ids) # self.shared(input_ids)

            if extra_token_indices.numel() > 0:
                # Projected embedding for extra tokens
                # extra_token_indices = extra_token_indices.cuda()
                extra_token_embeddings = self.fixed_matrix[extra_token_indices]
                projected_embeddings = self.projection_layer(extra_token_embeddings)
                projected_embeddings = projected_embeddings.to(torch.float32)

                # Flatten extra_tokens_mask to match the shape for embedding replacement
                flat_extra_tokens_mask = extra_tokens_mask.view(-1)
                flat_input_embeddings = input_embeddings.view(-1, self.d_model)
                flat_input_embeddings[flat_extra_tokens_mask] += self.alpha * projected_embeddings

                inputs_embeds = flat_input_embeddings.view_as(input_embeddings)

        # Continue with the forward pass as usual
        outputs = super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)
        return outputs