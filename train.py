import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from .qwen2_demo import Qwen2_72B


class CompositionModel(nn.Module):
    def __init__(
        self,
        augmenting_model: Qwen2_72B,
        anchor_model: Qwen2_72B,
        augmenting_layers,
        anchor_layers,
    ):
        super(CompositionModel, self).__init__()
        self.augmenting_model = augmenting_model
        self.anchor_model = anchor_model

        self.augmenting_layers = augmenting_layers
        self.anchor_layers = anchor_layers

        self.projection_layers = nn.ModuleList(
            [
                nn.Linear(
                    augmenting_model.config.hidden_size, anchor_model.config.hidden_size
                )
                for _ in range(len(augmenting_layers))
            ]
        )
        self.cross_attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=anchor_model.config.hidden_size,
                    num_heads=anchor_model.config.num_attention_heads,
                )
                for _ in range(len(anchor_layers))
            ]
        )

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        anchor_outputs = self.anchor_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        augmenting_outputs = self.augmenting_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        anchor_hidden_states = [
            anchor_outputs.hidden_states[i] for i in self.anchor_layers
        ]
        augmenting_hidden_states = [
            augmenting_outputs.hidden_states[i] for i in self.augmenting_layers
        ]

        for i in range(len(self.anchor_layers)):
            proj_hidden = self.projection_layers[i](augmenting_hidden_states[i])
            cross_attn_output, _ = self.cross_attention_layers[i](
                query=anchor_hidden_states[i].transpose(0, 1),
                key=proj_hidden.transpose(0, 1),
                value=proj_hidden.transpose(0, 1),
            )
            anchor_hidden_states[i] = anchor_hidden_states[
                i
            ] + cross_attn_output.transpose(0, 1)

        final_output = anchor_hidden_states[-1]
        logits = self.anchor_model.lm_head(final_output)
        logits = logits.float()
        loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=anchor_outputs.past_key_values,
            hidden_states=anchor_outputs.hidden_states,
            attentions=anchor_outputs.attentions,
        )
class CustomDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=131072):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()
def train(model, dataset, optimizer, criterion, epochs=2, batch_size=8):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()

    for epoch in range(epochs):
        for batch in dataloader:
            input_ids, attention_mask = batch
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits
            
            # Shift labels and logits to align them
            labels = input_ids.clone()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss = criterion(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}")
        

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen2-1.5B-Instruct")
    augmenting_model = Qwen2_72B.from_pretrained("Qwen2-1.5B-Instruct")
    anchor_model = Qwen2_72B.from_pretrained("Qwen2-72B-Instruct")
    augmenting_layers = [0, 8, 16, 24]  # Example layers
    anchor_layers = [0, 8, 16, 24]      # Example layers

    model = CompositionModel(augmenting_model, anchor_model, augmenting_layers, anchor_layers)
    
    texts = ["Your training data here"]  # Your composition training data
    dataset = CustomDataset(tokenizer, texts)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    train(model, dataset, optimizer, criterion, epochs=3)