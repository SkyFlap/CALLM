import wandb
import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from .qwen2_demo import Qwen2_72B

class CompositionModel(nn.Module):
    def __init__(self, augmenting_model, anchor_model, augmenting_layers, anchor_layers):
        super(CompositionModel, self).__init__()
        self.augmenting_model = augmenting_model
        self.anchor_model = anchor_model
        self.augmenting_layers = augmenting_layers
        self.anchor_layers = anchor_layers

        self.projection_layers = nn.ModuleList(
            [nn.Linear(augmenting_model.config.hidden_size, anchor_model.config.hidden_size) for _ in range(len(augmenting_layers))]
        )
        self.cross_attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=anchor_model.config.hidden_size, num_heads=anchor_model.config.num_attention_heads) for _ in range(len(anchor_layers))]
        )

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        anchor_outputs = self.anchor_model.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_hidden_states=True)
        augmenting_outputs = self.augmenting_model.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_hidden_states=True)

        anchor_hidden_states = [anchor_outputs.hidden_states[i] for i in self.anchor_layers]
        augmenting_hidden_states = [augmenting_outputs.hidden_states[i] for i in self.augmenting_layers]

        for i in range(len(self.anchor_layers)):
            proj_hidden = self.projection_layers[i](augmenting_hidden_states[i])
            cross_attn_output, _ = self.cross_attention_layers[i](
                query=anchor_hidden_states[i].transpose(0, 1),
                key=proj_hidden.transpose(0, 1),
                value=proj_hidden.transpose(0, 1),
            )
            anchor_hidden_states[i] = anchor_hidden_states[i] + cross_attn_output.transpose(0, 1)

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
    def __init__(self, tokenizer, file_path):
        self.tokenizer = tokenizer
        self.samples = []
        
        with open(file_path, "r") as f:
            data = json.load(f)
            for item in data:
                instruction = item['reference'] + '\n' + item.get('input', '')
                output = item['output']
                self.samples.append((instruction, output))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        instruction, output = self.samples[idx]
        inputs = self.tokenizer(instruction, return_tensors='pt', max_length=1024, truncation=True, padding='max_length')
        outputs = self.tokenizer(output, return_tensors='pt', max_length=1024, truncation=True, padding='max_length')

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = outputs['input_ids'].squeeze()

        return input_ids, attention_mask, labels

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return scheduler

def train(model, dataset, epochs=2, batch_size=1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    num_training_steps = epochs * len(dataloader)
    num_warmup_steps = num_training_steps // 10
    scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    # Initialize wandb
    wandb.init(project="composition_model_training", entity="your_wandb_username")
    wandb.watch(model, log="all", log_freq=10)

    for epoch in range(epochs):
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift labels and logits to align them
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute loss
            loss = criterion(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log metrics to wandb
            wandb.log({"loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]})

        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}")
        # Log epoch loss to wandb
        wandb.log({"epoch": epoch+1, "loss": loss.item()})

    wandb.finish()

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen2-1.5B-Instruct")
    augmenting_model = Qwen2_72B.from_pretrained("Qwen2-1.5B-Instruct")
    anchor_model = Qwen2_72B.from_pretrained("Qwen2-72B-Instruct")
    augmenting_layers = [0, 8, 16, 24]
    anchor_layers = [0, 8, 16, 24]

    model = CompositionModel(augmenting_model, anchor_model, augmenting_layers, anchor_layers)

    dataset = CustomDataset(tokenizer, "dataset.json")
    
    train(model, dataset, epochs=3, batch_size=1)
