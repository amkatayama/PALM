from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
import torch
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# from datasets import ConcatDataset
import os
from torch.optim import AdamW

class MultiFileTextDataset(Dataset):
    def __init__(self, tokenizer, file_paths, block_size):
        self.tokenizer = tokenizer
        self.examples = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                tokenized_text = tokenizer(text, truncation=True, padding="max_length", max_length=block_size, return_tensors="pt")
                self.examples.extend(tokenized_text.input_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

torch.cuda.init()
save_directory = "/mounts/layout/palm/pretrained"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForCausalLM.from_pretrained(save_directory).to(device)
print(f"Model loaded on {model.device}")
# print("Tokenizer and Model successfully loaded!")

# Set the padding token to the EOS token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

data_dir = "/mounts/layout/palm/inputfiles/idk"

# List all text files in the directory
data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".txt")]

train_dataset = MultiFileTextDataset(
    tokenizer=tokenizer,
    file_paths = data_files,
    block_size=50  # Define the maximum sequence length
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Fine-Tuning Configuration
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=train_dataset,
# )

# Create a DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Define an optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Manual training loop
model.train()
for epoch in range(3):  # Let's say you want to train for 3 epochs
    for step, batch in enumerate(train_dataloader):
        inputs = batch.to(device)
        print(f"Inputs loaded on {inputs.device}")
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

# trainer.train()
torch.save(model, "/mounts/layout/palm/fineTunedModel")
# model.save_model("/mounts/layout/palm/fineTunedModel")
# trainer.save_model("./fine_tuned_model")
