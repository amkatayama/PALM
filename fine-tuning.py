from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# from datasets import ConcatDataset
import os

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

save_directory = "/mounts/layout/palm/pretrained"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForCausalLM.from_pretrained(save_directory).to(device)
print("Tokenizer and Model successfully loaded!")

data_dir = "/mounts/layout/palm/inputfiles/idk"

# List all text files in the directory
data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".txt")]

train_dataset = MultiFileTextDataset(
    tokenizer=tokenizer,
    file_path = data_files,
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

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model("/mounts/layout/palm/fineTunedModel")
# trainer.save_model("./fine_tuned_model")
