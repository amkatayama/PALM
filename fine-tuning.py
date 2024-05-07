from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

save_directory = "/mounts/layout/palm/pretrained"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(save_directory)
# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained(save_directory).to(device)

data_dir = "/mounts/layout/palm/inputfiles"

# List all text files in the directory
data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".txt")]

# Create training datasets for each file
train_datasets = []
for file in data_files:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            text=text,
            block_size=128  # Define the maximum sequence length
        )
        train_datasets.append(train_dataset)

# Concatenate all training datasets
combined_train_dataset = ConcatDataset(train_datasets)

# train_dataset = TextDataset(
    # tokenizer=tokenizer,
    # file_path="/mounts/layout/palm/inputfiles/idk/",
    # file_path="./hello.txt",
    # block_size=128  # Define the maximum sequence length
# )

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
    train_dataset=combined_train_dataset,
)

trainer.train()
trainer.save_model("./fine_tuned_model")
