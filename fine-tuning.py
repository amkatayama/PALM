from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

save_directory = "/mounts/layout/palm/pretrained"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(save_directory)
# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained(save_directory).to(device)

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="/mounts/layout/palm/inputfiles/idk/",
    # file_path="./hello.txt",
    block_size=128  # Define the maximum sequence length
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
trainer.save_model("./fine_tuned_model")
