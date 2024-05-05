from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Step 1: Prepare Your Data
# Assuming you have a directory 'data' containing your text files
#data_files = "./data/*.txt"
data_files = "try.txt"

# Step 2: Tokenize Your Data
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
print("LLM loaded")
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=data_files,
    block_size=128
)

# Step 3: Define Fine-tuning Parameters
training_args = TrainingArguments(
    output_dir="./gpt-j-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2
)

# Step 4: Fine-tune the Model
model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-j-6B")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("./gpt-j-finetuned")
tokenizer.save_pretrained("./gpt-j-finetuned")
