from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

save_directory = "/mounts/layout/palm/pretrained"

tokenizer = AutoTokenizer.from_pretrained(save_directory)

model = AutoModelForCausalLM.from_pretrained(save_directory)

gen = pipeline("text-generation",model=model,tokenizer=tokenizer)

generated_text = gen("My Name is philipp")
print(generated_text)