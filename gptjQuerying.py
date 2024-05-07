from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

inFile = open("/mounts/layout/palm/QandA/model.txt", "r")

modelName = inFile.read()
modelDirectory = '/mounts/layout/palm/fineTunedModel/' + modelName
inFile.close()

tokenizer = AutoTokenizer.from_pretrained(modelDirectory)
model = AutoModelForCausalLM.from_pretrained(modelDirectory)

inFile = open("/mounts/layout/palm/QandA/query.txt")
query = inFile.read()

gen = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=300)

generated_text = gen(query)
print(generated_text)
