from transformers import GPTJForCausalLM, GPTJConfig
import torch

# Set the path where you want to save the model
pretrained_path = "/mounts/layout/palm/pretrained/gpt-j-6b-model.pt"

device = "cuda"
# uncomment if it is first time downloading the pretrained model
# model = GPTJForCausalLM.from_pretrained(
#     "EleutherAI/gpt-j-6B",
#     revision="float16",
#     torch_dtype=torch.float16,
# )

# torch.save(model, pretrained_path)

model = torch.load(pretrained_path)
model = model.to(device)
print("Successfully loaded pretrained GPTJ!")

# testing query on pretrained model
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)

