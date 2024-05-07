from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

save_directory = "/mounts/layout/palm/pretrained"

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
