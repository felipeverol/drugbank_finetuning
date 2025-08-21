from unsloth import FastLanguageModel
from transformers import TextStreamer

max_seq_length = 2048
dtype = None
load_in_4bit = True

# Modelo ajustado com LoRA
lora_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="verolfelipe/Llama-Metabolism-Absorption-LoRA-4",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(lora_model)

# Modelo base para comparação
base_model, _ = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",  
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(base_model)

alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

while True:
    user_instruction = input("Enter the instruction: ")

    prompt = alpaca_prompt.format(user_instruction, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    print("\n--- LoRA Model Response ---")
    streamer_lora = TextStreamer(tokenizer)
    _ = lora_model.generate(**inputs, streamer=streamer_lora, max_new_tokens=128)

    print("\n--- Base Model Response ---")
    streamer_base = TextStreamer(tokenizer)
    _ = base_model.generate(**inputs, streamer=streamer_base, max_new_tokens=128)
