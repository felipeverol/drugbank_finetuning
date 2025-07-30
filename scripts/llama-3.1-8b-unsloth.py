#!/usr/bin/env python
# coding: utf-8

# ### Configurações do wandb e huggingface
import os
import wandb
from huggingface_hub import HfFolder

# Setar as variáveis de ambiente
os.environ["WANDB_API_KEY"] = ""
os.environ["HF_TOKEN"] = ""

# Login no WandB
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Login no Hugging Face
HfFolder.save_token(os.getenv("HF_TOKEN"))

print("✅ Logins realizados com sucesso!")

# ### Instanciar modelo e tokenizer + configuração do PEFT model

from unsloth import FastLanguageModel

max_seq_length = 2048 
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)

# ### Preparação do dataset

alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("verolfelipe/drug_bank_metabolism_absorption_alpaca_no_input", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# ### Configurações de treino

from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 50,
        num_train_epochs = 2,
        learning_rate = 2e-4,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs-llama",
        report_to = "wandb", 
    ),
)

# ### Treinar !!!
trainer_stats = trainer.train()

# ### Salvar modelo e tokenizer

# Salva adapters
model.push_to_hub("verolfelipe/Llama-Metabolism-Absorption-LoRA-3", token = "")
tokenizer.push_to_hub("verolfelipe/Llama-Metabolism-Absorption-LoRA-3", token = "")

# Salva mergeado (16 bits)
model.push_to_hub_merged("verolfelipe/Llama-Metabolism-Absorption", tokenizer, save_method = "merged_16bit", token = "")
