import json
import pandas as pd
import random
from unsloth import FastLanguageModel
from tqdm import tqdm

### Configurações

# Caminho do JSON de entrada 
dataset_path = "../../datasets/metabolism_absorption_alpaca.json"

# Caminho do arquivo CSV de saída
output_path = "../../avaliacao/avaliacao_500.csv"

alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""

### Carrega dados
with open(dataset_path, "r", encoding="utf-8") as f:
    dados = json.load(f)

# Amostragem aleatória de 500 exemplos
dados = random.sample(dados, 500)

### Carrega modelos

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

# Modelo base
base_model, _ = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(base_model)

### Função para gerar respostas

def gerar_resposta(model, prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()

### Gera respostas

resultados = []
for exemplo in tqdm(dados):
    pergunta = exemplo["instruction"]
    resposta_esperada = exemplo.get("output", "")

    prompt = alpaca_prompt.format(pergunta)

    resposta_lora = gerar_resposta(lora_model, prompt)
    resposta_base = gerar_resposta(base_model, prompt)

    resultados.append({
        "pergunta": pergunta,
        "resposta_esperada": resposta_esperada,
        "resposta_modelo_treinado": resposta_lora,
        "resposta_modelo_base": resposta_base
    })

### Salva resultados em CSV

df_resultado = pd.DataFrame(resultados)
df_resultado.to_csv(output_path, index=False)
print(f"\nArquivo salvo em: {output_path}")
