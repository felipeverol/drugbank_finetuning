import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

# Arquivo CSV de entrada e saída
path = "../../avaliacao/avaliacao_500.csv"

tqdm.pandas()

# Função para garantir que os dados são strings 
def safe_str(x):
    return str(x) if pd.notnull(x) else ""

# ROUGE-L 
def rouge_l_score(referencia, gerado):
    referencia = safe_str(referencia)
    gerado = safe_str(gerado)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(referencia, gerado)
    return score['rougeL'].fmeasure

# BLEU 
def bleu_score(referencia, gerado):
    referencia = safe_str(referencia)
    gerado = safe_str(gerado)
    reference_tokens = [referencia.split()]
    generated_tokens = gerado.split()
    if not generated_tokens or not reference_tokens[0]:
        return 0.0
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothie)

# EQUAL
def respostas_iguais(referencia, gerado):
    return int(safe_str(referencia).strip() == safe_str(gerado).strip())

df = pd.read_csv(path)

# Aplica as métricas para o modelo base
df["rougeL_base"] = df.progress_apply(lambda row: rouge_l_score(row["resposta_esperada"], row["resposta_modelo_base"]), axis=1)
df["bleu_base"] = df.progress_apply(lambda row: bleu_score(row["resposta_esperada"], row["resposta_modelo_base"]), axis=1)
df["equal_base"] = df.progress_apply(lambda row: respostas_iguais(row["resposta_esperada"], row["resposta_modelo_base"]), axis=1)

# Aplica as métricas para o modelo ajustado
df["rougeL_ft"] = df.progress_apply(lambda row: rouge_l_score(row["resposta_esperada"], row["resposta_modelo_treinado"]), axis=1)
df["bleu_ft"] = df.progress_apply(lambda row: bleu_score(row["resposta_esperada"], row["resposta_modelo_treinado"]), axis=1)
df["equal_ft"] = df.progress_apply(lambda row: respostas_iguais(row["resposta_esperada"], row["resposta_modelo_treinado"]), axis=1)

df.to_csv(path, index=False)

# Mostra médias
print("\n=== MÉDIAS DAS MÉTRICAS ===")
print("Modelo Base:")
print(f"ROUGE-L: {df['rougeL_base'].mean():.4f}")
print(f"BLEU:    {df['bleu_base'].mean():.4f}")
print(f"EQUAL:   {df['equal_base'].mean():.4f}")

print("\nModelo Fine-tuned:")
print(f"ROUGE-L: {df['rougeL_ft'].mean():.4f}")
print(f"BLEU:    {df['bleu_ft'].mean():.4f}")
print(f"EQUAL:   {df['equal_ft'].mean():.4f}")