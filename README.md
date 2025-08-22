# DrugBank Metabolism & Absorption LLM Fine-Tuning

This repository contains the data and the scripts for my undergraduate research project focused on fine-tuning large language models (LLMs) to answer questions about drug metabolism and absorption using DrugBank data.

## Project Overview

The goal of this project is to fine-tune a LLM and make it capable of accurately answering questions related to the metabolism and absorption of drugs. The project involves:

- Extracting and processing relevant data from DrugBank XML file.
- Generating question-answer pairs in the Alpaca format.
- Incorporating negative examples to improve model robustness.
- Fine-tuning a Llama 3.1-8B model using the Unsloth library.
- Evaluating the model's performance using standard NLP metrics.

## Repository Structure

```
drugbank_finetuning/
├── avaliacao/                # Evaluation outputs (CSV files)
├── datasets/                 # Processed datasets (JSON, XML, TXT)
├── scripts/                  # All scripts for data prep, training, and evaluation
│   ├── gerar avaliacoes/     # Scripts for generating and evaluating model responses
│   ├── preparacao dos dados/ # Data extraction and prompt generation scripts
│   └── treinamento/          # Model training scripts
├── .gitignore
```

## Main Components

- **Data Preparation**
  - `extrai-metabolism-absorption.py`: Extracts metabolism and absorption fields from the DrugBank XML.
  - `metabolism-absorption-prompts.py`: Generates question-answer pairs and incorporates negative examples.

- **Model Training**
  - `llama-3.1-8b-unsloth.py`: Fine-tunes the Llama 3.1-8B model using the Unsloth library and the prepared dataset.

- **Evaluation**
  - `gera_respostas_csv.py`: Both the base and fine-tuned model generate responses for a sample of questions.
  - `gera_metricas_csv.py`: Calculates ROUGE-L, BLEU, and Exact Match metrics for model evaluation.
  - `testar_llama.py`: Interactive script to test the fine-tuned and base models.

- **Datasets**
  - `metabolism_absorption_alpaca.json`: Main training dataset in Alpaca format.
  - `negative_examples.txt`: Negative examples for robustness.
  - `metabolism_absorption.xml`: Extracted DrugBank metabolism and absorption data.

## How to Use

1. **Prepare the Data**
   - Run the data extraction and prompt generation scripts in `scripts/preparacao dos dados/`.

2. **Fine-Tune the Model**
   - Use `llama-3.1-8b-unsloth.py` to fine-tune the model with your Hugging Face and Weights & Biases credentials.

3. **Evaluate the Model**
   - Generate responses and metrics using the scripts in `scripts/gerar avaliacoes/`.

## Requirements

- Python 3.11.13
- See [`requirements.txt`](scripts/requirements.txt) for all dependencies.


**Author:** Felipe Rocha Verol\ 
**Institution:** State University of Campinas (UNICAMP)\  
**Contact:** f248552@dac.unicamp.br\
**Year:** 2025