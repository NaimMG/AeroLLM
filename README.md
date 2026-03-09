# 🛩️ AeroLLM — Fine-tuning LLM sur données ASRS

LLM spécialisé en sécurité aéronautique, fine-tuné avec QLoRA sur 2000 rapports NASA ASRS.

[![HuggingFace](https://img.shields.io/badge/🤗%20Model-Chasston/aerollm--phi3.5--mini--asrs-blue)](https://huggingface.co/Chasston/aerollm-phi3.5-mini-asrs)
[![GitHub](https://img.shields.io/badge/GitHub-NaimMG/AeroLLM-black)](https://github.com/NaimMG/AeroLLM)

## 🎯 Objectif

Analyser automatiquement des rapports d'incidents aéronautiques ASRS (NASA Aviation Safety Reporting System) : causes, facteurs contributifs, recommandations de sécurité — comme un expert FAA/EASA.

## 🧠 Stack

| Composant | Détail |
|-----------|--------|
| Modèle base | microsoft/Phi-3.5-mini-instruct (3.8B) |
| Technique | QLoRA 4-bit (nf4) + LoRA adapters |
| Dataset | 2000 rapports NASA ASRS |
| GPU | Google Colab T4 (15GB VRAM) |
| Durée training | ~60 minutes |
| Framework | TRL 0.29 + PEFT + Transformers |

## 📊 Résultats du fine-tuning

| Step | Training Loss | Validation Loss |
|------|-------------|-----------------|
| 100  | 6.20        | 5.85            |
| 200  | 4.71        | 4.51            |
| 300  | 4.21        | 4.17            |
| 400  | 4.15        | 4.08            |

Loss divisée par 1.5 en 2 epochs — le modèle a bien appris le jargon ASRS.

## 🚀 Utilisation rapide
```python
# Installation
pip install "peft==0.15.2" "bitsandbytes==0.45.5" "transformers==4.44.0" "accelerate==0.34.0"
```
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = 'microsoft/Phi-3.5-mini-instruct'
ADAPTER_ID = 'Chasston/aerollm-phi3.5-mini-asrs'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base, ADAPTER_ID)
model.eval()

prompt = (
    '<|system|>\n'
    'Tu es un expert en securite aeronautique FAA/EASA. '
    'Analyse ce rapport ASRS.<|end|>\n'
    '<|user|>\n'
    'Aeronef : B737-800\n'
    'Phase de vol : Final Approach\n'
    'Rapport : GPWS terrain warning at 500ft AGL. Go-around executed. Cause: incorrect QNH.'
    '<|end|>\n'
    '<|assistant|>\n'
)

inputs = tokenizer(prompt, return_tensors='pt')
input_ids = inputs['input_ids'].to(next(base.parameters()).device)
attention_mask = inputs['attention_mask'].to(next(base.parameters()).device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

## 📁 Structure du projet
```
AeroLLM/
├── data/
│   └── processed/
│       └── train.json        # 2000 exemples ASRS formatés
├── src/
│   └── prepare_data.py       # Pipeline préparation dataset
├── notebooks/
│   └── train_colab.ipynb     # Notebook QLoRA complet
├── requirements.txt
└── .env.example
```

## 🔁 Reproduire le fine-tuning

1. Clone le repo et prépare le dataset :
```bash
git clone https://github.com/NaimMG/AeroLLM.git
cd AeroLLM
pip install -r requirements.txt
python src/prepare_data.py
```

2. Ouvre `notebooks/train_colab.ipynb` sur Google Colab avec un GPU T4

3. Lance les cellules dans l'ordre — le modèle se sauvegarde automatiquement sur HuggingFace Hub toutes les 100 steps

## 📦 Dataset

- **Source :** `elihoole/asrs-aviation-reports` (38 655 rapports NASA ASRS)
- **Format :** Alpaca instruction/input/output
- **Taille :** 2000 exemples, split 90/10 train/eval
- **Colonnes :** narrative, synopsis, anomalie, facteurs contributifs, phase de vol, facteurs humains

## ⚙️ Config LoRA
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    lora_dropout=0.05,
    task_type='CAUSAL_LM'
)
```