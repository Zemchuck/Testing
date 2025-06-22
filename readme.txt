BANKING77 INTENT CLASSIFIER – MLOPS LAB

This repository contains the code and artefacts created while completing Lab 4 of the AGH MLOps course.  The goal is to fine‑tune a DistilBERT model on the Banking77 dataset, evaluate it, probe the model for weaknesses and generate explanations — all with proper experiment tracking and automation.

CONTENTS

• Core scripts
- train_classifier.py             – fine‑tunes DistilBERT on Banking77 and logs everything to MLflow
- evaluate_model.py               – evaluates the trained model, writes metrics and saves plots
- exploration.py                  – exploratory data analysis (class distribution, text length …)
- captum_explainability.py        – token‑level explanations generated with Captum
- giskard_behavioral_testing.py   – behavioural scan and HTML report via Giskard

• Artefacts & outputs
- model/                          – final fine‑tuned model (🤗 Transformers format)
- results/                        – intermediate training checkpoints
- *.png                           – plots created during EDA / evaluation
- giskard_behavioural_report.html – standalone behavioural‑scan report

• Experiment tracking
- mlruns/                         – local MLflow tracking directory (ignored by Git)

• Project plumbing
- pyproject.toml                  – dependencies and project metadata
- .gitignore                      – ignore rules (Python caches, venv, MLflow artefacts …)
- .venv/                          – local virtual environment (not committed)

QUICK START

Clone the repo and enter it:
git clone <your‑fork‑url>
cd Testing

Create a reproducible environment with uv (or use venv/conda):
uv venv && source .venv/bin/activate
uv pip install -r pyproject.toml

(Option A) Train from scratch:
uv run python train_classifier.py       # produces checkpoints + MLflow logs
(Option B) Skip training and use the ready model stored in ./model

Evaluate and generate visualisations:
uv run python evaluate_model.py
open class_distribution.png
open f1_per_class.png

Explainability (Captum):
uv run python captum_explainability.py

Behavioural testing (Giskard):
uv run python giskard_behavioral_testing.py
open giskard_behavioural_report.html

Notes:
• All scripts auto‑detect CUDA and fall back to CPU.
• Launch  "mlflow ui -p 5000"  to browse experiment runs stored in mlruns/.

KEY RESULTS

Accuracy (test set)  : 94.6 %
Macro F1 (test set)  : 0.946
Per‑class scores     : see f1_per_class.png

BEHAVIOURAL SCAN (GISKARD)

• HTML report: giskard_behavioural_report.html
• 1 medium performance‑bias issue on slice "money" (precision –6.6 %).
• No critical over/under‑confidence or ethical bias findings.

PROJECT SETUP & CONVENTIONS

• Python 3.11, managed with uv for deterministic builds.
• Coding style enforced via ruff + black (optional pre‑commit hooks).
• Experiments tracked with MLflow (data stored in mlruns/, ignored by Git).
• Large artefacts (models, checkpoints) are committed for convenience; configure Git LFS if size becomes an issue.

RE‑TRAINING WITH CUSTOM HYPER‑PARAMETERS

train_classifier.py exposes most Hugging Face TrainingArguments via CLI.  Example:

uv run python train_classifier.py --learning_rate 2e-5 --num_train_epochs 5 --per_device_train_batch_size 16

A new MLflow run will appear automatically.

LICENSE & CITATION

Code © 2025 MLOps Lab, AGH — MIT License.
Banking77 dataset © PolyAI.