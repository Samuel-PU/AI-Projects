FakeNewsNet‑Mini

Lightweight, notebook‑free demo that assigns a first‑pass credibility score to any news article. Runs in under one second on a standard laptop, no GPUs, no heavy libraries. Model artefact is under 4 MB.

Quick start

Create and activate a virtual environment.

pip install -r requirements.txt

python build_dataset.py   – one‑time scrape of FakeNewsNet into data/processed/train.csv

python -m fake_news.train – trains the model and saves models/tfidf_logreg_calibrated.pkl

python fake_news\classify_url.py <url> – returns probability an article is fake

python classify_text.py – paste headline and optional body for an instant score

Pipeline summary

• Word n‑grams (1‑2) TF‑IDF, 20 000 features
• Character n‑grams (3‑5) TF‑IDF, 7 000 features
• Metadata: article length and one‑hot publisher domain
• Class‑balanced logistic regression with sigmoid probability calibration

Evaluation on a 20 % hold‑out set

• ROC‑AUC 0.85
• Accuracy 0.77
• Precision (fake) 0.81
• Recall   (fake) 0.71

Repository layout

fake‑newsnet/
build_dataset.py        scrape → CSV
classify_text.py        interactive headline/body
requirements.txt
README.md               this file
fake_news/              importable package
fetch.py              robust scraper
helpers.py            small feature selectors
predict.py            inference utilities
train.py              training + plotting
classify_url.py       URL classifier CLI
data/                   raw & processed data (not tracked)
models/                 trained pipelines (not tracked)

Next steps

• Add sentiment signal
• Test lightweight transformer embeddings (ONNX)
• Integrate SHAP explanations

Licence

MIT licence – free for personal and commercial use. Attribution welcomed.
