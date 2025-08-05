"""Lightweight text cleaning and DistilBERT embedding."""
import re, torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "distilbert-base-uncased"
_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
_bert = AutoModel.from_pretrained(MODEL_NAME)

def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()

@torch.no_grad()
def encode(text: str) -> torch.Tensor:
    text = clean(text)
    toks = _tok(text, return_tensors="pt", truncation=True, max_length=512)
    outs = _bert(**toks)
    # CLS token is first embedding
    return outs.last_hidden_state[:, 0, :].squeeze(0)
