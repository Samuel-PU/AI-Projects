#!/usr/bin/env python
import sys
from fake_news.predict import predict
if len(sys.argv) != 2:
    print("Usage: python classify_url.py <url>")
    sys.exit(1)
url = sys.argv[1]
p = predict(url)
verdict = "🟥 FAKE" if p > 0.5 else "🟩 REAL"
print(f"{verdict}  ({p*100:.1f}% fake probability)")
