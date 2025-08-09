# build_dataset.py – scrape the 4 FakeNewsNet CSVs ➜ data/processed/train.csv
import glob, csv, pathlib, pandas as pd, validators
from tqdm import tqdm                                   # progress-bar
from fake_news.fetch import scrape

MAX_PER_FILE = 600          # tweak for speed / quota

def pick_url_column(df):
    """Return the first column whose name ends with 'url' (case-insensitive)."""
    for col in df.columns:
        if col.lower().endswith("url"):
            return col
    raise ValueError("No *url column found in CSV")

def normalise(url: str) -> str:
    url = str(url).strip()
    if url and not url.lower().startswith(("http://", "https://")):
        url = "http://" + url          # add scheme if missing
    return url

def main():
    rows = []
    csv_paths = sorted(glob.glob("data/raw/*.csv"))
    for csv_path in tqdm(csv_paths, desc="CSV files", unit="file"):
        label = 1 if "fake" in csv_path.lower() else 0   # 1 = fake, 0 = real
        df = pd.read_csv(csv_path, dtype=str, na_filter=False)
        url_col = pick_url_column(df)

        # use tqdm on URLs too
        for raw_url in tqdm(
            df[url_col].tolist()[:MAX_PER_FILE],
            desc=f" scraping {pathlib.Path(csv_path).name}",
            leave=False,
        ):
            url = normalise(raw_url)
            if not validators.url(url):
                continue
            try:
                text = scrape(url)[:2000]
                if len(text) > 80:
                    rows.append({"text": text, "label": label})
            except Exception:
                pass  # skip dead links

    out = pathlib.Path("data/processed/train.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n✅  Wrote {out} with {len(rows)} rows")

if __name__ == "__main__":
    main()
