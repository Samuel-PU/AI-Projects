"""Command‑line wrapper around fake_news.train for convenience."""
import argparse, subprocess, sys
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    cmd = [sys.executable, "-m", "fake_news.train", "--epochs", str(args.epochs)]
    subprocess.run(cmd, check=True)
if __name__ == "__main__":
    main()
