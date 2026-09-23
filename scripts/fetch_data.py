"""Download MovieLens 100k into data/.

    python scripts/fetch_data.py

The competition this system was built for used a private Amazon dataset that
cannot be redistributed, so the public version runs on MovieLens, which has the
same shape: timestamped interactions, and items with a title and categories to
build content features from.
"""
import argparse
import io
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "data"))
    args = parser.parse_args()

    target = Path(args.out)
    if (target / "ml-100k" / "u.data").exists():
        print(f"{target / 'ml-100k'} already there. Nothing to do.")
        return

    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        # Without certifi the system store is used, which on some machines
        # does not carry the chain this host presents.
        context = ssl.create_default_context()

    print(f"Downloading {URL} ...")
    try:
        raw = urllib.request.urlopen(URL, timeout=120, context=context).read()
    except Exception as exc:
        raise SystemExit(
            f"Download failed: {exc}\n\n"
            "If this is a certificate error, `pip install certifi` and try "
            "again, or fetch the zip by hand and unpack it into data/."
        )

    target.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(raw)).extractall(target)
    print(f"Ready: {target / 'ml-100k'}, {len(raw) / 1e6:.1f} MB downloaded.")


if __name__ == "__main__":
    main()
