from __future__ import annotations

from pathlib import Path

from data import load_data
from preprocessing import preprocess


def main() -> None:
    df = load_data()
    df_clean = preprocess(df)

    out_path = Path("data/jobs_cleaned.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.write_parquet(out_path)


if __name__ == "__main__":
    main()
