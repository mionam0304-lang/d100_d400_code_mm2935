from pathlib import Path

import kagglehub
import polars as pl


def load_data() -> pl.DataFrame:
    file_path = "ai_job_dataset.csv"

    dataset_dir = kagglehub.dataset_download(
        "bismasajjad/global-ai-job-market-and-salary-trends-2025"
    )

    csv_path = Path(dataset_dir) / file_path

    df = pl.read_csv(
        csv_path,
        encoding="latin1",
    )

    return df
