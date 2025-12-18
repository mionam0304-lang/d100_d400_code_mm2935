import polars as pl


def create_sample_split(
    df: pl.DataFrame,
    id_column: str = "job_id",
    training_frac: float = 0.8,
) -> pl.DataFrame:
    """
    Deterministic ID-based train/test split.
    """

    return df.with_columns(
        pl.when(
            (pl.col(id_column).cast(pl.Utf8).hash(seed=0) % 100) < training_frac * 100
        )
        .then(pl.lit("train"))
        .otherwise(pl.lit("test"))
        .alias("sample")
    )
