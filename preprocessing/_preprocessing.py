import polars as pl


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col("posting_date").str.strptime(pl.Date, "%Y/%m/%d"),
            pl.col("application_deadline").str.strptime(pl.Date, "%Y/%m/%d"),
        ]
    )
