import polars as pl


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    # ----------------------------
    # 1. Process the dtype of date
    # ----------------------------
    df = df.with_columns(
        [
            pl.col("posting_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            pl.col("application_deadline").str.strptime(
                pl.Date, "%Y-%m-%d", strict=False
            ),
        ]
    )

    # ----------------------------
    # 2. Aggregate country into geographic areas
    # ----------------------------
    df = df.with_columns(
        pl.when(
            pl.col("company_location").is_in(
                ["China", "India", "Japan", "South Korea", "Israel"]
            )
        )
        .then(pl.lit("Asia"))
        .when(
            pl.col("company_location").is_in(
                [
                    "Germany",
                    "United Kingdom",
                    "Austria",
                    "Switzerland",
                    "Norway",
                    "Finland",
                    "Ireland",
                ]
            )
        )
        .then(pl.lit("Europe"))
        .when(pl.col("company_location") == "United States")
        .then(pl.lit("US"))
        .when(pl.col("company_location") == "Australia")
        .then(pl.lit("Australia"))
        .otherwise(pl.lit("Other"))  # Please reconsider the clarification if it happens
        .alias("company_area")
    )

    # ----------------------------
    # 3. Aggregate industry into broader groups
    # ----------------------------
    df = df.with_columns(
        pl.when(pl.col("industry").is_in(["Technology", "Telecommunications"]))
        .then(pl.lit("Tech & Telecom"))
        .when(pl.col("industry").is_in(["Finance", "Real Estate"]))
        .then(pl.lit("Finance & Real Estate"))
        .when(pl.col("industry").is_in(["Healthcare", "Education", "Government"]))
        .then(pl.lit("Public & Social"))
        .when(pl.col("industry").is_in(["Manufacturing", "Automotive"]))
        .then(pl.lit("Manufacturing"))
        .when(pl.col("industry") == "Energy")
        .then(pl.lit("Energy"))
        .when(pl.col("industry").is_in(["Transportation", "Retail"]))
        .then(pl.lit("Consumer & Transport"))
        .when(pl.col("industry").is_in(["Media", "Gaming"]))
        .then(pl.lit("Media & Entertainment"))
        .when(pl.col("industry") == "Consulting")
        .then(pl.lit("Consulting"))
        .otherwise(pl.lit("Other"))  # Please reconsider the clarification if it happens
        .alias("industry_group")
    )
    # ----------------------------
    # 4. Split required skills into a list
    # ----------------------------
    df = df.with_columns(
        pl.col("required_skills")
        .str.split(",")  # Split comma-separated skills into a list
        .list.eval(
            pl.element().str.strip_chars()
        )  # Remove leading/trailing spaces from each skill
        .alias("skills_list")  # List of individual skills
    )

    # ----------------------------
    # 5. Create summary skill feature
    # ----------------------------
    df = df.with_columns(
        pl.col("skills_list")
        .list.len()
        .alias("num_skills")  # Number of listed skills as a proxy for job complexity
    )

    return df
