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
    AREA_MAP = {
        "Asia": [
            "China",
            "India",
            "Singapore",
            "South Korea",
            "Israel",
            "Japan",
        ],
        "Europe": [
            "Switzerland",
            "France",
            "Germany",
            "United Kingdom",
            "Austria",
            "Sweden",
            "Norway",
            "Netherlands",
            "Ireland",
            "Denmark",
            "Finland",
        ],
        "North America": [
            "United States",
            "Canada",
        ],
        "Australia": [
            "Australia",
        ],
    }

    def map_country_to_area(col_name: str) -> pl.Expr:
        expr = pl.when(pl.lit(False)).then(pl.lit("Other"))

        for area, countries in AREA_MAP.items():
            expr = expr.when(pl.col(col_name).is_in(countries)).then(pl.lit(area))

        return expr.otherwise(pl.lit("Other"))

    # ----------------------------
    # Aggregate company location into areas
    # ----------------------------
    df = df.with_columns(map_country_to_area("company_location").alias("company_area"))
    # ----------------------------
    # Aggregate employee residence into areas
    # ----------------------------
    df = df.with_columns(
        map_country_to_area("employee_residence").alias("residence_area")
    )
    # ----------------------------
    # Same country indicator
    # ----------------------------
    df = df.with_columns(
        (pl.col("company_location") == pl.col("employee_residence")).alias(
            "same_country"
        )
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
