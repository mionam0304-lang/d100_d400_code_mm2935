import matplotlib.pyplot as plt
import polars as pl


def plot_salary_hist(df: pl.DataFrame, log_scale: bool = False, bins: int = 50) -> None:

    s = df["salary_usd"]

    plt.figure()
    plt.hist(s, bins=50, edgecolor="black")
    plt.xlabel("Salary USD")
    plt.ylabel("Number of Jobs")
    plt.title("Distribution of Annual Salaries in the AI Market (USD)")
    plt.show()


def plot_group_median_salary(
    df: pl.DataFrame,
    group_col: str,
    order: list[str] | None = None,
    min_n: int = 20,
) -> None:
    g = (
        df.group_by(group_col)
        .agg(
            pl.col("salary_usd").median().alias("median_salary"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= min_n)  # groups with small samples are excluded
    )

    if order is not None:
        # if a group is not included order, it will be excluded
        g = (
            g.filter(pl.col(group_col).is_in(order))
            .with_columns(
                pl.col(group_col).replace(order, list(range(len(order)))).alias("_ord")
            )
            .sort("_ord")
            .drop("_ord")
        )
    else:
        g = g.sort("median_salary")

    plt.figure()
    plt.bar(
        [str(x) for x in g[group_col].to_list()],
        g["median_salary"].to_list(),
        edgecolor="black",
    )
    plt.xlabel(group_col)
    plt.xticks(rotation=90)
    plt.ylabel("Median Salary (USD)")
    plt.title(f"Median Salary by {group_col} (n ≥ {min_n})")
    plt.tight_layout()
    plt.show()


def plot_salary_scatter(df, corr_cols) -> None:

    y = df["salary_usd"].to_numpy()

    for xcol in corr_cols:
        x = df[xcol].to_numpy()

        plt.figure()
        plt.scatter(x, y, alpha=0.3, s=10)
        plt.xlabel(xcol)
        plt.ylabel("salary_usd")
        plt.title(f"salary_usd vs {xcol}")
        plt.tight_layout()
        plt.show()
