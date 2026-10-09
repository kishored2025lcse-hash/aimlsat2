from pathlib import Path

import pandas as pd


DATA_FILE = Path(__file__).with_name("bank_marketing_part1_Data.csv")


def main():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    if df.empty:
        raise ValueError(f"Dataset is empty: {DATA_FILE}")

    print(f"Dataset: {DATA_FILE.name}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        print("\nNo numeric columns available for statistical analysis.")
        return

    print("\nNumeric summary:")
    print(numeric_df.describe().round(3).to_string())

    if len(numeric_df.columns) > 1:
        print("\nCorrelation matrix:")
        print(numeric_df.corr().round(3).to_string())

    if "y" not in df.columns:
        print(
            "\nNo 'y' target column was found. Classification/model training "
            "cannot run until a target label is added."
        )


if __name__ == "__main__":
    main()
