import pandas as pd
from pathlib import Path


def load_splits(base_path):
    base = Path(base_path)

    train_path = base / "train.csv"
    val_path = base / "val.csv"
    test_path = base / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing file: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing file: {test_path}")

    train_df = pd.read_csv(
        train_path,
        low_memory=False,
        na_values=["\\N", "NULL", "null", "None", ""],
    )
    val_df = pd.read_csv(
        val_path,
        low_memory=False,
        na_values=["\\N", "NULL", "null", "None", ""],
    )
    test_df = pd.read_csv(
        test_path,
        low_memory=False,
        na_values=["\\N", "NULL", "null", "None", ""],
    )

    required_cols = ["serial_number", "date", "model"]
    for split_name, df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {split_name}.csv: {missing}")

        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(["serial_number", "date"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df


def load_single_csv(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(
        csv_path,
        low_memory=False,
        na_values=["\\N", "NULL", "null", "None", ""],
    )

    required_cols = ["serial_number", "date", "model"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in local csv: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["serial_number", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
