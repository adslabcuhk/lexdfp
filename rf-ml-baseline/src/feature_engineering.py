import pandas as pd

def add_future_failure_label(df, window=7):
    df = df.sort_values(['serial_number', 'date'])

    df['failure_future_7d'] = (
        df.groupby('serial_number')['failure']
        .transform(lambda x: x[::-1].rolling(window, min_periods=1).max()[::-1])
    )

    return df


def build_features(df, features, window=7):
    df = df.sort_values(["serial_number", "date"]).copy()

    for col in features:
        grp = df.groupby("serial_number")[col]

        df[f"{col}_mean"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_max"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        df[f"{col}_delta"] = grp.transform(
            lambda x: x.diff(window).fillna(0)
        )

    return df


def get_feature_columns(df):
    return [c for c in df.columns if ("mean" in c or "max" in c or "delta" in c)]
