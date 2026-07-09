import json

def load_feature_config(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def infer_disk_model(df):
    if "model" not in df.columns:
        raise ValueError("Column 'model' not found in dataframe.")

    models = df["model"].dropna().astype(str).unique().tolist()

    if len(models) == 0:
        raise ValueError("No valid disk model found in dataframe.")

    if len(models) > 1:
        raise ValueError(f"Multiple disk models found in dataframe: {models}")

    return models[0]


def normalize_feature_names(feature_list, available_columns):
    normalized = []

    for feat in feature_list:
        feat = str(feat).strip()

        candidates = []

        # 已经是完整列名
        if feat in available_columns:
            candidates.append(feat)

        # smart_xxx -> smart_xxx_raw / normalized
        if feat.startswith("smart_"):
            candidates.append(f"{feat}_raw")
            candidates.append(f"{feat}_normalized")

        # 万一 json 里哪天只有数字
        if feat.isdigit():
            candidates.append(f"smart_{feat}_raw")
            candidates.append(f"smart_{feat}_normalized")

        matched = None
        for c in candidates:
            if c in available_columns:
                matched = c
                break

        if matched is not None and matched not in normalized:
            normalized.append(matched)

    return normalized

def get_selected_features(df, json_path, min_non_null_ratio=0.001):
    feature_cfg = load_feature_config(json_path)
    disk_model = infer_disk_model(df)

    if disk_model not in feature_cfg:
        raise KeyError(f"Disk model '{disk_model}' not found in feature config json.")

    raw_feature_list = feature_cfg[disk_model]
    selected_features = normalize_feature_names(raw_feature_list, df.columns.tolist())

    if len(selected_features) == 0:
        raise ValueError(
            f"No valid features matched dataframe columns for disk model '{disk_model}'. "
            f"Raw json features: {raw_feature_list}"
        )

    # 新增：过滤掉几乎全空的 base features
    kept_features = []
    dropped_features = []

    for col in selected_features:
        non_null_ratio = df[col].notna().mean()
        if non_null_ratio >= min_non_null_ratio:
            kept_features.append(col)
        else:
            dropped_features.append((col, non_null_ratio))

    print(f"Detected disk model: {disk_model}")
    print(f"JSON selected features: {raw_feature_list}")
    print(f"Matched dataframe features ({len(selected_features)}): {selected_features}")
    print(f"Kept non-empty features ({len(kept_features)}): {kept_features}")
    print(f"Dropped near-empty features ({len(dropped_features)}): {dropped_features}")

    if len(kept_features) == 0:
        raise ValueError(
            f"All matched features are empty or nearly empty for disk model '{disk_model}'."
        )

    return kept_features
