import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif   
import gmpy2

DATASETS = {
    'diabetes': ('diabetes_kmeans.csv', 'Outcome', [])
}

def load_csv_data(filename, target_col, drop_cols=None):
    df = pd.read_csv(filename, delimiter=',')
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    df = df.apply(lambda s: s.astype(np.int64) if s.name != target_col else s)
    return np.array_split(df, 3, axis=1)

def load_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    file, target, drops = DATASETS[name]
    return load_csv_data(file, target, drops), target


def compute_ranks(data_parts, target_col):
    ranks = []
    for part in data_parts:
        cols = [c for c in part.columns if c != target_col]
        if cols:
            ranks.append(part[cols].rank().astype(int))
        else:
            ranks.append(pd.DataFrame(index=part.index))
    return ranks

def compute_spearman_correlation(ranked_parts, target_col):
    # Find which part (if any) has target ranks; fallback: last part
    target_rank = None
    for rp in ranked_parts:
        if target_col in rp.columns:
            target_rank = rp[target_col].astype(int)
            break
    # If target ranks are not stored separately, cannot proceed
    if target_rank is None:
        # Assume target in last original part (not ranked); cannot compute -> return empty
        return {}
    n = len(target_rank)
    denom = n * (n**2 - 1)
    res = {}
    for rp in ranked_parts:
        for col in rp.columns:
            if col == target_col:
                continue
            fr = rp[col].astype(int)
            d2 = (fr - target_rank).apply(lambda d: d * d).sum()
            res[col] = 1 - (6 * d2) / denom
    return res


    