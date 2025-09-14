import numpy as np
import pandas as pd
import gmpy2
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.feature_selection import mutual_info_classif

gmpy2.get_context().precision = 100

# =====================================================================
# Configuration
# =====================================================================

DATASETS = {
    'beans': ('beans_kmeans.csv', 'Class', []),
    'diabetes': ('diabetes_kmeans.csv', 'Outcome', []),
    'divorce': ('divorce.csv', 'Class', []),
    'parkinsons': ('parkinsons_kmeans.csv', 'status', ['name']),
    'rice': ('rice_binned_kmeans.csv', 'Class', []),
    'wdbc': ('wdbc_binned_kmeans.csv', 'Diagnosis', ['ID'])
}

# =====================================================================
# Utilities
# =====================================================================

def safe_int(x):
    return int(x) if not isinstance(x, int) else x

def log2_safe(p: gmpy2.mpfr):
    return gmpy2.log2(p) if p > 0 else gmpy2.mpfr(0)

def paillier_add_accumulate(encrypted_values, n_squared):
    """Multiply ciphertexts mod n^2 (homomorphic addition)."""
    acc = 1
    for c in encrypted_values:
        acc = (acc * int(c)) % n_squared
    return acc

# =====================================================================
# Data Loading
# =====================================================================

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

# =====================================================================
# Feature Selection Helpers
# =====================================================================

def get_min_mutual_info_feature(data_parts, target_col):
    for part in data_parts:
        if target_col in part.columns:
            feature_cols = [c for c in part.columns if c != target_col]
            if not feature_cols:
                raise ValueError("No feature columns in target-containing part.")
            X = part[feature_cols].values
            y = part[target_col].values
            mi_scores = mutual_info_classif(X, y, discrete_features=True)
            return min(dict(zip(feature_cols, mi_scores)), key=lambda k: mi_scores[feature_cols.index(k)])
    raise ValueError(f"Target column '{target_col}' not found.")

# =====================================================================
# Ranking & Encryption
# =====================================================================

def compute_ranks(data_parts, target_col):
    ranks = []
    for part in data_parts:
        cols = [c for c in part.columns if c != target_col]
        if cols:
            ranks.append(part[cols].rank().astype(int))
        else:
            ranks.append(pd.DataFrame(index=part.index))
    return ranks

# =====================================================================
# Spearman (Plain)
# =====================================================================

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



# =====================================================================
# Mutual Information (Plain)
# =====================================================================

def compute_mutual_information(data_parts, target_col):
    # Locate target column
    target_series = None
    for p in data_parts:
        if target_col in p.columns:
            target_series = p[target_col]
            break
    if target_series is None:
        return {}
    n = len(target_series)
    unique_Y = pd.unique(target_series)
    # Precompute Y indicators & entropy H(Y)
    Y_ind = {y: (target_series == y).astype(int).to_numpy() for y in unique_Y}
    H_y = gmpy2.mpfr(0)
    for y in unique_Y:
        py = gmpy2.mpfr(int(Y_ind[y].sum())) / gmpy2.mpfr(n)
        if py > 0:
            H_y -= py * log2_safe(py)
    mi = {}
    for part in data_parts:
        for col in part.columns:
            if col == target_col:
                continue
            X = part[col]
            unique_X = pd.unique(X)
            X_ind = {xv: (X == xv).astype(int).to_numpy() for xv in unique_X}
            count_x = {xv: X_ind[xv].sum() for xv in unique_X}
            H_y_given_x = gmpy2.mpfr(0)
            for xv in unique_X:
                cx = count_x[xv]
                if cx == 0:
                    continue
                for y in unique_Y:
                    c_xy = int((X_ind[xv] & Y_ind[y]).sum())
                    if c_xy == 0:
                        continue
                    p_xy = gmpy2.mpfr(c_xy) / gmpy2.mpfr(n)
                    p_y_given_x = gmpy2.mpfr(c_xy) / cx
                    H_y_given_x -= p_xy * log2_safe(p_y_given_x)
            mi[col] = float(H_y - H_y_given_x)
    print(mi)
    return mi

# =====================================================================
# Reporting / Plotting
# =====================================================================

def print_spearman_results(target_feature, plain_corrs, enc_corrs):
    print(f"Target Feature (Spearman): {target_feature}")
    print(f"{'Feature':<25}{'Plain':<22}{'Encrypted'}")
    for f, v in plain_corrs.items():
        print(f"{f:<25}{v:<22}{enc_corrs.get(f, None)}")

def plot_elbow(values_dict, title, ylabel):
    if not values_dict:
        return
    sorted_items = sorted(values_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    labels, scores = zip(*sorted_items)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(scores)), [abs(s) for s in scores], marker='o')
    plt.xticks(range(len(labels)), labels, rotation=50, ha='right')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Features")
    plt.tight_layout()
    plt.show()

def print_mutual_info_results(target_col, plain_mi, enc_mi):
    print(f"\nMutual Information (target={target_col})")
    print(f"{'Feature':<25}{'Plain':<22}{'Encrypted'}")
    for f, v in plain_mi.items():
        print(f"{f:<25}{v:<22}{enc_mi.get(f, None)}")

def plot_mi(mi_dict, title):
    if not mi_dict:
        return
    sorted_items = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
    labels, scores = zip(*sorted_items)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(scores)), scores, marker='o')
    plt.xticks(range(len(labels)), labels, rotation=50, ha='right')
    plt.title(title)
    plt.ylabel("Mutual Information")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.show()

# =====================================================================
# Main
# =====================================================================

def main(dataset_name='diabetes'):
    data_parts, target_col = load_dataset(dataset_name)

    target_feature = get_min_mutual_info_feature(data_parts, target_col)
    ranked_parts = compute_ranks(data_parts, target_col)


    # Spearman
    plain_spearman = compute_spearman_correlation(ranked_parts, target_feature)


    # Mutual Information
    plain_mi = compute_mutual_information(data_parts, target_col)

if __name__ == "__main__":
    main('diabetes')