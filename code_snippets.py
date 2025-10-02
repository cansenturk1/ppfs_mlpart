def compute_ranks(data_parts, label_col):
    ranks = []
    for part in data_parts:
        cols = [c for c in part.columns if c != label_col]
        if cols:
            # Use method='average' to handle ties, will be converted to int later
            ranks.append(part[cols].rank(method='average').astype(int))
        else:
            ranks.append(pd.DataFrame(index=part.index))
    return ranks


def compute_spearman_correlation(ranked_parts, correlation_target):
    # Find which part (if any) has target ranks; fallback: last part
    target_rank = None
    for rp in ranked_parts:
        if correlation_target in rp.columns:
            target_rank = rp[correlation_target].astype(int)
            break
    # If target ranks are not stored separately, cannot proceed
    if target_rank is None:
        # Assume target in last original part (not ranked); cannot compute -> return empty
        return {}
    
    res = {}
    for rp in ranked_parts:
        for col in rp.columns:
            if col == correlation_target:
                continue
            
            feature_rank = rp[col].astype(int)
            
            # ρ = cov(R_x, R_y) / (std(R_x) * std(R_y))
            mean_feature = feature_rank.mean()
            mean_target = target_rank.mean()
            
            # Covariance
            cov = ((feature_rank - mean_feature) * (target_rank - mean_target)).sum()
            
            # Standard deviations
            std_feature = ((feature_rank - mean_feature) ** 2).sum() ** 0.5
            std_target = ((target_rank - mean_target) ** 2).sum() ** 0.5
            
            # Pearson correlation on ranks (Spearman correlation)
            if std_feature > 0 and std_target > 0:
                res[col] = cov / (std_feature * std_target)
            else:
                res[col] = 0.0  # Handle case where all ranks are identical

    return res