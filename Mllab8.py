# Mllab8.py
# K-Modes clustering on Adult Income (categorical subset)
# Author: You :)
# ---------------------------------------------
# Features used: workclass, education, marital-status, occupation
# Outputs:
#   - kmodes_cost.png (cost vs K plot)
#   - kmodes_cluster_sizes.png (bar chart of cluster sizes)
#   - adult_kmodes_clusters.csv (data with cluster labels)
# ---------------------------------------------

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from kmodes.kmodes import KModes
except ImportError as e:
    raise SystemExit(
        "kmodes package not found. Install with:\n\n"
        "    pip install kmodes\n"
    ) from e


# ---------------------------------------------
# CONFIG
# ---------------------------------------------
# Set your file path here. If not found, the script falls back to 'adult.csv' in the current directory.
DEFAULT_CSV_PATH = r"D:\c\.vscode\ML LAB\adult.csv"  # <-- change to your path if needed

# Columns we want (categorical)
REQUIRED_CATEGORICAL = ["workclass", "education", "marital-status", "occupation"]


# ---------------------------------------------
# 1) Load & preprocess dataset
# ---------------------------------------------
def _maybe_assign_adult_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the dataset has no headers (common for UCI 'adult.data'), assign standard Adult headers.
    """
    # UCI Adult has 15 columns (including 'income' label)
    if list(df.columns) == list(range(df.shape[1])) and df.shape[1] in (14, 15):
        headers = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country"
        ]
        if df.shape[1] == 15:
            headers.append("income")
        df.columns = headers
    # Trim whitespace from column names if any
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_adult_csv(path: str) -> pd.DataFrame:
    """
    Loads the Adult Income dataset and returns a cleaned DataFrame
    containing only categorical columns: workclass, education,
    marital-status, occupation.
    """
    # Resolve path: try given, else try 'adult.csv' in CWD
    resolved = path if os.path.exists(path) else "adult.csv"
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"Could not find dataset.\nTried:\n  - {path}\n  - {os.path.abspath('adult.csv')}\n"
            "Place 'adult.csv' in the working directory or update DEFAULT_CSV_PATH."
        )

    # Read CSV with forgiving parsing
    df = pd.read_csv(
        resolved,
        header=0,                 # will be corrected if no headers present
        na_values=["?", " ?", "? "],
        skipinitialspace=True
    )

    # If the file had no header row, assign standard headers
    df = _maybe_assign_adult_headers(df)

    # Build a lowercase lookup map to be robust to case/hyphen variants
    lower_map = {c.lower(): c for c in df.columns}

    # Ensure required columns exist
    missing = [c for c in REQUIRED_CATEGORICAL if c not in lower_map]
    if missing:
        # Try a few common variants (e.g., underscores instead of hyphens)
        alt_map = {}
        for name in df.columns:
            # normalized: lowercase + hyphens to underscores, strip spaces
            norm = str(name).strip().lower().replace("-", "_")
            alt_map[norm] = name

        need = []
        for req in REQUIRED_CATEGORICAL:
            norm_req = req.replace("-", "_")
            if req not in lower_map and norm_req in alt_map:
                lower_map[req] = alt_map[norm_req]
            elif req not in lower_map:
                need.append(req)

        if need:
            raise KeyError(
                "Required columns not found.\n"
                f"Expected: {REQUIRED_CATEGORICAL}\n"
                f"Found: {list(df.columns)}"
            )

    sel_cols = [lower_map[c] for c in REQUIRED_CATEGORICAL]
    cat_df = df[sel_cols].copy()
    cat_df.columns = REQUIRED_CATEGORICAL  # unify names

    # Clean entries: strip whitespace, convert '?' and 'nan' to NaN, then fill
    for col in cat_df.columns:
        cat_df[col] = (
            cat_df[col]
            .astype(str)
            .str.strip()
            .replace({"?": np.nan, "nan": np.nan})
        )

    cat_df = cat_df.fillna("Unknown")

    # Optional for clarity
    for col in cat_df.columns:
        cat_df[col] = cat_df[col].astype("category")

    return cat_df


# ---------------------------------------------
# 2) Fit K-Modes and compute cost over K range
# ---------------------------------------------
def kmodes_cost_sweep(
    cat_df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
    init: str = "Cao",
    n_init: int = 5,
    random_state: int = 42,
    verbose: int = 0
):
    """
    Runs K-Modes for K in [k_min, k_max], returns list of (K, cost, model).
    """
    results = []
    X = cat_df.astype(str).values  # K-Modes expects array-like of strings

    for k in range(k_min, k_max + 1):
        km = KModes(
            n_clusters=k,
            init=init,        # 'Cao' is stable; 'Huang' is also common
            n_init=n_init,    # multiple restarts
            verbose=verbose,
            random_state=random_state
        )
        _ = km.fit_predict(X)
        results.append((k, km.cost_, km))
        print(f"K={k:2d} | cost={km.cost_:,.0f}")

    return results


# ---------------------------------------------
# 3) Plot cost to choose optimal K (Elbow plot)
# ---------------------------------------------
def plot_cost(results, title="K-Modes Cost vs Number of Clusters", save_path="kmodes_cost.png", show=True):
    ks = [r[0] for r in results]
    costs = [r[1] for r in results]

    plt.figure(figsize=(9, 5))
    plt.plot(ks, costs, marker="o", linestyle="-", color="#1f77b4")
    plt.xticks(ks)
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Cost (sum of mismatches)")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Cost plot saved to: {os.path.abspath(save_path)}")
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------
# 4) Fit final model and inspect modes
# ---------------------------------------------
def fit_final_kmodes(
    cat_df: pd.DataFrame,
    k: int,
    init="Cao",
    n_init=5,
    random_state=42,
    verbose=0
):
    X = cat_df.astype(str).values
    km = KModes(
        n_clusters=k,
        init=init,
        n_init=n_init,
        verbose=verbose,
        random_state=random_state
    )
    labels = km.fit_predict(X)

    sizes = pd.Series(labels).value_counts().sort_index()
    print("\nCluster sizes:")
    print(sizes)

    modes = pd.DataFrame(km.cluster_centroids_, columns=cat_df.columns)
    print("\nCluster modes (centroids):")
    print(modes)

    clustered = cat_df.copy()
    clustered["cluster"] = labels
    return km, clustered


# ---------------------------------------------
# 5) Optional: visualize cluster sizes
# ---------------------------------------------
def plot_cluster_sizes(clustered: pd.DataFrame, save_path="kmodes_cluster_sizes.png", show=True):
    sizes = clustered["cluster"].value_counts().sort_index()

    plt.figure(figsize=(8, 4))
    sizes.plot(kind="bar", color="#ff7f0e")
    plt.xlabel("Cluster")
    plt.ylabel("Number of records")
    plt.title("Cluster sizes (K-Modes)")
    plt.grid(axis="y", alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Cluster sizes plot saved to: {os.path.abspath(save_path)}")
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------
# 6) Simple elbow heuristic (optional)
# ---------------------------------------------
def suggest_k_from_elbow(results):
    """
    Naive elbow picker: choose K after the largest drop in cost.
    """
    ks = np.array([r[0] for r in results])
    costs = np.array([r[1] for r in results], dtype=float)
    if len(costs) < 2:
        return ks[0]
    deltas = np.diff(costs)  # negative values; the most negative is the largest drop
    elbow_idx = int(np.argmin(deltas))
    suggested_k = int(ks[elbow_idx + 1])
    return suggested_k


# ---------------------------------------------
# Main
# ---------------------------------------------
def main(csv_path: str = DEFAULT_CSV_PATH, k_min: int = 2, k_max: int = 10, final_k: int | None = None, show_plots: bool = True):
    # 1) Load & preprocess
    cat_df = load_adult_csv(csv_path)
    print(f"Loaded {len(cat_df)} rows with columns: {list(cat_df.columns)}")

    # 2) Sweep K to compute costs
    results = kmodes_cost_sweep(cat_df, k_min=k_min, k_max=k_max, init="Cao", n_init=5, random_state=42)

    # 3) Plot cost (elbow method)
    plot_cost(results, title="K-Modes (Cao init) Cost vs K", save_path="kmodes_cost.png", show=show_plots)

    # 4) Choose final K (user-provided or heuristic)
    if final_k is None:
        suggested_k = suggest_k_from_elbow(results)
        print(f"\n[Heuristic] Suggested K (elbow): {suggested_k}. Override by passing final_k=... if needed.")
        final_k = suggested_k
    else:
        print(f"\nUsing user-provided K: {final_k}")

    # 5) Fit final model and save labeled data
    _, clustered = fit_final_kmodes(cat_df, k=final_k, init="Cao", n_init=5, random_state=42)
    out_csv = "adult_kmodes_clusters.csv"
    clustered.to_csv(out_csv, index=False)
    print(f"✅ Clustered data saved to '{os.path.abspath(out_csv)}'.")

    # 6) Plot cluster sizes
    plot_cluster_sizes(clustered, save_path="kmodes_cluster_sizes.png", show=show_plots)


if __name__ == "__main__":
    # You can run with defaults (uses DEFAULT_CSV_PATH), or pass CLI arguments:
    #   python Mllab8.py "D:\c\.vscode\ML LAB\adult.csv" 2 12 6
    #                                     ^path         ^k_min ^k_max ^final_k (optional)
    args = sys.argv[1:]
    csv_path = args[0] if len(args) > 0 else DEFAULT_CSV_PATH
    k_min = int(args[1]) if len(args) > 1 else 2
    k_max = int(args[2]) if len(args) > 2 else 10
    final_k = int(args[3]) if len(args) > 3 else None
    main(csv_path=csv_path, k_min=k_min, k_max=k_max, final_k=final_k, show_plots=True)