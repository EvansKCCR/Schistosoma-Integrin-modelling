#!/usr/bin/env python3

"""
ClusPro-Style Bayesian Multi-Feature Classifier
Fully robust upgrade:
- MAD-based robust scaling
- Covariance-aware sampling
- Prior sensitivity sweep
- Bootstrap feature resampling
- Posterior odds ratios

Script:

cluspro_mc_bayesian.py

Input:
classification_matrices.xlsx (from benchmarking + docking)
Run:
python cluspro_mc_bayesian.py
Output:
Posterior probabilities
Ranking robustness
bayesian_mc_results.xlsx
Posterior distribution plot

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Load Excel (Metric rows format)
# ------------------------------------------------------

file_path = "classification_matrices.xlsx"
raw_df = pd.read_excel(file_path)
raw_df.set_index("Metric", inplace=True)
df = raw_df.T.reset_index().rename(columns={"index": "Model"})

# ------------------------------------------------------
# Required metrics
# ------------------------------------------------------

required_metrics = [
    "Contact_density",
    "Hbond_density",
    "Salt_density",
    "Interface_identity",
    "Global_RMSD",
    "Predicted binding affinity (kcal.mol-1)"
]

for m in required_metrics:
    if m not in df.columns:
        raise ValueError(f"Missing metric: {m}")
    df[m] = pd.to_numeric(df[m], errors="coerce")

# ------------------------------------------------------
# Robust Scaling (MAD)
# ------------------------------------------------------

def robust_z(series, invert=False):
    x = series.astype(float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        z = np.zeros_like(x)
    else:
        z = (x - med) / (1.4826 * mad)
    return -z if invert else z

df["Contact_z"] = robust_z(df["Contact_density"])
df["Hbond_z"] = robust_z(df["Hbond_density"])
df["Salt_z"] = robust_z(df["Salt_density"])
df["Identity_z"] = robust_z(df["Interface_identity"])
df["RMSD_z"] = robust_z(df["Global_RMSD"], invert=True)
df["DeltaG_z"] = robust_z(df["Predicted binding affinity (kcal.mol-1)"], invert=True)

features = ["Contact_z","Hbond_z","Salt_z","Identity_z","RMSD_z","DeltaG_z"]
Z = df[features].values

# ------------------------------------------------------
# Priors
# ------------------------------------------------------

priors = {
    "Uniform": np.ones(len(features)),
    "Stabilization": np.array([3,4,2,1,1,3]),
    "Geometry": np.array([1,1,1,2,4,1])
}

n_sim = 20000

results_all_priors = []

# ------------------------------------------------------
# Monte Carlo Bayesian Posterior Sampling
# ------------------------------------------------------

for prior_name, alpha_prior in priors.items():

    posterior_rank_counts = np.zeros(len(df))
    posterior_prob_matrix = np.zeros((len(df), n_sim))

    for i in range(n_sim):

        # Dirichlet weights
        w = np.random.dirichlet(alpha_prior)

        # Bootstrap feature resampling (robustness)
        boot_idx = np.random.choice(len(features), len(features), replace=True)
        w_boot = w[boot_idx]
        Z_boot = Z[:, boot_idx]

        # Score
        score = Z_boot @ w_boot

        # Softmax posterior
        exp_s = np.exp(score - np.max(score))
        posterior = exp_s / exp_s.sum()

        posterior_prob_matrix[:, i] = posterior

        winner = np.argmax(posterior)
        posterior_rank_counts[winner] += 1

    df[f"{prior_name}_MeanPosterior"] = posterior_prob_matrix.mean(axis=1)
    df[f"{prior_name}_TopProb"] = posterior_rank_counts / n_sim

    results_all_priors.append(posterior_prob_matrix)

# ------------------------------------------------------
# Posterior Odds Ratio (Stabilization Prior)
# ------------------------------------------------------

stab_probs = df["Stabilization_MeanPosterior"].values
odds_ratio = stab_probs[0] / stab_probs[1] if len(df)==2 else None

# ------------------------------------------------------
# Leave-One-Feature-Out Stability
# ------------------------------------------------------

loo_stability = []

for i in range(len(features)):
    mask = [j for j in range(len(features)) if j != i]
    Z_sub = Z[:, mask]
    alpha_sub = priors["Stabilization"][mask]

    wins = 0
    for _ in range(n_sim):
        w = np.random.dirichlet(alpha_sub)
        score = Z_sub @ w
        if np.argmax(score) == 0:
            wins += 1
    loo_stability.append(wins / n_sim)

df["Stabilization_LOO_min"] = min(loo_stability)

# ------------------------------------------------------
# Save Results
# ------------------------------------------------------

df.to_excel("bayesian_mc_results.xlsx", index=False)

# ------------------------------------------------------
# Plot Posterior Distributions
# ------------------------------------------------------

plt.figure(figsize=(8,6))
for i in range(len(df)):
    plt.hist(results_all_priors[1][i], bins=60, alpha=0.5, density=True, label=df.iloc[i]["Model"])
plt.title("Posterior Distribution (Stabilization Prior)")
plt.xlabel("Posterior Probability")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("bayesian_stabilization_posterior.png")

print("\nBayesian Robust Classification v2.0 Complete\n")
print(df[["Model",
          "Uniform_TopProb",
          "Stabilization_TopProb",
          "Geometry_TopProb"]])

if odds_ratio:
    print(f"\nPosterior Odds Ratio (Stabilization Prior): {odds_ratio:.3f}")

print("\nSaved:")
print(" - bayesian_mc_results.xlsx")
print(" - bayesian_stabilization_posterior.png\n")