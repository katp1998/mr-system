import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

df = pd.read_csv("survey-results.csv")

emotion_cols = ["Happy", "Excited", "Tense", "Fearful", "Sad", "Hopeful"]

# p_ij (proportion of raters per emotion per track)
n = 30
pij = df[emotion_cols] / n
print("\nSTEP 1: Proportions (p_ij):")
print(pij.round(3).head())

# P_i (agreement per track)
pi = (df[emotion_cols] * (df[emotion_cols] - 1)).sum(axis=1) / (n * (n - 1))
print("\nSTEP 2: Agreement per track (P_i):")
print(pi.round(3))

# Mean observed agreement (P')
P_bar = pi.mean()
print(f"\nSTEP 3: Mean observed agreement (P'): {P_bar:.3f}")

# p_j (overall proportion per emotion) and expected agreement (P'_e)
pj = df[emotion_cols].sum(axis=0) / (len(df) * n)
P_e_bar = (pj ** 2).sum()
print("\nSTEP 4: Category proportions (p_j):")
print(pj.round(3))
print(f"Expected agreement by chance (P̄_e): {P_e_bar:.3f}")

# Fleiss’ Kappa
kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
print(f"\nSTEP 5: Fleiss’ Kappa (κ): {kappa:.3f}")

# Interpretation (Landis & Koch, 1977)
if kappa < 0.00:
    interpretation = "Poor agreement"
elif kappa < 0.21:
    interpretation = "Slight agreement"
elif kappa < 0.41:
    interpretation = "Fair agreement"
elif kappa < 0.61:
    interpretation = "Moderate agreement"
elif kappa < 0.81:
    interpretation = "Substantial agreement"
else:
    interpretation = "Almost perfect agreement"

print(f"Interpretation: {interpretation}")

# Optional verification using statsmodels built-in
M = df[emotion_cols].to_numpy()
kappa_check = fleiss_kappa(M, method='fleiss')
print(f"\nVerification using statsmodels: κ = {kappa_check:.3f}")
