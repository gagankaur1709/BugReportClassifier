# decide_and_run_test.py
import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon
import re

# Function to extract AUC values from the CV_list(AUC) column
def extract_auc_values(cv_list):
    # Regular expression to capture floats after 'np.float64(' only
    pattern = r"np\.float64\(([-+]?\d*\.\d+|\d+)\)"

    # Find all matches and convert them to floats
    float_values = [float(match) for match in re.findall(pattern, cv_list)]

    return float_values


# Load the CSV files
baseline_results = pd.read_csv("./results/pytorch_NB.csv")
sbert_results = pd.read_csv("./results/pytorch_SBERT.csv")

# Extract AUC values for both models
baseline_auc_values = extract_auc_values(baseline_results["CV_list(AUC)"].iloc[0])
baseline_median_auc = np.median(baseline_auc_values)
sbert_auc_values = extract_auc_values(sbert_results["CV_list(AUC)"].iloc[0])
sbert_median_auc = np.median(sbert_auc_values)

# Calculate differences between the two models for each fold
differences = np.array(baseline_auc_values) - np.array(sbert_auc_values)

# Check for normality of differences using Shapiro-Wilk test
shapiro_stat, shapiro_p_value = shapiro(differences)
print(f"\nShapiro-Wilk Test:")
print(f"Statistic = {shapiro_stat}, p-value = {shapiro_p_value}")

# Decide which statistical test to run based on normality test
if shapiro_p_value > 0.05:
    # Data is normally distributed, use paired t-test
    t_stat, p_value = ttest_rel(baseline_auc_values, sbert_auc_values)
    print("\nPaired T-Test Results:")
    print(f"t-statistic = {t_stat}, p-value = {p_value}")
else:
    # Data is not normally distributed, use Wilcoxon signed-rank test
    w_stat, p_value = wilcoxon(differences)
    print("\nWilcoxon Signed-Rank Test Results:")
    print(f"w-statistic = {w_stat}, p-value = {p_value}")

print(f"Baseline median = {baseline_median_auc}, SBERT median = {sbert_median_auc}")