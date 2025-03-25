import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon
import re


# Function to extract metric values from the CV_list columns
def extract_metric_values(cv_list, metric):
    """
    Extracts numeric values from a string representation of a list.
    For AUC, it uses a specific regex to handle 'np.float64(...)' format.
    For other metrics, it uses a general regex to extract numeric values.
    """
    if metric == "AUC":
        # Specific regex for AUC values (handles 'np.float64(...)' format)
        pattern = r"np\.float64\(([-+]?\d*\.\d+|\d+)\)"
    else:
        # General regex for other metrics (extracts numeric values)
        pattern = r"[-+]?\d*\.\d+|\d+"

    # Find all matches and convert them to floats
    float_values = [float(match) for match in re.findall(pattern, cv_list)]
    return float_values


# List of projects to evaluate
projects = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']

# List of metrics to evaluate (only Precision, Accuracy, and F1)
metrics = ["FI", "Precision", "Accuracy"]  # Note: "FI" is F1-score based on your previous code

# Perform statistical tests for each project and metric
for project in projects:
    print(f"\n\n=== Project: {project.upper()} ===")

    # Construct file paths
    baseline_path = f"./results/{project}_NB.csv"
    sbert_path = f"./results/{project}_SBERT.csv"

    # Load the CSV files
    try:
        baseline_results = pd.read_csv(baseline_path)
        sbert_results = pd.read_csv(sbert_path)
    except FileNotFoundError:
        print(f"Warning: Files not found for project {project}. Skipping...")
        continue

    for metric in metrics:
        print(f"\n=== Statistical Analysis for {metric} ===")

        # Extract metric values for both models
        try:
            baseline_values = extract_metric_values(baseline_results[f"CV_list({metric.upper()})"].iloc[0], metric)
            sbert_values = extract_metric_values(sbert_results[f"CV_list({metric.upper()})"].iloc[0], metric)
        except KeyError:
            print(f"Warning: Metric {metric} not found in project {project}. Skipping...")
            continue

        # Calculate differences between the two models for each fold
        differences = np.array(baseline_values) - np.array(sbert_values)

        # Check for normality of differences using Shapiro-Wilk test
        shapiro_stat, shapiro_p_value = shapiro(differences)
        print(f"Shapiro-Wilk Test for {metric}:")
        print(f"Statistic = {shapiro_stat}, p-value = {shapiro_p_value}")

        # Decide which statistical test to run based on normality test
        if shapiro_p_value > 0.05:
            # Data is normally distributed, use paired t-test
            t_stat, p_value = ttest_rel(baseline_values, sbert_values)
            print(f"Paired T-Test Results for {metric}:")
            print(f"t-statistic = {t_stat}, p-value = {p_value}")
        else:
            # Data is not normally distributed, use Wilcoxon signed-rank test
            w_stat, p_value = wilcoxon(baseline_values, sbert_values)
            print(f"Wilcoxon Signed-Rank Test Results for {metric}:")
            print(f"w-statistic = {w_stat}, p-value = {p_value}")

        # Calculate medians for both models
        baseline_median = np.median(baseline_values)
        sbert_median = np.median(sbert_values)
        print(f"Median {metric} (Baseline): {baseline_median}")
        print(f"Median {metric} (SBERT): {sbert_median}")

        # Interpret the results
        if p_value > 0.05:
            print(f"Conclusion for {metric}: No significant difference between models.")
        else:
            if baseline_median > sbert_median:
                print(f"Conclusion for {metric}: Baseline model performs significantly better.")
            else:
                print(f"Conclusion for {metric}: SBERT model performs significantly better.")