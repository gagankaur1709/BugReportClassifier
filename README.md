# BugReportClassifier
# Performance Bug Classifier

A machine learning pipeline to classify bug reports as **performance-related** or **non-performance**, using SBERT embeddings, VADER sentiment analysis, and SVM. Outperforms traditional TF-IDF + Naive Bayes baselines in key projects.

![Pipeline Diagram](docs/pipeline.png) *(Example architecture diagram - add your image here)*

## Key Features
- 🚀 **Hybrid NLP Approach**: Combines SBERT (semantic embeddings) + VADER (sentiment) for richer context
- ⚖️ **Statistical Rigor**: 30-fold cross-validation with Wilcoxon/paired t-tests
- 📊 **Reproducible**: Exact random seeds and versioned dependencies
- 🔍 **Interpretable**: Clear feature engineering (title/body vs. comments)

## Installation
Checkout the manual.pdf file
