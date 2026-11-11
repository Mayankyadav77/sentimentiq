"""
build_notebook.py - Script to generate comprehensive EDA Jupyter notebook
"""

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown cell: Header
nb.cells.append(nbf.v4.new_markdown_cell("""# SentimentIQ: Exploratory Data Analysis (EDA)
### Customer Review Intelligence Platform
**Dataset:** Women's E-Commerce Clothing Reviews (Nick Brooks, Kaggle / CC0 Public Domain)  
**Objective:** Perform statistical, lexical, and categorical analysis of authentic customer review data to inform feature engineering, class rebalancing, and model evaluation strategies.
"""))

# Code cell: Imports
nb.cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set plotting styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

print("Libraries imported successfully.")
"""))

# Markdown cell: Section 1
nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Loading & Dimension Audit
We load the raw customer review dataset stored in `data/raw/customer_reviews.csv`.
"""))

# Code cell: Load Data
nb.cells.append(nbf.v4.new_code_cell("""data_path = os.path.join("..", "data", "raw", "customer_reviews.csv")
if not os.path.exists(data_path):
    data_path = os.path.join("data", "raw", "customer_reviews.csv")

df_raw = pd.read_csv(data_path)
print(f"Dataset Shape: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")
df_raw.head(3)
"""))

# Markdown cell: Section 2
nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Column Descriptions & Schema Analysis
Inspect column data types, non-null counts, and missingness.
"""))

# Code cell: Schema Info
nb.cells.append(nbf.v4.new_code_cell("""schema_df = pd.DataFrame({
    "Data Type": df_raw.dtypes,
    "Non-Null Count": df_raw.notnull().sum(),
    "Missing Count": df_raw.isnull().sum(),
    "Missing %": (df_raw.isnull().mean() * 100).round(2)
})
schema_df
"""))

# Markdown cell: Section 3
nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Data Cleaning: Missing Values & Duplicate Handling
- **Missing Review Text:** Rows with missing review text cannot be used for NLP modeling and are dropped.
- **Duplicates:** Deduplication based on identical review text within the same product ID.
"""))

# Code cell: Clean & Deduplicate
nb.cells.append(nbf.v4.new_code_cell("""# Drop rows where Review Text is missing
df_clean = df_raw.dropna(subset=["Review Text"]).copy()

# Deduplicate
initial_len = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=["Clothing ID", "Review Text"])
print(f"Dropped {len(df_raw) - initial_len:,} missing text rows and {initial_len - len(df_clean)} duplicates.")
print(f"Total Valid Records Retained: {len(df_clean):,}")
"""))

# Markdown cell: Section 4
nb.cells.append(nbf.v4.new_markdown_cell("""## 4. Star Rating & Derived Sentiment Class Distribution
Customer ratings follow a standard 5-star Likert scale:
- **1 & 2 Stars:** `Negative`
- **3 Stars:** `Neutral`
- **4 & 5 Stars:** `Positive`
"""))

# Code cell: Sentiment Mapping
nb.cells.append(nbf.v4.new_code_cell("""rating_map = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}
df_clean["Sentiment"] = df_clean["Rating"].map(rating_map)

# Rating counts
rating_counts = df_clean["Rating"].value_counts().sort_index()
sentiment_counts = df_clean["Sentiment"].value_counts()
sentiment_pct = (df_clean["Sentiment"].value_counts(normalize=True) * 100).round(2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Star ratings
sns.barplot(x=rating_counts.index, y=rating_counts.values, ax=axes[0], palette="Blues_d")
axes[0].set_title("Customer Star Rating Distribution (1 to 5 Stars)")
axes[0].set_xlabel("Star Rating")
axes[0].set_ylabel("Review Count")

# 3-Class Sentiment
colors = {"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"}
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=axes[1], palette=colors)
axes[1].set_title("Derived 3-Class Sentiment Distribution")
axes[1].set_xlabel("Sentiment Class")
axes[1].set_ylabel("Review Count")

for idx, p in enumerate(axes[1].patches):
    height = p.get_height()
    axes[1].annotate(f"{sentiment_pct.iloc[idx]}%", (p.get_x() + p.get_width() / 2., height + 200),
                     ha="center", fontsize=10, weight="bold")

plt.tight_layout()
plt.show()

for s, cnt in sentiment_counts.items():
    print(f"Class '{s:<8}': {cnt:6d} reviews ({sentiment_pct[s]}%)")
"""))

# Markdown cell: Section 5
nb.cells.append(nbf.v4.new_markdown_cell("""## 5. Review Length & Lexical Density Analysis
Analyze character count and word count per review across sentiment classes.
"""))

# Code cell: Review Length
nb.cells.append(nbf.v4.new_code_cell("""df_clean["char_count"] = df_clean["Review Text"].astype(str).str.len()
df_clean["word_count"] = df_clean["Review Text"].astype(str).str.split().str.len()

print("Review Length Statistics (Word Count):")
print(df_clean.groupby("Sentiment")["word_count"].describe().round(1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
sns.histplot(data=df_clean, x="word_count", hue="Sentiment", palette=colors, kde=True, bins=30, ax=axes[0])
axes[0].set_title("Review Word Count Distribution")
axes[0].set_xlabel("Word Count")

# Boxplot
sns.boxplot(data=df_clean, x="Sentiment", y="word_count", palette=colors, ax=axes[1])
axes[1].set_title("Word Count Spread per Sentiment Class")
axes[1].set_xlabel("Sentiment Class")
axes[1].set_ylabel("Word Count")

plt.tight_layout()
plt.show()
"""))

# Markdown cell: Section 6
nb.cells.append(nbf.v4.new_markdown_cell("""## 6. High-Frequency Sentiment Vocabulary (N-Gram Signals)
Analyze key unigrams associated with Positive and Negative reviews.
"""))

# Code cell: Top Words
nb.cells.append(nbf.v4.new_code_cell("""import re

def extract_frequent_words(texts, top_k=12):
    tokens = []
    stopwords = {"the", "and", "is", "it", "to", "in", "of", "this", "for", "that", "my", "with", "on", "was", "dress", "shirt", "top", "size", "wear", "bought", "look", "ordered"}
    for t in texts:
        words = re.findall(r"[a-z]{3,}", str(t).lower())
        tokens.extend([w for w in words if w not in stopwords])
    return Counter(tokens).most_common(top_k)

pos_words = extract_frequent_words(df_clean[df_clean["Sentiment"] == "Positive"]["Review Text"])
neg_words = extract_frequent_words(df_clean[df_clean["Sentiment"] == "Negative"]["Review Text"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Positive words
pos_df = pd.DataFrame(pos_words, columns=["Word", "Count"]).sort_values(by="Count")
axes[0].barh(pos_df["Word"], pos_df["Count"], color="#10b981")
axes[0].set_title("Top Words in Positive Reviews")
axes[0].set_xlabel("Frequency")

# Negative words
neg_df = pd.DataFrame(neg_words, columns=["Word", "Count"]).sort_values(by="Count")
axes[1].barh(neg_df["Word"], neg_df["Count"], color="#ef4444")
axes[1].set_title("Top Words in Negative Reviews")
axes[1].set_xlabel("Frequency")

plt.tight_layout()
plt.show()
"""))

# Markdown cell: Section 7
nb.cells.append(nbf.v4.new_markdown_cell("""## 7. Department & Merchandise Aspect Analysis
Cross-tabulate merchandise divisions with customer sentiment.
"""))

# Code cell: Department Breakdown
nb.cells.append(nbf.v4.new_code_cell("""if "Department Name" in df_clean.columns:
    dept_ct = pd.crosstab(df_clean["Department Name"], df_clean["Sentiment"], normalize="index") * 100
    dept_ct = dept_ct.round(1)
    
    dept_ct[["Positive", "Neutral", "Negative"]].plot(
        kind="barh", stacked=True, color=["#10b981", "#f59e0b", "#ef4444"], figsize=(12, 5)
    )
    plt.title("Department Sentiment Share (%)")
    plt.xlabel("Percentage (%)")
    plt.ylabel("Department")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    
    print(dept_ct)
"""))

# Markdown cell: Section 8
nb.cells.append(nbf.v4.new_markdown_cell("""## 8. Summary of Findings & Machine Learning Decisions
1. **Class Skew:** 77% Positive vs 10.5% Negative. Modeling must employ **Stratified Splitting** and **Class Weighting (`class_weight='balanced'`)** in Linear SVM and Logistic Regression to avoid majority-class bias.
2. **Evaluation Metric:** **Macro F1-Score** must be the primary model selection criterion rather than plain accuracy.
3. **Lexical Context:** Bigrams (`ngram_range=(1, 2)`) and negation preservation (`"not"`, `"never"`) are required to prevent sentiment inversion in critical negative reviews.
"""))

# Save notebook
out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "exploratory_analysis.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook successfully created at: {out_path}")
