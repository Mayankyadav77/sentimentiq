# SentimentIQ – Customer Review Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.49-FF4B4B.svg)](https://streamlit.io/)
[![License: CC0](https://img.shields.io/badge/Dataset%20License-CC0--Public%20Domain-green.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Tests](https://img.shields.io/badge/Tests-39%20Passing-brightgreen.svg)]()

> A production-style, end-to-end Machine Learning and Natural Language Processing platform designed for customer review sentiment classification, model explainability, categorical intelligence, and executive analytics.

---

## 1. Project Title & Overview

**SentimentIQ** is a production-grade NLP platform that analyzes customer feedback to extract actionable business intelligence. Unlike toy sentiment applications that rely on hardcoded heuristics or black-box predictions, SentimentIQ implements a complete ML lifecycle: automated data ingestion, negation-preserving tokenization, sublinear TF-IDF vectorization, multi-algorithm cross-validation, token-level feature attribution, and an enterprise multi-page Streamlit dashboard.

---

## 2. Problem Statement

E-commerce and SaaS platforms receive thousands of customer reviews daily across multiple product departments. Key operational challenges include:
- **Severe Class Imbalance:** Happy customers frequently leave positive reviews, resulting in $75\%+$ majority-class representation that biases standard accuracy metrics.
- **Semantic Inversion via Negations:** Standard NLP pipelines strip stop words like *"not"* and *"never"*, causing negative reviews (*"not comfortable"*) to be misclassified as positive.
- **Lack of Interpretability:** Business stakeholders distrust black-box predictions without clear evidence of which words drove the decision.
- **Operational Latency:** Enterprises require both sub-second real-time single review scoring and vectorized batch processing for uploaded spreadsheets.

SentimentIQ directly solves these challenges through balanced loss optimization, negation-preserving tokenization, linear model explainability ($w_j \cdot x_j$), and modular architecture.

---

## 3. Project Objectives

1. Develop a reusable, thread-safe NLP preprocessing pipeline that preserves sentiment-critical negations.
2. Prevent data leakage by enforcing strict train-only fitting of vocabulary and inverse document frequency (IDF) weights.
3. Train, tune, and empirically benchmark three fundamental ML classifiers: **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear Support Vector Machine (LinearSVC)**.
4. Address real-world class imbalance using Stratified Splitting and balanced class weighting (`class_weight='balanced'`).
5. Select champion models based on **Validation Macro F1-Score** rather than biased overall accuracy.
6. Provide model-specific explainability showing positive and negative feature weights for every prediction.
7. Deliver a SaaS-style Streamlit dashboard supporting single review analysis, CSV bulk processing, executive analytics, and model performance auditing.

---

## 4. Key Features

- **Multi-Model Benchmark:** Compare Multinomial Naive Bayes, Logistic Regression, and Linear SVM side-by-side.
- **Dynamic Model Switcher:** Switch inference models in real time to compare decision boundaries and probabilities.
- **Token-Level Attribution:** Deconstructs linear model coefficients to display exact words pushing or opposing the prediction.
- **Vectorized Bulk Inference:** Drag-and-drop CSV upload capable of scoring hundreds of reviews in milliseconds with downloadable enriched results.
- **Executive Analytics:** Interactive Plotly charts for department-level sentiment cross-tabulation, N-gram keyword frequencies, and review length distributions.
- **Automated Test Suite:** 39 automated unit tests verifying schema validation, preprocessing consistency, zero data leakage, and prediction formats. Code coverage is not measured.

---

## 5. Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | Core programming language |
| **Data Manipulation** | Pandas 2.2, NumPy 2.2 | Data ingestion, vector manipulation, and aggregations |
| **Machine Learning** | Scikit-learn 1.8 | TF-IDF vectorization, GridSearch cross-validation, Naive Bayes, Logistic Regression, LinearSVC |
| **NLP & Lexical** | NLTK 3.9 | Corpus stopwords, tokenization, WordNet lemmatizer |
| **Model Serialization**| Joblib 1.5 | Deserialization and caching of fitted models and feature extractors |
| **Web Dashboard** | Streamlit 1.49 | Multi-page SaaS user interface |
| **Data Visualization** | Plotly 6.3, Seaborn, Matplotlib | Interactive heatmaps, donut charts, and attribution bar charts |
| **Testing** | Pytest 9.2 | Automated unit and integration testing |

---

## 6. System Architecture

```mermaid
flowchart TD
    subgraph Data Pipeline
        RAW[Raw Customer Review CSV\n23,486 rows] --> LOADER[src/data_loader.py\nSchema Validation & Deduplication]
        LOADER --> SPLIT[Stratified Train/Val/Test Split\n70% Train | 15% Val | 15% Test]
        SPLIT --> PREP[src/preprocessing.py\nContraction Expansion & Negation Preservation]
    end

    subgraph Feature Engineering & Training
        PREP --> TFIDF[src/feature_extraction.py\nSublinear TF-IDF (1, 2) N-Grams\nFit Strictly on Train Split]
        TFIDF --> MNB[Multinomial Naive Bayes\nalpha tuning]
        TFIDF --> LR[Logistic Regression\nBalanced Class Weights]
        TFIDF --> SVM[Linear SVM (LinearSVC)\nMargin Maximization]
        MNB & LR & SVM --> EVAL[src/evaluate.py\nMacro F1 Benchmark on Validation Split]
        EVAL --> CHAMPION[Champion Model Selection\nLinear SVM - Val Macro F1: 0.6531]
        CHAMPION --> ARTIFACTS[models/sentiment_model.pkl\nmodels/tfidf_vectorizer.pkl\nmodels/model_metadata.json]
    end

    subgraph Runtime Inference & Explainability
        ARTIFACTS --> PREDICT[src/predict.py\nInference Service with Model Switcher]
        ARTIFACTS --> EXPLAIN[src/explainability.py\nToken Feature Attribution w_j * x_j]
    end

    subgraph Streamlit Web Application
        PREDICT & EXPLAIN --> UI[Streamlit Multi-Page Portal]
        UI --> P1[Home Overview & KPIs]
        UI --> P2[1. Single Review Analysis & Word Attribution]
        UI --> P3[2. Bulk CSV Analysis & Export]
        UI --> P4[3. Analytics & Department Dashboard]
        UI --> P5[4. Model Performance & Confusion Matrix]
    end
```

---

## 7. Project Folder Structure

```text
sentimentiq/
│
├── app.py                          # Streamlit application portal & executive overview
├── requirements.txt                # Pinned production dependencies
├── README.md                       # Comprehensive portfolio & system documentation
├── .gitignore                      # Git exclusion rules
│
├── .streamlit/
│   └── config.toml                 # Streamlit server and theme configuration
│
├── data/
│   ├── raw/
│   │   └── customer_reviews.csv    # 23,486 authentic customer reviews
│   ├── processed/
│   │   ├── train.csv               # 15,847 stratified training records
│   │   ├── val.csv                 # 3,396 stratified validation records
│   │   └── test.csv                # 3,396 held-out test records
│   └── sample_bulk_reviews.csv     # 50 sample reviews for 1-click bulk testing
│
├── models/
│   ├── sentiment_model.pkl         # Champion model (Linear SVM)
│   ├── model_linear_svc.pkl        # Tuned Linear SVM
│   ├── model_logistic_regression.pkl # Tuned Logistic Regression
│   ├── model_naive_bayes.pkl       # Tuned Multinomial Naive Bayes
│   ├── tfidf_vectorizer.pkl        # Fitted TF-IDF feature extractor
│   └── model_metadata.json         # Benchmark metrics, hyperparameters, error logs
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Interactive EDA, distributions, and lexical charts
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Ingestion, validation, deduplication, stratified splitting
│   ├── preprocessing.py            # Negation-preserving text normalization & tokenization
│   ├── feature_extraction.py       # Sublinear TF-IDF (1, 2) n-gram vectorization
│   ├── train.py                    # Multi-model training, grid search CV, and selection
│   ├── evaluate.py                 # Multi-metric evaluation, confusion matrix, error analysis
│   ├── predict.py                  # Real-time & batch prediction engine with caching
│   ├── explainability.py           # Model-specific feature attribution (w_j * x_j)
│   └── analytics.py                # Business KPIs, category cross-tabulation, n-gram mining
│
├── pages/
│   ├── 1_Sentiment_Analysis.py     # Single review scoring + token-level attribution chart
│   ├── 2_Bulk_Analysis.py          # CSV batch upload, progress bar, filtering & export
│   ├── 3_Analytics_Dashboard.py    # Plotly distributions, department breakdowns, word bars
│   └── 4_Model_Performance.py      # Benchmark comparison, confusion heatmaps, error audit
│
└── tests/
    ├── __init__.py
    ├── test_data_validation.py     # Schema integrity and missing column exceptions
    ├── test_preprocessing.py       # Contractions, negations, HTML/URL stripping
    ├── test_feature_extraction.py  # Unigrams/bigrams, sparsity, zero data leakage
    ├── test_evaluation.py          # Artifact completeness and metric calculations
    ├── test_prediction.py          # Single/batch inference, probabilities vs decision scores
    └── test_analytics.py           # KPIs, distributions, and date handling
```

---

## 8. Dataset Source & Licensing

- **Dataset:** Women’s E-Commerce Clothing Reviews
- **Origin & Curators:** Nick Brooks (originally published on Kaggle via commercial e-commerce feedback)
- **Licensing:** **CC0: Public Domain** (Open for research, educational, and commercial applications)
- **Raw Dimensions:** 23,486 records, 11 columns
- **Retained Records:** 22,639 valid reviews (after dropping 845 rows missing review text and 2 duplicate reviews)
- **Rating to Sentiment Mapping (5-Star Likert Scale):**
  - **Negative:** Ratings 1.0 & 2.0 (2,369 reviews, 10.47%)
  - **Neutral:** Rating 3.0 (2,823 reviews, 12.47%)
  - **Positive:** Ratings 4.0 & 5.0 (17,447 reviews, 77.07%)

> **Note:** These are full-corpus figures (Train 15,847 + Val 3,396 + Test 3,396 = 22,639). Dashboard KPIs use the same denominator for consistency.

---

## 9. NLP Preprocessing Workflow

Our preprocessing pipeline (`src/preprocessing.py`) executes seven deterministic stages:
1. **Null Safeguard:** Safely replaces `None` and `NaN` with empty strings `""`.
2. **HTML & URL Removal:** Strips hyperlinks (`http://...`, `www...`) and tags (`<br/>`, `<b>`).
3. **Contraction Expansion:** Expands informal negative contractions (*"didn't"* $\rightarrow$ *"did not"*, *"wasn't"* $\rightarrow$ *"was not"*, *"won't"* $\rightarrow$ *"will not"*).
4. **Case Normalization:** Converts text to lowercase.
5. **Special Character Filtering:** Removes digits and symbols using `[^a-zA-Z\s]`, preserving alphabetic words and spaces.
6. **Negation-Preserving Stopword Filtering:** Computes `stopwords.words("english") - NEGATION_WORDS`, explicitly preserving `{"not", "no", "never", "nor", "neither", "without", "hardly", "barely"}`.
7. **Optional Lemmatization:** Employs NLTK WordNetLemmatizer for grammatical root extraction.

---

## 10. TF-IDF Feature Extraction

$$\text{TF-IDF}(t, d, D) = (1 + \log(\text{tf}(t, d))) \times \left(\log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1\right)$$

- **Unigrams + Bigrams (`ngram_range=(1, 2)`):** Captures single words alongside critical 2-word collocations (*"not good"*, *"highly recommend"*, *"damaged zipper"*).
- **Sublinear TF (`sublinear_tf=True`):** Replaces raw frequency with $1 + \log(\text{tf})$ to prevent long rants from dominating feature space.
- **Corpus Bounds (`min_df=2, max_df=0.90`):** Filters rare misspellings (min_df) and non-informative corpus buzzwords (max_df).
- **Strict Zero Data Leakage:** The vectorizer is fitted **strictly on the training split**. Validation and test splits are transformed without modifying the learned vocabulary.

---

## 11. Machine Learning Algorithms Used

1. **Multinomial Naive Bayes (`MultinomialNB`):** Probabilistic baseline applying Bayes' theorem under feature conditional independence. Fast $O(N \cdot D)$ training.
2. **Logistic Regression (`LogisticRegression`):** Discriminative linear model with softmax probability estimation and $L_2$ regularization. Employs `class_weight='balanced'`.
3. **Linear Support Vector Machine (`LinearSVC`):** Convex quadratic margin maximization finding optimal separating hyperplanes in 10,000-dimensional sparse space. Employs `class_weight='balanced'`.

---

## 12. Model Comparison Results (Actual Experimental Benchmark)

Evaluated on the independent **3,396-sample validation split** (`data/processed/val.csv`):

| Model | Hyperparameters | Accuracy | Macro F1 | Weighted F1 | Macro Recall | 3-Fold CV F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | `alpha=0.1` | 83.19% | 0.5931 | 0.8088 | 0.5554 | 0.5762 |
| **Logistic Regression** | `C=0.1, balanced` | 79.15% | 0.6284 | 0.8105 | 0.6868 | 0.6363 |
| **Linear SVM (`LinearSVC`)** 🏆 | `C=0.1, balanced` | **84.22%** | **0.6531** | **0.8408** | **0.6481** | **0.6369** |

*Selection Rationale:* **Linear SVM** was selected as champion due to the highest Validation Macro F1 score (`0.6531`), providing superior minority-class recovery without collapsing majority-class precision.

---

## 13. Final Evaluation Metrics on Held-Out Test Set

Evaluated **strictly once** on the unseen 3,396-sample test split (`data/processed/test.csv`):

```text
==============================================================================
CHAMPION MODEL TEST SET EVALUATION (Linear SVM)
==============================================================================
Test Accuracy        : 83.24%
Test Macro Precision : 63.87%
Test Macro Recall    : 63.54%
Test Macro F1-Score  : 63.63%
Test Weighted F1     : 82.93%

Per-Class Performance on Test Set:
  Class 'Negative': Precision = 56.27%, Recall = 59.44%, F1 = 0.5781 (Support = 355)
  Class 'Neutral ': Precision = 42.59%, Recall = 37.26%, F1 = 0.3975 (Support = 424)
  Class 'Positive': Precision = 92.75%, Recall = 93.92%, F1 = 0.9334 (Support = 2617)
==============================================================================
```

### Test Set Confusion Matrix

| Actual \ Predicted | Negative | Neutral | Positive |
| :--- | :---: | :---: | :---: |
| **Negative** | **211** | 49 | 95 |
| **Neutral** | 67 | **158** | 199 |
| **Positive** | 97 | 64 | **2,456** |

---

## 14. Application Visual UI Layout

The Streamlit user interface is organized into five responsive modules:
1. **Executive Landing Page (`app.py`):** Real-time KPI summary cards, sentiment donut chart, and instant single-review testing.
2. **Single Review Analyzer (`1_Sentiment_Analysis.py`):** Live text area, preset demonstration buttons, model selector, probability/decision score gauges, and token-level attribution horizontal bar charts.
3. **Bulk Review Processor (`2_Bulk_Analysis.py`):** Drag-and-drop CSV upload, 1-click sample dataset loader, batch prediction progress bar, sentiment radio filters, and enriched CSV download button.
4. **Analytics Intelligence (`3_Analytics_Dashboard.py`):** Merchandise department cross-tabulation, top positive/negative N-gram frequency charts, and review word count distribution boxplots.
5. **Model Evaluation Center (`4_Model_Performance.py`):** Algorithmic comparison matrix, interactive normalized confusion matrix heatmaps, and test set error audit.

---

## 15. Installation Instructions

### Prerequisites
- Python 3.11+ (64-bit recommended)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/SentimentIQ.git
cd SentimentIQ
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 16. How to Ingest Data & Train the Models

To execute data ingestion, cleaning, stratified splitting, hyperparameter tuning, and serialization:

```bash
# Step 1: Ingest, clean, and partition genuine customer review data
python src/data_loader.py

# Step 2: Train, tune, benchmark models, and evaluate champion on test set
python src/train.py
```

---

## 17. How to Run the Streamlit Application

Launch the local SaaS dashboard:

```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 18. Example Predictions & Explainability

### Example 1: Clearly Positive Review
```text
Input: "Absolutely love this cardigan! Beautiful colors, great fit, and soft fabric. 10/10!"
Predicted Sentiment: POSITIVE
Model: Linear SVM (Margin Score: +2.5281)
Top Supporting Terms: 'love' (+0.192), 'great' (+0.185), 'soft' (+0.153)
Explanation: "Prediction 'Positive' was strongly influenced by terms: 'love', 'great', 'soft'."
```

### Example 2: Negative Review with Negation
```text
Input: "The dress was not good. Poor stitching, fabric tore on first wear. Would not recommend."
Predicted Sentiment: NEGATIVE
Model: Linear SVM (Margin Score: +1.2429)
Top Supporting Terms: 'poor' (+0.416), 'not recommend' (+0.298), 'not' (+0.183)
Explanation: "Prediction 'Negative' was strongly influenced by terms: 'poor', 'not recommend', 'not'."
```

### Example 3: Neutral Review
```text
Input: "It fits okay. Nothing special, average quality for the price."
Predicted Sentiment: NEUTRAL
Model: Linear SVM (Margin Score: +0.1348)
Top Supporting Terms: 'okay' (+0.368), 'nothing special' (+0.185), 'average' (+0.138)
```

---

## 19. Architectural Limitations

1. **Bag-of-Words Context Boundary:** While unigrams and bigrams capture local phrases (*"not good"*, *"super soft"*), long-range rhetorical contrast across multiple sentences can challenge linear models.
2. **Neutral Class Ambiguity:** 3-star reviews often express balanced mixed sentiment rather than pure indifference, leading to lower neutral F1 scores.
3. **Probability vs. Distance:** Linear SVM outputs geometric margin distances rather than true posterior probabilities.

---

## 20. Future Improvements

1. **Aspect-Based Sentiment Analysis (ABSA):** Implement dependency parsing to extract sentiments directed at specific product aspects (*"fit"*, *"price"*, *"fabric"*).
2. **Transformer Integration:** Benchmark DistilBERT or RoBERTa against the linear SVM baseline.
3. **Active Learning Feedback Loop:** Allow domain reviewers to correct misclassified reviews directly in the UI and trigger model retraining.

---

## 21. Running Automated Tests

Run the full automated test suite (39 unit tests):

```bash
python -m pytest tests/ -v
```

---

# INTERVIEW PREPARATION (30 Technical Questions & Answers)

### A. Python & Software Engineering
1. **Q: How does SentimentIQ prevent memory exhaustion when serving predictions in Streamlit?**  
   *A:* It uses the Singleton design pattern in `src/predict.py` with in-memory caching (`self._cached_models`). Models and vectorizers are loaded lazily on first access and preserved in memory across user sessions, eliminating repeated disk I/O.
2. **Q: What is the difference between `@st.cache_data` and `@st.cache_resource`?**  
   *A:* `@st.cache_data` is used for serializable data structures (DataFrames, dictionaries, query outputs) and copies by value. `@st.cache_resource` is used for non-serializable, stateful global objects like database connections, ML models, or thread pools.
3. **Q: Why insert project root into `sys.path` in multi-page Streamlit applications?**  
   *A:* Streamlit executes scripts in `pages/` from their subfolder context. Inserting the project root ensures that absolute package imports (`from src.predict import ...`) resolve uniformly regardless of which script is executed.
4. **Q: Why use Python's `re.compile()` for URL and HTML regexes?**  
   *A:* Compiling regular expressions ahead of time avoids re-parsing regex syntax trees inside iterative loops, boosting text processing throughput by up to $3\times$.

### B. Natural Language Processing & Preprocessing
5. **Q: Why does SentimentIQ explicitly preserve negation words during stopword removal?**  
   *A:* Standard stopword lists remove words like *"not"*, *"no"*, and *"never"*. In sentiment analysis, this causes catastrophic polarity inversion—turning *"not comfortable"* into *"comfortable"*.
6. **Q: Why expand contractions before stripping punctuation?**  
   *A:* Stripping apostrophes first converts *"didn't"* into *"didnt"* or splits it into *"didn"* and *"t"*. Expanding contractions beforehand (*"didn't"* $\rightarrow$ *"did not"*) cleanly surfaces the negation token *"not"*.
7. **Q: What is the difference between Stemming and Lemmatization?**  
   *A:* Stemming cuts off suffixes heuristically, often producing non-words (*"studies"* $\rightarrow$ *"studi"*). Lemmatization uses vocabulary dictionaries and morphological analysis to return authentic root lemmas (*"better"* $\rightarrow$ *"good"*).
8. **Q: How does SentimentIQ handle emojis and special symbols?**  
   *A:* The preprocessing regex strips non-alphabetic characters while preserving spacing. For future extensions, an emoji-to-text dictionary (e.g., `:smile:` $\rightarrow$ `"positive"`) could be injected before character filtering.

### C. TF-IDF & Feature Engineering
9. **Q: What is TF-IDF and how is it calculated?**  
   *A:* $\text{TF-IDF} = \text{TF} \times \text{IDF}$. Term Frequency measures how often a word occurs in a document, while Inverse Document Frequency measures how rare the word is across the corpus.
10. **Q: Why use `sublinear_tf=True`?**  
    *A:* It scales term frequency logarithmically: $1 + \log(\text{TF})$. This prevents repetitive rants containing a word 15 times from dominating linear classifier decision boundaries over reviews containing it twice.
11. **Q: Why use bigrams (`ngram_range=(1, 2)`)?**  
    *A:* Pure unigrams lose word ordering. Bigrams enable the model to learn distinct feature weights for phrases like *"not good"*, *"cheap zipper"*, and *"highly recommend"*.
12. **Q: What do `min_df` and `max_df` control in `TfidfVectorizer`?**  
    *A:* `min_df=2` eliminates words appearing in fewer than 2 reviews (rare typos). `max_df=0.90` eliminates words appearing in $>90\%$ of reviews (corpus buzzwords that survived stopword filtering).
13. **Q: What is data leakage and how was it avoided during vectorization?**  
    *A:* Data leakage occurs when test set statistics influence model training. SentimentIQ fits `TfidfVectorizer` **strictly on the training split**. Validation and test reviews are projected using `transform()` only.

### D. Machine Learning Algorithms
14. **Q: What is the core assumption of Multinomial Naive Bayes?**  
    *A:* Features (word occurrences) are conditionally independent given the class: $P(X \mid y) = \prod P(x_i \mid y)$.
15. **Q: Why does Naive Bayes perform worse than Linear SVM on this dataset?**  
    *A:* The conditional independence assumption fails in natural language where words strongly depend on one another (*"customer service"*, *"poor fit"*). Additionally, severe class imbalance biases Naive Bayes priors heavily toward the majority class.
16. **Q: How does Logistic Regression handle multiclass classification?**  
    *A:* Using the Multinomial / Softmax formulation: $P(y = c \mid X) = \frac{e^{\mathbf{w}_c^T \mathbf{x} + b_c}}{\sum_j e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$, which normalizes class scores into a valid probability distribution.
17. **Q: What does `class_weight='balanced'` do mathematically?**  
    *A:* It adjusts loss weights inversely proportional to class frequencies: $w_c = \frac{N}{K \cdot N_c}$. Errors on the rare Negative class (10.5%) are penalized $\sim7.5\times$ more heavily than errors on the Positive class (77%).
18. **Q: Why are Linear SVMs well-suited for text classification?**  
    *A:* Text mapped to a 10,000-dimensional TF-IDF space is sparse and often linearly separable (*Cover's Theorem*). Linear SVMs find the maximum-margin hyperplane, offering strong generalization in high dimensions.
19. **Q: What is the regularization parameter $C$ in SVM?**  
    *A:* $C$ balances margin maximization against training error penalties. A smaller $C$ (e.g., $0.1$) allows a wider margin with some misclassifications (prevents overfitting), whereas a large $C$ strictly enforces correct classification of all training points.

### E. Model Evaluation & Error Analysis
20. **Q: Why is Accuracy a misleading metric for this customer review dataset?**  
    *A:* The dataset is 77% positive. A trivial model predicting "Positive" for every review achieves 77% accuracy while completely failing to detect customer complaints (0% recall on Negative reviews).
21. **Q: What is the difference between Macro F1 and Weighted F1?**  
    *A:* Macro F1 computes the unweighted average of F1 scores across classes, giving minority classes equal weight. Weighted F1 weights each class F1 by its support, reflecting real traffic distribution.
22. **Q: Why must the holdout test set be evaluated only once after model selection?**  
    *A:* Evaluating multiple models on the test set and selecting the best performer causes test set information to leak into model selection, leading to an overly optimistic estimate of generalization performance.
23. **Q: What is qualitative error analysis in NLP?**  
    *A:* Inspecting actual misclassified reviews to categorize error patterns (e.g., subtle sarcasm, mixed positive/negative clauses, neutral 3-star rating ambiguity) to guide future iterations.

### F. Explainability & Interpretability
24. **Q: How does SentimentIQ calculate token-level explainability for linear models?**  
    *A:* For linear classifiers, the decision score is $\sum_j x_j \cdot w_{c, j} + b_c$. The exact contribution of token $j$ is $x_j \cdot w_{c, j}$. Positive contributions support the class; negative contributions oppose it.
25. **Q: Can feature weights prove that a specific word caused the customer's sentiment?**  
    *A:* No. Feature weights reflect statistical correlation in the training corpus, not causal proof. SentimentIQ explicitly displays this disclaimer in the UI.
26. **Q: What are the limitations of linear model explainability?**  
    *A:* It evaluates terms within a bag-of-words space and does not capture long-range contextual modifiers across paragraphs or complex rhetorical satire.

### G. Production & Deployment
27. **Q: How do you deploy this project to Streamlit Community Cloud?**  
    *A:* Push the repository to GitHub, link the repository on share.streamlit.io, select `app.py` as the entry point, and Streamlit Cloud will automatically build dependencies from `requirements.txt`.
28. **Q: Why is having `.gitignore` critical before pushing to GitHub?**  
    *A:* It prevents checking in transient Python caches (`__pycache__`), virtual environments (`.venv`), temporary logs, and OS artifacts that bloat the repository and cause deployment collisions.
29. **Q: How does the application handle out-of-vocabulary (OOV) customer reviews?**  
    *A:* The fitted `ReviewFeatureExtractor` transforms unknown tokens into an all-zero sparse vector without crashing. The classifier classifies the review using model bias priors, and the UI displays a helpful notice.
30. **Q: How would you scale SentimentIQ to handle millions of daily reviews?**  
    *A:* Decouple inference from the web UI by deploying the model as a containerized FastAPI microservice on AWS ECS or Google Cloud Run, queueing review payloads via Apache Kafka or RabbitMQ, and storing aggregated analytics in ClickHouse or Snowflake.
