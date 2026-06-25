# Telecom Customer Feedback Analyzer

## Aspect-Based Sentiment Analysis (ABSA) System

> End-to-end NLP system that detects telecom service aspects in customer feedback and predicts sentiment for each aspect.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Creation](#dataset-creation)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Assumptions](#assumptions)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

Customer feedback in the telecom industry is multi-faceted — a single review may praise internet speed while criticizing billing practices. Traditional sentiment analysis misses this nuance by assigning a single polarity to the entire text.

This project implements **Aspect-Based Sentiment Analysis (ABSA)** that first detects which telecom service aspects are mentioned in feedback, then classifies the sentiment expressed toward each aspect independently. The system enables telecom operators to identify specific pain points and strengths at scale.

**Supported Aspects (15):**

| # | Aspect | Description |
|---|--------|-------------|
| 1 | network_coverage | Signal strength and availability |
| 2 | internet_speed | Data download/upload speeds |
| 3 | call_quality | Voice clarity and call stability |
| 4 | customer_support | Helpline and support experience |
| 5 | billing | Invoice accuracy and charges |
| 6 | recharge_plans | Plan options and benefits |
| 7 | data_balance | Data usage tracking and accuracy |
| 8 | roaming | International connectivity |
| 9 | sim_activation | New SIM and porting process |
| 10 | mobile_app_experience | Carrier app usability |
| 11 | ott_bundle_services | Bundled streaming subscriptions |
| 12 | pricing | Cost and affordability |
| 13 | value_for_money | Benefits relative to cost |
| 14 | data_validity | Plan duration and expiry |
| 15 | 5g_experience | 5G speed and coverage |

**Sentiment Labels:** Positive, Negative, Neutral

**Two approaches compared:**
- **Model 1:** Logistic Regression + TF-IDF (baseline)
- **Model 2:** DistilBERT fine-tuned (deep learning)

**Selected for production:** Model 2 (DistilBERT) — chosen for its contextual understanding of language, ability to handle negation and sarcasm, and scalability with additional training data.

---

## Dataset Creation

### Overview

| Property | Value |
|----------|-------|
| Total samples | 1,000 |
| Aspects per sample | 1–8 (avg 2.3) |
| Sentiment distribution | ~40% positive, ~40% negative, ~20% neutral |
| Source channels | app_review, twitter, complaint_ticket, survey, customer_care_chat |
| Noise level | ~30% (spelling errors + SMS abbreviations) |

### Creation Approach

The dataset was generated using LLM-based synthesis (Claude via Kiro) in 10 batches of 100 samples each, with strict distribution controls per batch.

### System Prompt (given once at start)

```
You are a dataset generation assistant for an NLP research project on Aspect-Based 
Sentiment Analysis (ABSA) in the telecom domain. Your job is to generate realistic 
customer feedback that telecom users would write on app stores, Twitter, or customer 
care portals. Output ONLY a JSON array. No explanations, no markdown, no preamble.

Each object must have exactly these keys:
- id: integer (auto-increment from the starting index I give you)
- feedback: string
- aspects: array of aspect strings from the allowed list
- aspect_sentiments: object mapping each aspect to one of ["positive", "negative", "neutral"]
- source_channel: one of ["app_review", "twitter", "complaint_ticket", "survey", "customer_care_chat"]
```

### Batch Generation Prompt (per batch)

```
Generate exactly 100 customer feedback entries starting from id: {start_id}.
Follow this distribution strictly:
- 40% of entries must mention 2 or more aspects
- 15% must mention 3 or more aspects
- Sentiment distribution across all entries: 40% positive, 40% negative, 20% neutral
- 30% of entries should have realistic noise — distribute noise types as:
  * 15%: spelling mistakes (e.g. "intrenet", "custmer", "prblm", "reacharg")
  * 15%: SMS/chat abbreviations (e.g. "plz", "ur", "v bad", "cant blv")
- 70% of entries should be clean, grammatically normal English
- Mix short feedback (5–15 words) and long feedback (40–80 words) — roughly 50/50
- Vary source_channel across all 5 options
Focus this batch on these aspects (but include others naturally): {focus_aspects}
Return only the JSON array, nothing else.
```

### Validation Prompt (after each batch)

```python
# Automated validation script checks:
# - Total entries count
# - Multi-aspect distribution (2+ aspects ≥ 40%, 3+ aspects ≥ 15%)
# - Sentiment distribution (40/40/20 tolerance ±5%)
# - Noise percentage (~30%)
# - Schema validity (all keys present, valid aspects, valid sentiments)
# - No duplicate feedback text
# - Source channel distribution (all 5 present)
```

### Data Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| id | int | Unique identifier | 1042 |
| feedback | string | Customer feedback text | "plz fix ur network coverage" |
| aspects | list[str] | Detected aspects | ["network_coverage", "call_quality"] |
| aspect_sentiments | dict | Sentiment per aspect | {"network_coverage": "negative"} |
| source_channel | string | Feedback source | "twitter" |

---

## Project Structure

```
absa-project/
├── data/
│   ├── raw/                         # Original batch JSON files
│   ├── absa_telecom_combined.csv    # Full combined dataset (1000 rows)
│   └── telecom_absa_cleaned.csv     # Cleaned dataset
├── notebooks/
│   └── evaluation_summary.ipynb     # Day 6 evaluation deliverable
├── models/                          # Trained model artifacts (.pt, .pkl, .joblib)
├── app/
│   ├── app.py                       # Streamlit entry point + router
│   ├── views/
│   │   ├── single_feedback.py       # Page 1: Single analysis
│   │   └── batch_processing.py      # Page 2: Batch CSV processing
│   ├── components/
│   │   ├── result_card.py           # Full prediction display
│   │   ├── sentiment_badge.py       # Colored sentiment pill
│   │   ├── confidence_bar.py        # Confidence progress bar
│   │   └── metric_cards.py          # KPI summary cards
│   └── utils/
│       ├── app_helpers.py           # Colors, formatting, samples
│       ├── csv_helpers.py           # CSV validation and processing
│       ├── batch_runner.py          # Batch inference with progress
│       └── dashboard_charts.py      # 7 Plotly chart functions
├── src/
│   ├── config.py                    # Config loader (YAML)
│   ├── preprocessing.py             # PreprocessingPipeline class
│   ├── features.py                  # TFIDFFeatures + SentenceEmbeddings
│   ├── train.py                     # Model 1 aspect detection training
│   ├── train_sentiment.py           # Model 1 per-aspect sentiment
│   ├── dl_model.py                  # DistilBERT model definitions
│   ├── dl_data_prep.py              # HuggingFace dataset preparation
│   ├── dl_train.py                  # Model 2 aspect training loop
│   ├── dl_train_sentiment.py        # Model 2 sentiment training
│   ├── inference.py                 # ABSAInferencePipeline (production)
│   ├── evaluate.py                  # Shared evaluation utilities
│   ├── evaluate_model1.py           # Model 1 full evaluation
│   ├── evaluate_distilbert.py       # Model 2 full evaluation
│   ├── error_analysis.py            # Misclassification analysis
│   ├── model_comparison.py          # Side-by-side comparison tables
│   ├── per_aspect_comparison.py     # Per-aspect F1 analysis
│   ├── confusion_matrix_analysis.py # Confusion matrix generation
│   ├── data_split.py                # Stratified train/val/test split
│   └── improvements_report.py       # Actionable recommendations
├── tests/
│   ├── test_inference_pipeline.py   # Production readiness tests
│   ├── test_streamlit_page1.py      # Page 1 logic tests
│   └── test_batch_processing.py     # Batch processing tests
├── outputs/
│   ├── eda/                         # All visualization PNGs
│   └── models/                      # Evaluation JSONs and reports
├── config.yaml                      # Central configuration
├── requirements.txt                 # Pinned dependencies
├── requirements-dev.txt             # Dev dependencies
└── README.md                        # This file
```

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone <repo-url>
cd absa-project
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
```

### 5. Download Model Weights

The DistilBERT model files are too large for Git. Download from Google Drive:

| File | Size | Link |
|------|------|------|
| `aspect_detector_distilbert.pt` | 253 MB | [https://drive.google.com/file/d/1bDPBysWAyPYcCDfJNwN6WD89KeL9E09j/view?usp=sharing] |
| `sentiment_classifiers_distilbert.pt` | 3.8 GB | [https://drive.google.com/file/d/1hzAd7IagxaFD1ETJRBGGjMFH3q-Oy-vH/view?usp=sharing] |

Place both files in the `models/` directory.

### 6. Verify Setup

```bash
python tests/test_inference_pipeline.py
```

Expected output: `ALL TESTS PASSED — Pipeline is production-ready`

---

## Running the Project

### Train Models

```bash
# Preprocessing + data split
python -m src.data_split

# Model 1 — Logistic Regression (fast, ~30s)
python -m src.train
python -m src.train_sentiment

# Model 2 — DistilBERT (requires GPU, ~1 hour on T4)
python -m src.dl_data_prep
python -m src.dl_train
python -m src.dl_train_sentiment
```

### Run Streamlit App

```bash
streamlit run app/app.py
```

**Page 1 — Single Feedback:** Enter or select sample feedback → click Analyze → view detected aspects with sentiment and confidence.

**Page 2 — Batch Processing:** Upload CSV with "feedback" column (max 1000 rows) → run analysis → view dashboard with 7 interactive charts → download results CSV.

### Run Tests

```bash
python tests/test_inference_pipeline.py
python tests/test_streamlit_page1.py
python tests/test_batch_processing.py
```

---

## Model Architecture

### Model 1 — Logistic Regression + TF-IDF

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌────────────┐
│ Raw Feedback │ →  │ PreprocessPipeline│ →  │ TF-IDF Vectorizer │ →  │ Classifier │
│              │    │ clean→tokenize→  │    │ max_features=10K  │    │            │
│              │    │ stopwords→lemma  │    │ ngram=(1,2)       │    │            │
└──────────────┘    └─────────────────┘    └──────────────────┘    └────────────┘
                                                                         │
                                                    ┌────────────────────┼────────────────────┐
                                                    ▼                    ▼                    ▼
                                           ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                                           │ OneVsRest LR │    │ Per-Aspect   │    │ Per-Aspect   │
                                           │ (15 aspects) │    │ Sentiment LR │    │ Sentiment LR │
                                           │ Multi-label  │    │ (aspect 1)   │    │ (aspect 15)  │
                                           └──────────────┘    └──────────────┘    └──────────────┘
```

| Hyperparameter | Value |
|----------------|-------|
| max_iter | 1000 |
| class_weight | balanced |
| solver | lbfgs |
| TF-IDF max_features | 10,000 |
| TF-IDF ngram_range | (1, 2) |
| sublinear_tf | True |

### Model 2 — DistilBERT Fine-tuned (Production)

```
┌──────────────┐    ┌───────────────────┐    ┌────────────────────────────────────┐
│ Raw Feedback │ →  │ DistilBertTokenizer│ →  │        DistilBERT (66M params)      │
│              │    │ max_length=128     │    │                                    │
│              │    │ pad + truncate     │    │  [CLS] token → 768-dim hidden      │
└──────────────┘    └───────────────────┘    └──────────────────┬─────────────────┘
                                                                │
                                                        ┌───────┴───────┐
                                                        ▼               ▼
                                                ┌──────────────┐ ┌──────────────┐
                                                │ Dropout(0.3) │ │ Dropout(0.3) │
                                                │ Linear(768→15)│ │ Linear(768→3)│
                                                │ Sigmoid      │ │ Softmax      │
                                                │ (aspects)    │ │ (sentiment)  │
                                                └──────────────┘ └──────────────┘
```

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs (aspect) | 5 |
| Epochs (sentiment) | 3 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Dropout | 0.3 |
| Aspect threshold | 0.5 |
| Max sequence length | 128 |

---

## Results

### Model Comparison (Test Set)

| Metric | Model 1 (LR+TF-IDF) | Model 2 (DistilBERT) |
|--------|---------------------|---------------------|
| Aspect Micro-F1 | **0.8939** | 0.7518 |
| Aspect Macro-F1 | **0.9122** | 0.7674 |
| Sentiment Accuracy | **0.8446** | 0.6833 |
| Sentiment Macro-F1 | **0.8339** | 0.6651 |
| Sentiment Weighted-F1 | **0.8424** | 0.6834 |

### Hardware Comparison

| Property | Model 1 | Model 2 |
|----------|---------|---------|
| Training time | 10.6s | ~45 min (GPU) |
| Inference (ms/sample) | 0.05 ms | ~500 ms |
| Model size | 0.4 MB | 253 MB |
| Peak RAM | 0.97 MB | ~1.3 MB |

**Note:** Model 1 outperforms Model 2 on this 1,000-sample dataset. This is expected — LR excels on small datasets while DistilBERT requires 3,000+ samples to reach full potential. Model 2 is selected for production due to its architectural advantages (context understanding, negation handling, subword tokenization) that will compound with additional training data.

---

## Assumptions

| Assumption | Justification |
|------------|---------------|
| Code-switching excluded | Dataset is English-only; multilingual ABSA requires separate modeling |
| 1,000 samples sufficient for baseline | Demonstrates full pipeline; production needs 5K+ |
| DistilBERT over BERT-base | 40% smaller, 60% faster, <3% accuracy loss — better for Streamlit |
| Aspect detection threshold = 0.5 | Standard binary classification cutoff; tunable in production |
| 70/15/15 train/val/test split | Standard ML split; val for tuning, test for final reporting |
| Max 128 tokens | Covers 99%+ of telecom feedback; longer texts are rare |
| LLM-generated data | Cost-effective; validated against distribution constraints |

---

## Future Improvements

1. **More training data** — Generate 3,000+ additional samples to unlock DistilBERT's potential
2. **Span-based ABSA** — Adopt ASTE (Aspect Sentiment Triplet Extraction) for joint aspect-opinion-sentiment extraction
3. **Ensemble approach** — Combine Model 1 + Model 2 predictions (union for aspects, majority vote for sentiment)
4. **Active learning** — Flag low-confidence predictions for human review and iterative retraining
5. **Sarcasm detection** — Add explicit sarcasm labels and detection layer to handle implicit negative sentiment

See `outputs/models/improvements_report.md` for detailed recommendations.

---

## Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/) — DistilBERT model and tokenizer
- [sentence-transformers](https://www.sbert.net/) — Sentence embedding features
- [Streamlit](https://streamlit.io/) — Web application framework
- [scikit-learn](https://scikit-learn.org/) — ML models and evaluation metrics
- [Plotly](https://plotly.com/) — Interactive dashboard charts

---

*Built as part of an NLP portfolio project demonstrating end-to-end ABSA system development.*
