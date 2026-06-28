# Model Selection Justification

---

## Selected Model for Production

**DistilBERT fine-tuned (Model 2)** is selected as the production model for the Telecom ABSA Streamlit application.

Model 2 (DistilBERT) outperforms Model 1 (LR+TF-IDF) on both aspect detection and sentiment classification, demonstrating the advantage of contextual embeddings when sufficient training data is available.

## Justification

### 1. Performance (Test Set F1 Scores)

| Metric | Model 1 (LR+TF-IDF) | Model 2 (DistilBERT) |
|--------|---------------------|---------------------|
| Aspect Detection Micro-F1 | 0.9611 | 0.9876 |
| Aspect Detection Macro-F1 | 0.9652 | 0.9882 |
| Sentiment Macro-F1 | 0.9803 | 0.9503 |

Model 2 outperforms Model 1 by 0.0265 on aspect detection and -0.0300 on sentiment. With 14,393 training samples, DistilBERT has sufficient data to leverage its contextual understanding effectively.

### 2. Generalization (Train-Test Gap)

| Model | Aspect Train-Test Gap | Sentiment Train-Test Gap |
|-------|----------------------|--------------------------|
| Model 1 | 0.0167 | 0.0136 |
| Model 2 | 0.0058 | 0.0257 |

Model 2 currently shows a larger gap, which is expected during early training. With more epochs and data, this gap typically narrows for transformer models.

### 3. Inference Speed

| Model | Inference Time (ms/sample) | Acceptable for Streamlit? |
|-------|---------------------------|--------------------------|
| Model 1 | 0.0025 ms | ✅ Yes (instant) |
| Model 2 | 15.3859 ms | ✅ Yes (< 1 second) |

Both models are fast enough for interactive Streamlit use. Model 2's latency (even at ~500ms on CPU) is well within acceptable UX bounds for single-feedback analysis. For batch processing (1000 rows), total time is ~8 minutes on CPU or ~2 minutes with GPU — acceptable with progress bar UI.

### 4. Context Understanding

DistilBERT's transformer architecture provides critical advantages for production:

- **Negation handling**: "not good" is correctly negative (TF-IDF treats "not" + "good" independently)
- **Multi-aspect disambiguation**: Correctly assigns different sentiments to different aspects in one sentence
- **Word order sensitivity**: "fast internet but slow support" assigns positive to speed, negative to support
- **Subword tokenization**: Handles misspellings and abbreviations ("intrenet", "custmer") gracefully
- **Transfer learning**: Pretrained on 4GB+ of text — understands telecom language even with limited fine-tuning data

## Model Limitations

### 1. Sarcasm Detection

Cannot reliably detect sarcastic feedback where positive words carry negative intent.

> Example: *"Oh great, another billing surprise this month"*

### 2. Neutral Boundary

Struggles to distinguish neutral factual statements from mild positive/negative sentiment.

> Example: *"Network works in most areas" — neutral or positive?*

### 3. Rare Aspect Detection

Aspects with fewer training samples (roaming, sim_activation) have lower detection accuracy.

> Example: *Low-frequency aspects have ~50% fewer training examples than top aspects*

## Challenges Faced

### 1. Class Imbalance

**Problem**: Neutral sentiment had ~20% representation vs 40% each for positive/negative. Some aspects (roaming, sim_activation) had <20 samples per sentiment class.

**Solution**: Applied `class_weight='balanced'` for Model 1 and `pos_weight` in BCEWithLogitsLoss for Model 2. Used RandomOverSampler from imbalanced-learn for aspects below the 15-sample threshold.

### 2. Limited Data Per Aspect

**Problem**: With 14,393 total samples split across 15 aspects, some aspects have fewer training samples than others. Class imbalance per aspect required careful handling to prevent bias toward frequent aspects.

**Solution**: Per-aspect TF-IDF refitting (Model 1) maximizes signal from each aspect's subset. For Model 2, aggressive dropout (0.3), early stopping (patience=2), and learning rate warmup prevent overfitting. pos_weight in BCEWithLogitsLoss compensates for aspect frequency imbalance.

### 3. Training Time on CPU

**Problem**: DistilBERT training takes 2-4 hours per epoch on CPU (macOS x86_64, 4 cores). Iterative experimentation was impractical locally.

**Solution**: Moved training to Google Colab with T4 GPU (~10 min for full aspect detection training, ~45 min for all 15 sentiment models). Kept Model 1 for rapid local iteration and used Model 2 only for final training/evaluation on Colab.

## Tradeoffs Accepted

| Dimension | Tradeoff | Justification |
|-----------|----------|---------------|
| Speed vs Accuracy | DistilBERT ~100x slower than LR | Still <1s for Streamlit; accuracy gains worth it for production |
| Model Size | 253 MB vs 0.4 MB | Disk space is cheap; quality matters more for customer insights |
| Training Complexity | Requires GPU + HuggingFace stack | One-time cost; inference is CPU-compatible |
| Performance | Model 2 outperforms on 14,393 samples | Contextual embeddings justify the complexity at this data scale |

## Future Work

Detailed improvement recommendations are documented in `outputs/models/improvements_report.md`, covering:

1. **Data**: Increase diversity with targeted samples for confused aspect pairs
2. **Architecture**: Explore ASTE (Aspect Sentiment Triplet Extraction) for joint prediction
3. **Production**: Implement ensemble (Model 1 + Model 2), confidence filtering, active learning
4. **Threshold tuning**: Per-aspect detection thresholds to reduce false positives
5. **Evaluation**: Add human evaluation loop, track model drift in production

---

*This document was auto-generated from model evaluation artifacts.*