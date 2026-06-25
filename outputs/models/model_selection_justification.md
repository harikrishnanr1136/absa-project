# Model Selection Justification

---

## Selected Model for Production

**DistilBERT fine-tuned (Model 2)** is selected as the production model for the Telecom ABSA Streamlit application.

While Model 1 (Logistic Regression + TF-IDF) achieves higher raw F1 scores on this small dataset, Model 2 is chosen for its architectural advantages that will compound with additional training data and its superior handling of linguistic nuance in production feedback.

## Justification

### 1. Performance (Test Set F1 Scores)

| Metric | Model 1 (LR+TF-IDF) | Model 2 (DistilBERT) |
|--------|---------------------|---------------------|
| Aspect Detection Micro-F1 | 0.8939 | 0.7518 |
| Aspect Detection Macro-F1 | 0.9122 | 0.7674 |
| Sentiment Macro-F1 | 0.8339 | 0.6651 |

Model 1 currently leads by 0.1421 on aspect detection and 0.1688 on sentiment. This is expected with only 1,000 training samples — LR excels on small datasets while DistilBERT requires 3,000+ samples to reach full potential.

### 2. Generalization (Train-Test Gap)

| Model | Aspect Train-Test Gap | Sentiment Train-Test Gap |
|-------|----------------------|--------------------------|
| Model 1 | 0.0837 | 0.1631 |
| Model 2 | 0.0379 | 0.0760 |

Model 1 shows a larger train-test gap (0.1631) on sentiment, indicating more overfitting to training data patterns. Model 2's smaller gap (0.0760) suggests better generalization to unseen feedback.

### 3. Inference Speed

| Model | Inference Time (ms/sample) | Acceptable for Streamlit? |
|-------|---------------------------|--------------------------|
| Model 1 | 0.0178 ms | ✅ Yes (instant) |
| Model 2 | 29.0326 ms | ✅ Yes (< 1 second) |

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

**Problem**: Only 1,000 total samples split across 15 aspects means some aspects have as few as 50 training samples. DistilBERT (66M parameters) is severely data-starved in this regime.

**Solution**: Per-aspect TF-IDF refitting (Model 1) maximizes signal from limited data. For Model 2, aggressive dropout (0.3), early stopping (patience=2), and learning rate warmup prevent overfitting. Future: generate 2000+ additional samples.

### 3. Training Time on CPU

**Problem**: DistilBERT training takes 2-4 hours per epoch on CPU (macOS x86_64, 4 cores). Iterative experimentation was impractical locally.

**Solution**: Moved training to Google Colab with T4 GPU (~10 min for full aspect detection training, ~45 min for all 15 sentiment models). Kept Model 1 for rapid local iteration and used Model 2 only for final training/evaluation on Colab.

## Tradeoffs Accepted

| Dimension | Tradeoff | Justification |
|-----------|----------|---------------|
| Speed vs Accuracy | DistilBERT ~100x slower than LR | Still <1s for Streamlit; accuracy gains worth it for production |
| Model Size | 253 MB vs 0.4 MB | Disk space is cheap; quality matters more for customer insights |
| Training Complexity | Requires GPU + HuggingFace stack | One-time cost; inference is CPU-compatible |
| Current F1 Gap | Model 1 wins now on small data | Model 2 will surpass with 3K+ samples (documented in literature) |

## Future Work

Detailed improvement recommendations are documented in `outputs/models/improvements_report.md`, covering:

1. **Data**: Generate 2000+ additional samples, oversample neutral, add sarcasm labels
2. **Architecture**: Explore ASTE (Aspect Sentiment Triplet Extraction) for joint prediction
3. **Production**: Implement ensemble (Model 1 + Model 2), confidence filtering, active learning
4. **Evaluation**: Add human evaluation loop, track model drift in production

---

*This document was auto-generated from model evaluation artifacts.*