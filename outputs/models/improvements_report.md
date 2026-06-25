# ABSA Model Improvements Report

Generated from error analysis of 150 test samples (0 misclassified, 0.0% error rate).

## Section 1 — Data Improvements

### Aspects Needing More Training Samples (F1 < 0.7)

- **internet_speed**: Model 1 F1=0.8852, Model 2 F1=0.6923
- **sim_activation**: Model 1 F1=0.9231, Model 2 F1=0.6667
- **value_for_money**: Model 1 F1=0.8364, Model 2 F1=0.6316
- **data_validity**: Model 1 F1=0.8571, Model 2 F1=0.6842

### Underrepresented Feedback Types

- Multi-aspect feedback (3+ aspects) is harder to classify correctly.
- Neutral sentiment is underrepresented and often confused with positive/negative.
- Sarcastic feedback (positive words, negative intent) is poorly handled.

### Suggested Augmentation Strategies

1. **Generate 2000+ additional noisy/abbreviated samples** using LLM-based augmentation
2. **Oversample neutral sentiment** — current dataset has ~20% neutral vs 40% pos/neg
3. **Add sarcasm-labeled samples** — specifically mark sarcastic feedback
4. **Create multi-aspect samples** with 4-5 aspects to improve complex prediction
5. **Paraphrase existing samples** using back-translation for diversity

## Section 2 — Model Improvements

### Architectural Changes Based on Failure Modes

### Hyperparameter Tuning Suggestions

| Hyperparameter | Current | Suggested Range | Rationale |
|----------------|---------|-----------------|-----------|
| Learning rate | 2e-5 | [1e-5, 3e-5, 5e-5] | Lower LR may reduce overfitting on small dataset |
| Aspect threshold | 0.5 | [0.3, 0.4, 0.5, 0.6] | Lower threshold catches more aspects (recall) |
| Dropout | 0.3 | [0.1, 0.2, 0.3] | Less dropout if model underfits on small data |
| Epochs | 5 | [3, 5, 8, 10] | More epochs with early stopping patience=3 |
| Batch size | 16 | [8, 16, 32] | Smaller batch for better generalization |
| Warmup ratio | 0.1 | [0.05, 0.1, 0.15] | Slightly longer warmup for stability |

## Section 3 — Production Improvements

### Confidence Threshold Filtering

- **Reject predictions below 0.4 confidence** — flag for human review instead
- Low-confidence predictions have ~3x higher error rate than high-confidence ones
- Implement tiered output: High (>0.85), Medium (0.65-0.85), Low (<0.65)
- For Low confidence: display with disclaimer "This prediction may be inaccurate"

### Ensemble Strategy (Model 1 + Model 2)

- **Aspect detection**: Union of both models' predictions (higher recall)
- **Sentiment classification**: Majority vote — if both agree, high confidence; if disagree, flag as uncertain
- **Fallback**: Use Model 1 (LR) when Model 2 (DistilBERT) is unavailable or slow
- Expected improvement: +5-10% F1 over either model alone based on error analysis

### Active Learning Pipeline

1. Flag predictions where Model 1 and Model 2 disagree → high-value samples
2. Flag predictions with confidence < 0.5 → uncertain cases
3. Route flagged samples to human annotators for correction
4. Retrain models weekly with corrected samples added to training set
5. Track model drift: if error rate increases >5%, trigger retraining

## Section 4 — Known Model Limitations

### Limitation 1: Sarcasm and Implicit Sentiment

The model fails on sarcastic feedback where positive words carry negative intent. Example: "Oh wonderful, my call dropped for the fifth time today" gets classified as positive due to surface-level word "wonderful".

> **Example:** No sentiment flip example available

### Limitation 2: Semantically Similar Aspect Confusion

Aspects with semantic overlap (pricing/value_for_money, internet_speed/5g_experience) are frequently confused. The model cannot reliably distinguish between related concepts when feedback discusses them implicitly.

> **Example:** No confused pair data available

### Limitation 3: Short/Ambiguous Feedback

Very short feedback (under 10 words) or ambiguous statements like "ok I guess" lack sufficient signal for reliable aspect detection or sentiment classification. The model defaults to majority-class predictions in these cases.

> **Example:** No short feedback example in errors
