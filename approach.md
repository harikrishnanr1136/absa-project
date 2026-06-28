# Technical Approach Document

## Telecom ABSA System

---

## 1. Problem Framing

### Why ABSA is harder than standard sentiment analysis

Standard sentiment analysis assigns a single polarity to an entire text. In telecom customer feedback, a single sentence often expresses contradictory sentiments toward different aspects: *"The 5G speed is amazing but customer support is terrible"* — assigning one sentiment loses critical information. ABSA requires the model to simultaneously identify *what* is being discussed (aspect detection) and *how* the user feels about it (sentiment classification) — a fundamentally harder compositional reasoning task.

### Problem Decomposition

The ABSA task was decomposed into two sequential subtasks:

**Subtask 1 — Multi-label Aspect Detection:**
- Input: raw feedback text
- Output: binary vector of 15 aspects (multiple can be active)
- Formulation: multi-label classification with BCEWithLogitsLoss (Model 2) or OneVsRestClassifier (Model 1)

**Subtask 2 — Per-Aspect Sentiment Classification:**
- Input: raw feedback text + detected aspect
- Output: one of {positive, negative, neutral} per aspect
- Formulation: 15 independent 3-class classifiers, each trained on the subset of data where that aspect is present

### Why pipeline over end-to-end

A pipeline approach (detect aspects → classify sentiment) was chosen over end-to-end ASTE (Aspect Sentiment Triplet Extraction) models because:

| Factor | Pipeline | End-to-End (ASTE) |
|--------|----------|-------------------|
| Interpretability | Each subtask is independently debuggable | Black-box joint prediction |
| Error isolation | Can identify if errors are in detection or sentiment | Cannot separate error sources |
| Modularity | Can upgrade one component without retraining the other | Must retrain entire model |
| Data requirement | Works with limited samples via decomposition | Requires 5,000+ samples for joint learning |
| Complexity | Standard classification models | Requires span extraction + relation modeling |

**Accepted tradeoff:** The pipeline approach may miss aspect-opinion interactions that joint models capture (e.g., "fast" near "internet" vs "fast" near "activation"). With more data, migrating to ASTE would be the natural next step.

---

## 2. Dataset Creation Methodology

### Why LLM generation over manual annotation

| Factor | LLM Generation | Manual Annotation |
|--------|---------------|-------------------|
| Cost | $0 (Kiro subscription) | $500-2000 for 14K+ samples |
| Speed | 14,600 samples generated in batches | 2-4 weeks |
| Consistency | Controlled distribution via prompts | High inter-annotator variance |
| Scalability | Can generate 10K+ with same process | Linear cost increase |
| Domain knowledge | Encoded in system prompt | Requires trained annotators |

**Trade-off accepted:** LLM-generated data may not capture real conversational patterns, regional slang, or genuine emotional intensity that real customer data contains. This is documented as a known limitation.

### How realism was ensured

1. **Noise injection (30%)**: Split between spelling mistakes ("intrenet", "custmer") and SMS abbreviations ("plz", "ur", "v bad") — mimics real app store reviews and Twitter
2. **Length variation**: 50/50 split between short (5-15 words) and long (40-80 words) — mirrors real distribution where tweets are short but complaints are detailed
3. **Source channels (5)**: Different writing styles per channel — formal (survey) vs casual (twitter) vs urgent (complaint_ticket)
4. **Multi-aspect feedback (40%+)**: Real reviews often discuss multiple aspects; single-aspect-only data would create an unrealistic distribution

### Batch theme strategy

Aspects were grouped by semantic relatedness across batches:
- Batch 1: network_coverage, call_quality, 5g_experience (infrastructure)
- Batch 2: internet_speed, data_balance, data_validity (data-related)
- Batch 3: billing, recharge_plans, pricing, value_for_money (commercial)
- Batch 4: customer_support (service)
- Batch 5: sim_activation, roaming, mobile_app_experience (operational)
- Batch 6: ott_bundle_services, value_for_money, recharge_plans (value)
- Batches 7-10: mixed, edge cases, long-form, short-form

This ensures each aspect has dedicated attention while also appearing naturally in other batches.

### Known limitations of LLM-generated data

- Lacks genuine emotional intensity of frustrated customers
- May over-represent coherent grammar even in "noisy" samples
- Aspect co-occurrence patterns may not match real telecom feedback
- Sarcasm and implicit sentiment are underrepresented
- Cultural and regional language patterns are missing

---

## 3. Preprocessing Design Decisions

### Why lemmatization over stemming

Stemming (Porter algorithm) is too aggressive for domain-specific telecom vocabulary:

| Word | Stemmed | Lemmatized | Better for ABSA |
|------|---------|-----------|-----------------|
| "billing" | "bill" | "billing" | Lemma preserves the aspect term |
| "connectivity" | "connect" | "connectivity" | Lemma keeps the domain meaning |
| "recharges" | "recharg" | "recharge" | Lemma produces a real word |
| "charged" | "charg" | "charge" | Lemma is interpretable |

Lemmatization produces valid English words that preserve semantic meaning needed for both aspect identification and TF-IDF feature quality.

### Why negation words were kept in stopword removal

Standard stopword removal eliminates "not", "no", "never". In sentiment analysis, these flip polarity entirely:
- "good service" → positive
- "**not** good service" → negative

Removing "not" makes both sentences identical to the model. Our `SENTIMENT_KEEP_WORDS` set preserves: {not, no, never, very, too, but, however, although, though, yet, only, just} — all words that modify sentiment meaning.

### Why domain terms were protected

Terms like "5g", "sim", "ott", "volte", "prepaid", "postpaid", "recharge" are domain-critical vocabulary that should never be removed, stemmed, or modified. They directly correspond to aspect categories and their presence is a strong signal for aspect detection.

### Why sublinear_tf=True in TF-IDF

Customer feedback often contains repetitions ("very very slow", "bad bad service"). Without sublinear TF, a word appearing 10 times gets 10x the weight of a word appearing once. `sublinear_tf` applies `tf = 1 + log(tf)`, dampening repetition effects and focusing on word *presence* rather than raw frequency.

### Why max_length=128 tokens for DistilBERT

Analysis of feedback lengths showed: mean=27 words, max=63 words. With DistilBERT tokenization (subword), 128 tokens covers 99%+ of all feedback without truncation. Using 256 or 512 would double memory usage and inference time for zero benefit on this dataset.

### Abbreviation expansion rationale

Telecom feedback from app stores and Twitter contains heavy abbreviations. Expanding them ("plz"→"please", "ur"→"your", "tbh"→"to be honest") normalizes the vocabulary, reducing feature sparsity in TF-IDF and improving subword tokenization in DistilBERT. Without expansion, "ur" and "your" would be treated as unrelated tokens.

---

## 4. Feature Engineering Decisions

### TF-IDF: why ngram_range=(1,2)

Unigrams alone miss critical sentiment-bearing phrases:
- "not good" → unigrams see {"not", "good"} separately
- "call quality" → the bigram is a direct aspect indicator
- "very slow" → captures intensifier + adjective pattern
- "no signal" → captures negation + symptom

Bigrams (n=2) capture these phrases while keeping vocabulary manageable. Trigrams (n=3) were not used because they would explode vocabulary size with limited training samples, leading to severe sparsity.

### Sentence embeddings: why all-MiniLM-L6-v2

| Model | Dimension | Speed | Quality | Chosen? |
|-------|-----------|-------|---------|---------|
| all-MiniLM-L6-v2 | 384 | Fast (80ms/batch) | Good | ✅ |
| all-mpnet-base-v2 | 768 | Medium | Best | Too slow for Streamlit |
| paraphrase-TinyBERT | 128 | Fastest | Lower | Insufficient quality |

MiniLM-L6-v2 offers the best speed/quality tradeoff for a Streamlit deployment where inference latency matters.

### Why two feature types were compared

TF-IDF and sentence embeddings have complementary strengths:
- TF-IDF: captures exact telecom keyword matches (explicit aspect signals)
- Embeddings: capture semantic similarity and context (implicit sentiment signals)

Comparing both validates which approach is more suitable for the data size and task structure.

---

## 5. Model Selection Rationale

### Why Logistic Regression as baseline

- **Fast and interpretable**: Trains in 10 seconds, feature weights explain decisions
- **Strong on small data**: Fast, interpretable, strong baseline — with good TF-IDF features often competitive with neural models
- **OneVsRest strategy**: Naturally handles multi-label by training 15 independent binary classifiers
- **class_weight="balanced"**: Built-in handling of aspect frequency imbalance
- **Reproducible**: Deterministic with random_state=42

### Why DistilBERT as main model

- **Contextual embeddings**: "fast internet" vs "fast activation" get different representations based on context
- **Negation handling**: "not good" is embedded differently from "good" — critical for sentiment
- **Subword tokenization**: Handles misspellings ("intrenet") and abbreviations gracefully
- **Transfer learning**: Pretrained on 4GB+ of text — brings general language understanding even with limited fine-tuning data
- **Size**: 66M parameters, 253 MB on disk — feasible for CPU inference in Streamlit

### Why not larger models

| Model | Parameters | Rejected Because |
|-------|-----------|-----------------|
| BERT-base | 110M | 2x slower inference, 2x memory, marginal gain on 1K samples |
| RoBERTa | 125M | Same size issues, no tokenizer advantage for this domain |
| BERT-large | 340M | 5x slower, requires GPU for inference — not Streamlit-friendly |
| GPT-based | 1B+ | Requires API calls — can't run locally, cost per prediction |

---

## 6. Training Strategy

### Class imbalance handling

- **Model 1 (LR)**: `class_weight="balanced"` adjusts loss inversely proportional to class frequency
- **Model 2 (DistilBERT aspect)**: `pos_weight` in BCEWithLogitsLoss computed as `neg_count/pos_count` per aspect — upweights rare aspects like sim_activation (72 samples) vs recharge_plans (260 samples)
- **Model 2 (DistilBERT sentiment)**: `CrossEntropyLoss(weight=class_weights)` where weight is inversely proportional to sentiment frequency per aspect

### Early stopping: patience=2

With limited per-aspect samples, overfitting risk is present (typically epoch 2-3 for DistilBERT). Patience=2 allows one epoch of no improvement before stopping — balances between premature stopping and memorization.

### Learning rate 2e-5

Standard for BERT fine-tuning (per original BERT paper). Higher rates (5e-5+) cause catastrophic forgetting of pretrained knowledge. Lower rates (1e-6) would require more epochs and risk underfitting.

### Warmup ratio 0.1

The first 10% of training steps use linearly increasing learning rate. This prevents the randomly initialized classification head from causing large gradients that destabilize the pretrained DistilBERT weights in early training.

---

## 7. Evaluation Strategy

### Why macro-F1 as primary metric

- **Accuracy** is misleading with class imbalance (predicting majority class gives high accuracy)
- **Micro-F1** is dominated by frequent aspects (network_coverage, internet_speed)
- **Macro-F1** weighs all 15 aspects equally — ensures rare aspects (roaming, sim_activation) matter in evaluation
- A model scoring high macro-F1 truly generalizes across all aspects

### Why separate evaluation

Aspect detection and sentiment classification have independent error modes:
- A model might detect aspects perfectly but misclassify sentiment (or vice versa)
- Combined metrics would mask which component needs improvement
- Separate evaluation enables targeted debugging (error_analysis.py)

### Why train/val/test all reported

| Split | Purpose |
|-------|---------|
| Train metrics | Ceiling — how well model fits training data |
| Val metrics | Tuning — used for hyperparameter selection and early stopping |
| Test metrics | Generalization — never seen during training or tuning |
| Train-Test gap | Overfitting indicator — large gap = model memorized training data |

Model 1 train-test gap: 0.08 (aspect F1), 0.16 (sentiment F1)
Model 2 train-test gap: 0.04 (aspect F1), 0.08 (sentiment F1)

### Error analysis methodology

1. Run inference on full test set
2. Compare predictions to ground truth at aspect+sentiment level
3. Classify errors by type (false_positive_aspect, sentiment_flip, neutral_confusion, etc.)
4. Identify patterns (noisy inputs, multi-aspect, short texts)
5. Generate actionable improvement recommendations

---

## 8. Production Design Decisions

### Why inference.py is framework-agnostic

`src/inference.py` imports zero Streamlit modules. This enables:
- Use from Streamlit app (current)
- Use from FastAPI REST endpoint (future)
- Use from batch scripts and evaluation pipelines
- Unit testing without Streamlit installation

### Why @st.cache_resource for model loading

DistilBERT loads ~253 MB of weights + tokenizer. Without caching, every page interaction would reload the model (~20 seconds). `@st.cache_resource` loads once per session and shares across all reruns — critical for interactive UX.

### Why config.yaml for all constants

Centralizing configuration eliminates:
- Hardcoded paths scattered across 25+ files
- Inconsistency between training and inference (different aspect lists)
- Need to modify source code for deployment changes (paths differ on Colab vs local)

Every path, label list, hyperparameter, and threshold lives in one file that all modules reference.

---

## 9. Challenges and Solutions

### Challenge 1: PyTorch incompatibility with Python 3.13

**Challenge:** The project started with Python 3.13 (latest Homebrew), but PyTorch had no wheels for 3.13, blocking all deep learning work.

**Solution:** Installed Python 3.11 via Homebrew alongside 3.13, created a new venv with 3.11, and pinned compatible versions: `torch==2.2.2`, `transformers<4.40`, `sentence-transformers<3.0`.

**Tradeoff:** Locked to older package versions; some newer transformer features unavailable.

---

### Challenge 2: Class imbalance across aspects and sentiments

**Challenge:** Aspect frequency varied 5x (recharge_plans: 260 vs sim_activation: 72). Neutral sentiment was only ~20% vs 40% for positive/negative, causing models to underpredict neutral.

**Solution:** Applied `pos_weight` in BCEWithLogitsLoss scaled by neg/pos ratio per aspect. Used `class_weight="balanced"` in LR. Applied RandomOverSampler for aspects below 15-sample threshold.

**Tradeoff:** Oversampling minority classes can lead to overfitting on duplicated examples; balanced weights reduce overall accuracy slightly in exchange for per-class fairness.

---

### Challenge 3: CPU training too slow for DistilBERT iteration

**Challenge:** Full DistilBERT training took 2-4 hours per experiment on macOS x86_64 (4 cores, no GPU). Iterative hyperparameter tuning was impractical locally.

**Solution:** Moved all DistilBERT training to Google Colab with T4 GPU (~10 min for aspect detection, ~45 min for all 15 sentiment models). Kept Model 1 (LR) for rapid local iteration.

**Tradeoff:** Dependency on external compute; training results not reproducible without GPU access.

---

### Challenge 4: Multi-label stratified splitting

**Challenge:** With 15 aspects and many unique combinations (291 unique aspect tuples), stratified train/val/test split failed because many combinations had only 1 sample (can't split a single sample across sets).

**Solution:** Grouped rare aspect combinations (count < 3) under a generic "__rare__" label for stratification. If that still failed, fell back to random split.

**Tradeoff:** Random split doesn't guarantee identical aspect distributions across splits, but the deviation was < 5% for all aspects — acceptable for 150-sample test set.

---

### Challenge 5: Streamlit import conflicts with app directory naming

**Challenge:** Having both `app.py` (file) and `app/` (directory) caused Python to treat `app` as the file module, breaking all `from app.components...` imports. Additionally, Streamlit auto-detected `app/pages/` and created phantom navigation links.

**Solution:** Removed root-level `app.py` (old FastAPI placeholder). Renamed `app/pages/` to `app/views/` to prevent Streamlit's multi-page auto-detection. Entry point stays at `app/app.py`.

**Tradeoff:** Non-standard directory naming (`views` instead of `pages`); requires running `streamlit run app/app.py` explicitly.

---

## 10. Known Limitations

1. **LLM-generated data** may not capture real user language nuances — genuine frustration, cultural expressions, and organic co-occurrence patterns differ from synthetic generation.

2. **15 fixed aspects** may miss emerging telecom issues (e.g., eSIM problems, data privacy concerns, tower radiation worries) that real customers discuss.

3. **CPU inference is slow for large batches** — processing large batches takes significant time on CPU. Production deployment would benefit from GPU or model distillation.

4. **Small dataset per rare aspect** — sim_activation (72 samples) and roaming (73 samples) have significantly fewer training examples, leading to lower and less stable F1 scores for these aspects.

5. **No multilingual support** — the system handles English only. Real telecom markets in India, Southeast Asia, and Africa have significant code-switching (English + local language) that this system cannot process.

6. **Sarcasm blindness** — the model cannot reliably detect sarcastic feedback where positive surface words carry negative intent (e.g., "Oh wonderful, another billing surprise this month").

7. **Threshold sensitivity** — aspect detection uses a fixed 0.5 threshold. Some aspects may benefit from lower thresholds (increase recall) while others need higher (increase precision). Per-aspect threshold tuning is not implemented.

---

*This document reflects the technical decisions made during development. See `outputs/models/improvements_report.md` for specific recommendations on addressing these limitations.*
