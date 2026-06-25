"""
Sentiment Word Clouds for ABSA Telecom Dataset
Generates word clouds for positive, negative, and neutral feedback.
"""

import json
import os
import re
import string

import ssl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# Bypass SSL verification for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords", quiet=True)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "absa_telecom_combined.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "eda")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use NLTK stopwords
STOPWORDS = set(stopwords.words("english"))

# Add domain-specific filler words to stopwords
STOPWORDS.update([
    "also", "would", "could", "even", "get", "got", "one", "two", "three",
    "much", "every", "still", "since", "like", "really", "always", "never",
    "actually", "just", "well", "back", "make", "made", "say", "said",
    "thing", "things", "need", "want", "use", "used", "using", "time",
    "going", "gone", "come", "take", "gives", "give", "day", "days",
])


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, remove stopwords."""
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)


def main():
    # Load data
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    # Build corpora per sentiment
    corpora = {"positive": [], "negative": [], "neutral": []}

    for entry in data:
        feedback = entry["feedback"]
        sentiments_in_entry = set(entry["aspect_sentiments"].values())

        for sentiment in sentiments_in_entry:
            if sentiment in corpora:
                corpora[sentiment].append(feedback)

    # Clean and join each corpus
    cleaned_corpora = {}
    for sentiment, texts in corpora.items():
        cleaned_texts = [clean_text(t) for t in texts]
        cleaned_corpora[sentiment] = " ".join(cleaned_texts)

    # Generate word clouds
    colormap_map = {
        "positive": "Greens",
        "negative": "Reds",
        "neutral": "Greys",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, sentiment in zip(axes, ["positive", "negative", "neutral"]):
        wc = WordCloud(
            width=800,
            height=500,
            background_color="white",
            colormap=colormap_map[sentiment],
            max_words=100,
            random_state=42,
            collocations=False,
        ).generate(cleaned_corpora[sentiment])

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(sentiment.upper(), fontsize=16, fontweight="bold", pad=10)
        ax.axis("off")

    plt.suptitle("Word Clouds by Sentiment Class", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "wordclouds.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Word clouds saved to: {output_path}")
    print(f"  Positive corpus: {len(corpora['positive'])} feedbacks")
    print(f"  Negative corpus: {len(corpora['negative'])} feedbacks")
    print(f"  Neutral corpus:  {len(corpora['neutral'])} feedbacks")


if __name__ == "__main__":
    main()
