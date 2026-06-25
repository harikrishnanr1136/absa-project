"""
Dataset Statistics Script for ABSA Telecom Dataset
Generates comprehensive statistics for the combined dataset.
"""

import json
import pandas as pd

import os
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "absa_telecom_combined.json")


def load_data(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def main():
    df = load_data(DATA_PATH)

    print("=" * 60)
    print("ABSA TELECOM DATASET - STATISTICS REPORT")
    print("=" * 60)

    # 1. Total number of entries
    print(f"\n{'─' * 60}")
    print("1. TOTAL ENTRIES")
    print(f"{'─' * 60}")
    print(f"   Total records: {len(df)}")

    # 2. Number of unique aspects
    print(f"\n{'─' * 60}")
    print("2. UNIQUE ASPECTS")
    print(f"{'─' * 60}")
    all_aspects = set()
    for aspects_list in df["aspects"]:
        all_aspects.update(aspects_list)
    print(f"   Unique aspect count: {len(all_aspects)}")
    print(f"   Aspects: {sorted(all_aspects)}")

    # 3. Feedback length statistics
    print(f"\n{'─' * 60}")
    print("3. FEEDBACK LENGTH (word count)")
    print(f"{'─' * 60}")
    df["word_count"] = df["feedback"].apply(lambda x: len(x.split()))
    print(f"   Average: {df['word_count'].mean():.1f} words")
    print(f"   Minimum: {df['word_count'].min()} words")
    print(f"   Maximum: {df['word_count'].max()} words")
    print(f"   Median:  {df['word_count'].median():.1f} words")
    print(f"   Std Dev: {df['word_count'].std():.1f} words")

    # 4. Multi-aspect entries
    print(f"\n{'─' * 60}")
    print("4. MULTI-ASPECT ENTRIES")
    print(f"{'─' * 60}")
    df["aspect_count"] = df["aspects"].apply(len)
    multi_2 = (df["aspect_count"] >= 2).sum()
    multi_3 = (df["aspect_count"] >= 3).sum()
    print(f"   Entries with 1 aspect:   {(df['aspect_count'] == 1).sum()} ({(df['aspect_count'] == 1).sum() * 100 / len(df):.1f}%)")
    print(f"   Entries with 2+ aspects: {multi_2} ({multi_2 * 100 / len(df):.1f}%)")
    print(f"   Entries with 3+ aspects: {multi_3} ({multi_3 * 100 / len(df):.1f}%)")
    print(f"   Average aspects/entry:   {df['aspect_count'].mean():.2f}")
    print(f"   Max aspects in entry:    {df['aspect_count'].max()}")

    # 5. Missing value check
    print(f"\n{'─' * 60}")
    print("5. MISSING VALUE CHECK")
    print(f"{'─' * 60}")
    columns = ["id", "feedback", "aspects", "aspect_sentiments", "source_channel"]
    for col in columns:
        null_count = df[col].isna().sum()
        empty_count = 0
        if col == "feedback":
            empty_count = (df[col].str.strip() == "").sum()
        elif col == "aspects":
            empty_count = (df[col].apply(len) == 0).sum()
        elif col == "aspect_sentiments":
            empty_count = (df[col].apply(len) == 0).sum()
        total_missing = null_count + empty_count
        status = "OK" if total_missing == 0 else "ISSUE"
        print(f"   {col:<20} null={null_count}  empty={empty_count}  [{status}]")

    # 6. Source channel distribution
    print(f"\n{'─' * 60}")
    print("6. SOURCE CHANNEL DISTRIBUTION")
    print(f"{'─' * 60}")
    channel_dist = df["source_channel"].value_counts()
    print(f"\n   {'Channel':<25} {'Count':>6} {'Percentage':>12}")
    print(f"   {'─' * 45}")
    for channel, count in channel_dist.items():
        pct = count * 100 / len(df)
        print(f"   {channel:<25} {count:>6} {pct:>10.1f}%")
    print(f"   {'─' * 45}")
    print(f"   {'TOTAL':<25} {channel_dist.sum():>6} {100.0:>10.1f}%")

    # Bonus: Sentiment distribution
    print(f"\n{'─' * 60}")
    print("7. SENTIMENT DISTRIBUTION (across all aspect-sentiments)")
    print(f"{'─' * 60}")
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    total_sentiments = 0
    for sent_dict in df["aspect_sentiments"]:
        for sentiment in sent_dict.values():
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            total_sentiments += 1
    print(f"\n   {'Sentiment':<15} {'Count':>6} {'Percentage':>12}")
    print(f"   {'─' * 35}")
    for sent, count in sentiment_counts.items():
        pct = count * 100 / total_sentiments
        print(f"   {sent:<15} {count:>6} {pct:>10.1f}%")
    print(f"   {'─' * 35}")
    print(f"   {'TOTAL':<15} {total_sentiments:>6} {100.0:>10.1f}%")

    # Bonus: Aspect frequency
    print(f"\n{'─' * 60}")
    print("8. ASPECT FREQUENCY")
    print(f"{'─' * 60}")
    aspect_freq = {}
    for aspects_list in df["aspects"]:
        for a in aspects_list:
            aspect_freq[a] = aspect_freq.get(a, 0) + 1
    print(f"\n   {'Aspect':<30} {'Count':>6} {'Percentage':>12}")
    print(f"   {'─' * 50}")
    for aspect, count in sorted(aspect_freq.items(), key=lambda x: -x[1]):
        pct = count * 100 / len(df)
        print(f"   {aspect:<30} {count:>6} {pct:>10.1f}%")

    print(f"\n{'=' * 60}")
    print("REPORT COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
