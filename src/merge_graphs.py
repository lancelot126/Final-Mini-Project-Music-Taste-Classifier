"""
Merge Kaggle "Spotify Tracks Dataset" audio features into song_pool.csv.

Usage:
    python merge_features.py song_pool.csv dataset.csv

Outputs (in the current folder):
    song_pool_merged.csv  - your song table with features filled in where matched
    match_review.csv      - every song with match status/score; CHECK the non-exact rows
    unmatched.csv         - songs not found (replace these in your pool)
"""
import re, sys, unicodedata, difflib
import pandas as pd

FEATURES = ["danceability", "energy", "key", "loudness", "mode", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo",
            "time_signature"]
FUZZY_THRESHOLD = 0.85


def norm(s):
    """lowercase, strip accents, drop remaster/version suffixes and punctuation."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\s-\s.*$", "", s)          # " - Remastered 2011", " - Radio Edit"
    s = re.sub(r"[\(\[].*?[\)\]]", "", s)   # "(feat. X)", "(1st mvt)"
    s = re.sub(r"[.'’]", "", s)             # b.i.g. -> big, don't -> dont
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def artist_names(artist):
    parts = re.split(r"\s+(?:ft\.?|feat\.?|featuring|&|and)\s+|,|;", str(artist), flags=re.I)
    return [norm(p) for p in parts if norm(p)]


def main(songs_path, kaggle_path):
    songs = pd.read_csv(songs_path, dtype=str, keep_default_na=False)
    kg = pd.read_csv(kaggle_path)
    kg = kg.drop(columns=[c for c in kg.columns if c.startswith("Unnamed")], errors="ignore")

    # The Kaggle file lists the same track once per genre/album -> keep most popular copy.
    kg["t_norm"] = kg["track_name"].map(norm)
    kg["a_norm"] = kg["artists"].map(norm)
    kg = (kg.sort_values("popularity", ascending=False)
            .drop_duplicates(subset=["t_norm", "a_norm"]).reset_index(drop=True))

    review, merged = [], songs.astype(object).copy()
    for c in ["kaggle_track_id", "kaggle_popularity"]:
        merged[c] = pd.Series([""] * len(merged), dtype=object)
    for i, row in songs.iterrows():
        t, names = norm(row["title"]), artist_names(row["artist"])
        # candidates: any listed artist appears in the dataset's artist string
        mask = pd.Series(False, index=kg.index)
        for n in names:
            mask |= kg["a_norm"].str.contains(n, regex=False)
        cand = kg[mask]
        best, score, status = None, 0.0, "none"
        if len(cand):
            sims = cand["t_norm"].map(lambda x: difflib.SequenceMatcher(None, t, x).ratio())
            j = sims.idxmax()
            score = float(sims[j])
            if score >= 0.999:
                best, status = cand.loc[j], "exact"
            elif score >= FUZZY_THRESHOLD:
                best, status = cand.loc[j], "fuzzy"
        rec = {"song_id": row["song_id"], "title": row["title"], "artist": row["artist"],
               "status": status, "score": round(score, 3),
               "matched_title": best["track_name"] if best is not None else "",
               "matched_artist": best["artists"] if best is not None else "",
               "matched_album": best["album_name"] if best is not None else ""}
        review.append(rec)
        if best is not None:
            for f in FEATURES:
                merged.at[i, f] = best[f]
            merged.at[i, "explicit"] = int(bool(best["explicit"]))
            merged.at[i, "duration_sec"] = round(best["duration_ms"] / 1000)
            merged.at[i, "in_kaggle_dataset"] = 1
            merged.at[i, "kaggle_track_id"] = best["track_id"]
            merged.at[i, "kaggle_popularity"] = best["popularity"]
        else:
            merged.at[i, "in_kaggle_dataset"] = 0

    review = pd.DataFrame(review)
    merged.to_csv("song_pool_merged.csv", index=False, encoding="utf-8-sig")
    review.to_csv("match_review.csv", index=False, encoding="utf-8-sig")
    review[review.status == "none"].to_csv("unmatched.csv", index=False, encoding="utf-8-sig")
    print(review.status.value_counts().to_string())
    print(f"\nMatched {int((review.status != 'none').sum())}/{len(review)} songs.")
    print("Open match_review.csv and check every 'fuzzy' row plus a sample of 'exact' rows.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])