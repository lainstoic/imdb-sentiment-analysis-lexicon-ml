"""
02_sentiwordnet_tfidf_classifier.py
Sentiment analysis assignment - Part 2 (Python)
  1. Apply SentiWordNet to the cleaned reviews produced by the R script
  2. Auto-label the data using the Bing lexicon scores (from R)
  3. TF-IDF + Logistic Regression classifier, with evaluation

Setup:  pip install nltk scikit-learn pandas matplotlib
Input:  reviews_clean.csv  (created by 01_preprocess_and_lexicons.R)
"""
import nltk
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.corpus import sentiwordnet as swn, wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)

# ---- Download required NLTK resources (only needed once) --------------
for pkg in ["wordnet", "sentiwordnet", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng", "punkt", "punkt_tab", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# ---- Load data ---------------------------------------------------------
df = pd.read_csv("reviews_clean.csv")
df["clean_text"] = df["clean_text"].fillna("")
print("Loaded", len(df), "reviews")

# =======================================================================
# PART A: SentiWordNet
# =======================================================================
lemmatizer = WordNetLemmatizer()

def penn_to_wn(tag):
    """Map Penn Treebank POS tags to WordNet POS tags."""
    if tag.startswith("J"): return wn.ADJ
    if tag.startswith("V"): return wn.VERB
    if tag.startswith("N"): return wn.NOUN
    if tag.startswith("R"): return wn.ADV
    return None

def swn_score(text):
    """Document score = sum over words of (PosScore - NegScore) using the
    first (most common) synset of each word, matched by POS tag."""
    total = 0.0
    for word, tag in nltk.pos_tag(nltk.word_tokenize(text)):
        wn_pos = penn_to_wn(tag)
        if wn_pos is None:
            continue
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        synsets = list(swn.senti_synsets(lemma, wn_pos))
        if synsets:
            total += synsets[0].pos_score() - synsets[0].neg_score()
    return total

print("Computing SentiWordNet scores (this can take a minute or two)...")
df["swn_score"] = df["clean_text"].apply(swn_score)
df["swn_label"] = df["swn_score"].apply(
    lambda s: "positive" if s > 0 else ("negative" if s < 0 else "neutral"))

print("\n--- SentiWordNet sample output ---")
print(df[["id", "original_label", "swn_score", "swn_label"]].head(10).to_string(index=False))
print("\nSentiWordNet label distribution:\n", df["swn_label"].value_counts())

mask = df["swn_label"] != "neutral"
print("SentiWordNet agreement with original labels: %.3f"
      % (df.loc[mask, "swn_label"] == df.loc[mask, "original_label"]).mean())

df.to_csv("reviews_with_swn.csv", index=False)

# =======================================================================
# PART B: Auto-labelling with Bing + TF-IDF + classifier
# =======================================================================
# Labels come from the Bing lexicon (computed in R). Neutral (score 0)
# reviews are dropped so we have a clean binary problem.
labelled = df[df["bing_label"] != "neutral"].copy()
print("\nReviews kept after dropping neutral:", len(labelled))
print(labelled["bing_label"].value_counts())

X_train, X_test, y_train, y_test, orig_train, orig_test = train_test_split(
    labelled["clean_text"], labelled["bing_label"], labelled["original_label"],
    test_size=0.2, random_state=42, stratify=labelled["bing_label"])

# TF-IDF representation (fit on TRAIN only to avoid data leakage)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("TF-IDF matrix shape (train):", X_train_tfidf.shape)

# Classifier: Logistic Regression
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_tfidf, y_train)
pred = clf.predict(X_test_tfidf)

print("\n=== Test results (against Bing-lexicon labels) ===")
print("Accuracy: %.4f" % accuracy_score(y_test, pred))
print(classification_report(y_test, pred, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, pred, labels=["negative", "positive"]))

ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=["negative", "positive"])
plt.title("Logistic Regression + TF-IDF (Bing labels)")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")

# Extra check: how does the model do against the ORIGINAL human labels?
print("Accuracy vs original IMDB labels on the same test reviews: %.4f"
      % accuracy_score(orig_test, pred))

# Most influential words
names = tfidf.get_feature_names_out()
coefs = clf.coef_[0]          # classes_ = ['negative','positive']
top = pd.Series(coefs, index=names).sort_values()
print("\nTop negative features:", list(top.index[:10]))
print("Top positive features:", list(top.index[-10:][::-1]))
