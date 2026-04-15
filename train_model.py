import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("data/kindle_reviews.csv")
df = df[['reviewText', 'overall']].dropna()

# -------------------------------
# LABELING (SMART LOGIC ✅)
# -------------------------------
def label_review(row):
    text = str(row['reviewText']).lower()

    # 🚨 FAKE PATTERNS
    if text.count("very") >= 3:
        return 0
    if text.count("!") >= 2:
        return 0
    if len(text.split()) < 4:
        return 0
    if len(set(text.split())) < len(text.split()) / 2:
        return 0

    # ✅ REAL PATTERNS
    if "but" in text or "however" in text:
        return 1

    return 1  # default

df['label'] = df.apply(label_review, axis=1)

# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df['cleaned'] = df['reviewText'].apply(clean_text)

# -------------------------------
# TRAIN MODEL
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label'], test_size=0.2
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

# -------------------------------
# SAVE
# -------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained successfully!")