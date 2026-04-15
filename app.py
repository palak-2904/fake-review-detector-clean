import streamlit as st
import joblib
import os
import pandas as pd

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Fake Review Detector", layout="wide")

st.title("🕵️ Smart Fake Review Detector")

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------------
# INPUT
# -------------------------------
review = st.text_area("✍️ Enter your review:")

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Analyze Review"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review")

    else:
        review_lower = review.lower()

        # -------------------------------
        # HYBRID RULE SYSTEM 🔥
        # -------------------------------
        if review_lower.count("very") >= 3 or review_lower.count("!") >= 2:
            pred = 0
            prob = [0.9, 0.1]

        elif any(phrase in review_lower for phrase in [
            "highly recommend",
            "excellent product",
            "great quality",
            "works perfectly",
            "amazing product"
        ]):
            pred = 0
            prob = [0.8, 0.2]

        elif "but" in review_lower or "however" in review_lower:
            pred = 1
            prob = [0.2, 0.8]

        else:
            vec = vectorizer.transform([review])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]

        # -------------------------------
        # SMART CONFIDENCE SYSTEM 🧠
        # -------------------------------
        confidence = max(prob)

        # Boost confidence
        if review_lower.count("very") >= 3:
            confidence += 0.2

        if review_lower.count("!") >= 2:
            confidence += 0.2

        if "but" in review_lower or "however" in review_lower:
            confidence += 0.15

        if len(review.split()) > 12:
            confidence += 0.1

        # Reduce confidence
        if len(review.split()) < 5:
            confidence -= 0.2

        # Clamp (IMPORTANT FIX)
        confidence = max(0, min(confidence, 0.95)) * 100

        # -------------------------------
        # RESULT DISPLAY ✅
        # -------------------------------
        col1, col2 = st.columns(2)

        with col1:
            if confidence < 50:
                st.warning("🤔 VERY LOW CONFIDENCE")
            elif confidence < 65:
                st.info("⚠️ LOW CONFIDENCE")
            elif pred == 1:
                st.success("✅ REAL REVIEW")
            else:
                st.error("❌ FAKE REVIEW")

            st.metric("Confidence", f"{confidence:.2f}%")

            # Progress bar
            st.progress(int(confidence))

        # -------------------------------
        # PROBABILITY CHART 📊
        # -------------------------------
        with col2:
            chart = pd.DataFrame({
                "Type": ["Fake", "Real"],
                "Probability": [prob[0], prob[1]]
            })
            st.bar_chart(chart.set_index("Type"))

        # -------------------------------
        # EXPLANATION 🔍
        # -------------------------------
        st.subheader("🧠 Why this result?")

        if "but" in review_lower:
            st.info("✔ Mixed opinion detected → likely real")

        if review_lower.count("very") >= 3:
            st.warning("⚠ Repetition detected")

        if review_lower.count("!") >= 2:
            st.warning("⚠ Too many exclamation marks")

        if len(review.split()) < 5:
            st.warning("⚠ Very short review")

        if any(phrase in review_lower for phrase in [
            "highly recommend",
            "great quality",
            "excellent product"
        ]):
            st.warning("⚠ Generic promotional language detected")

# -------------------------------
# DATASET INSIGHTS 📊
# -------------------------------
st.markdown("---")
st.subheader("📊 Dataset Insights")

data_path = os.path.join(BASE_DIR, "data", "kindle_reviews.csv")

try:
    df = pd.read_csv(data_path)

    st.success("✅ Dataset loaded")
    st.write("Total Reviews:", len(df))

    if 'overall' in df.columns:
        st.subheader("⭐ Rating Distribution")
        st.bar_chart(df['overall'].value_counts().sort_index())

        st.subheader("📊 Average Rating")
        st.metric("Average Rating", round(df['overall'].mean(), 2))

    if 'reviewText' in df.columns:
        st.subheader("📝 Sample Reviews")
        st.dataframe(df[['reviewText']].head(5))

except Exception as e:
    st.error(f"Error loading dataset: {e}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("🚀 Smart Fake Review Detector | ML + Rule-based AI")