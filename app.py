import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------

st.title("🎬 Movie Review Sentiment Analysis System")
st.subheader("Deep Learning Based Sentiment Classification")

# -----------------------------
# LOAD MODELS
# -----------------------------

@st.cache_resource
def load_models():

    simple_rnn = tf.keras.models.load_model(
        "simple_rnn_model.keras"
    )

    lstm = tf.keras.models.load_model(
        "lstm_model.keras"
    )

    gru = tf.keras.models.load_model(
        "gru_model.keras"
    )

    return simple_rnn, lstm, gru

simple_rnn_model, lstm_model, gru_model = load_models()

# -----------------------------
# WORD INDEX
# -----------------------------

word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

MAXLEN = 256

# -----------------------------
# PREPROCESSING
# -----------------------------

def preprocess_review(text):

    text = text.lower()

    words = text.split()

    encoded = [
        word_index.get(word, 2)
        for word in words
    ]

    encoded = [1] + encoded

    padded = pad_sequences(
        [encoded],
        maxlen=MAXLEN
    )

    return padded

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("Model Selection")

selected_model = st.sidebar.radio(
    "Choose Model",
    ["SimpleRNN", "LSTM", "GRU"]
)

# -----------------------------
# INPUT
# -----------------------------

review = st.text_area(
    "Enter your movie review here...",
    height=200
)

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    processed_review = preprocess_review(review)

    models = {
        "SimpleRNN": simple_rnn_model,
        "LSTM": lstm_model,
        "GRU": gru_model
    }

    selected = models[selected_model]

    prediction = selected.predict(
        processed_review,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if prediction > 0.5
        else "Negative"
    )

    confidence = (
        prediction
        if prediction > 0.5
        else 1 - prediction
    )

    positive_prob = float(prediction)
    negative_prob = float(1 - prediction)

    st.markdown("---")

    st.subheader("Prediction Result")

    st.success(
        f"Sentiment: {sentiment}"
    )

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    # -------------------------
    # Probability Graph
    # -------------------------

    st.subheader("Probability Distribution")

    prob_df = pd.DataFrame({
        "Sentiment":[
            "Positive",
            "Negative"
        ],
        "Probability":[
            positive_prob,
            negative_prob
        ]
    })

    fig = px.bar(
        prob_df,
        x="Sentiment",
        y="Probability",
        text="Probability"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------
    # Confidence Progress
    # -------------------------

    st.subheader("Confidence Chart")

    st.progress(
        float(confidence)
    )

    st.write(
        f"Confidence Score: {confidence*100:.2f}%"
    )

    # -------------------------
    # Compare Models
    # -------------------------

    st.markdown("---")
    st.subheader("Compare Predictions From All Models")

    comparison = []

    for name, mdl in models.items():

        pred = mdl.predict(
            processed_review,
            verbose=0
        )[0][0]

        label = (
            "Positive"
            if pred > 0.5
            else "Negative"
        )

        conf = (
            pred
            if pred > 0.5
            else 1 - pred
        )

        comparison.append({
            "Model": name,
            "Prediction": label,
            "Confidence (%)": round(
                conf * 100,
                2
            )
        })

    comparison_df = pd.DataFrame(
        comparison
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    fig2 = px.bar(
        comparison_df,
        x="Model",
        y="Confidence (%)",
        color="Prediction",
        text="Confidence (%)"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )