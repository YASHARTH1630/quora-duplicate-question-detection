import os
import requests
import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load model
MODEL_URL = "https://huggingface.co/YASHARTH1630/fake-news-model/resolve/main/similar.h5?download=true"

MODEL_PATH = "similar.h5"
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
def abs_diff(x):
    return tf.abs(x[0] - x[1])
def mul_sim(x):
    return tf.multiply(x[0], x[1])
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "abs_diff": abs_diff,
        "mul_sim": mul_sim,
    },
)

# Load tokenizer
with open("tokenizer1.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 300   

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("Duplicate Question Detection")

q1 = st.text_area("Question 1")
q2 = st.text_area("Question 2")

if st.button("Check Similarity"):

    q1 = clean_text(q1)
    q2 = clean_text(q2)

    seq1 = tokenizer.texts_to_sequences([q1])
    seq2 = tokenizer.texts_to_sequences([q2])

    pad1 = pad_sequences(seq1, maxlen=MAX_LEN)
    pad2 = pad_sequences(seq2, maxlen=MAX_LEN)

    prob = model.predict([pad1, pad2], verbose=0)[0][0]

    if prob > 0.38:
        st.success(f"Duplicate ({prob:.2%})")
    else:
        st.error(f"Not Duplicate ({1-prob:.2%})")
