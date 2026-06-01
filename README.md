# Duplicate Question Detection using Shared LSTM

An end-to-end Duplicate Question Detection web application built using Deep Learning and Natural Language Processing (NLP). The system identifies whether two questions have the same semantic meaning by leveraging a Siamese-style Shared LSTM architecture. Questions are preprocessed, tokenized, converted into padded sequences, and passed through a shared neural network that learns meaningful representations of both inputs. Similarity features are extracted using absolute difference and element-wise multiplication before generating a duplicate probability score. The project is developed using TensorFlow, Keras, and Streamlit.

## Features

* Deep Learning based duplicate question detection
* Shared LSTM (Siamese Network) architecture
* Text preprocessing and normalization
* Tokenization and sequence padding
* Semantic similarity learning
* Real-time duplicate prediction
* Streamlit based interactive UI
* Deployed web application

## Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-learn
* Streamlit
* Pickle

## Project Structure

```text
├── app.py
├── tokenizer1.pkl
├── similar.h5
├── requirements.txt
├── runtime.txt
└── README.md
```

## Workflow

The duplicate detection pipeline follows:

```text
Question 1
     ↓
Text Preprocessing
     ↓
Tokenization
     ↓
Padding
     ↓
Shared Embedding Layer
     ↓
Shared LSTM Encoder
     ↓
Feature Interaction
(Absolute Difference + Element-wise Multiplication)
     ↓
Dense Layers
     ↓
Duplicate Probability Score
```

The same process is applied to both questions using shared weights.

## NLP Preprocessing

Text preprocessing includes:

* Lowercasing
* URL removal
* Special character removal
* Whitespace normalization
* Tokenization
* Sequence padding

## Shared LSTM Architecture

The model uses a Siamese-style architecture where both questions pass through the same neural network.

```python
# Shared LSTM encoder
lstm = LSTM(128)

encoded_q1 = lstm(embedded_q1)
encoded_q2 = lstm(embedded_q2)

# Similarity features
abs_diff = tf.abs(encoded_q1 - encoded_q2)
mul_sim = tf.multiply(encoded_q1, encoded_q2)

# Concatenate features
merged = Concatenate()([abs_diff, mul_sim])
```

This architecture helps the model learn semantic relationships between question pairs.

## Running the Project

Clone the repository:

```bash
git clone https://github.com/YASHARTH1630/quora-duplicate-question-detection.git
```

Move into project directory:

```bash
cd quora-duplicate-question-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit app:

```bash
streamlit run app.py
```

## Requirements

```text
streamlit
tensorflow
numpy
pandas
scikit-learn
pickle-mixin
```

## Model Highlights

* Shared weights for both questions
* Learns semantic similarity between sentence pairs
* Handles variable-length questions through padding
* Efficient architecture for duplicate question detection
* Suitable for FAQ systems, search engines, and chatbot applications

## Deployment

The application is deployed using:

* Streamlit Community Cloud

🌐 Live Demo

https://quora-duplicate-question-detection-x8wzhhw9sr35mvle63tyha.streamlit.app/

Required deployment files:

```text
app.py
requirements.txt
runtime.txt
tokenizer1.pkl
similar.h5
```

## Future Improvements

* FastText embeddings
* Attention mechanism
* Sentence-BERT embeddings
* Transformer-based duplicate detection
* API deployment
* Batch question comparison

## Author

### YASHARTH YASH

📧 Email: [yasharththysu@gmail.com](mailto:yasharththysu@gmail.com)

💼 LinkedIn: https://www.linkedin.com/in/yasharth-yash-5b072437b/
