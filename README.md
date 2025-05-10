# Decoding Emotions: Sentiment Analysis of Social Media Conversations

This project aims to detect and decode human emotions from social media text using sentiment analysis techniques and machine learning models.

## Features
- Preprocess social media text (removing emojis, mentions, hashtags, URLs, etc.)
- Use of NLP techniques (TF-IDF/word embeddings)
- Sentiment/emotion classification using ML/DL models
- Interactive web app with Gradio

## Technologies Used
- Python
- Pandas, Numpy
- Scikit-learn
- NLTK / SpaCy
- TensorFlow / Keras (optional)
- Gradio for deployment

## Usage
1. Clone the repo
2. Install dependencies
3. Run `main.py` or `app.py`
4. Visit the local Gradio link

## License
[MIT](LICENSE)

pandas
numpy
scikit-learn
nltk
gradio
tensorflow  # optional if using DL

import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'@\w+|#\w+|http\S+', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return ' '.join(tokens)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2)

    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    return model
import gradio as gr
import pandas as pd
from preprocess import clean_text
from joblib import load

model = load("sentiment_model.joblib")

def predict_emotion(text):
    cleaned = clean_text(text)
    return model.predict([cleaned])[0]

iface = gr.Interface(fn=predict_emotion, inputs="text", outputs="label",
                     title="Emotion Decoder", description="Enter a social media post to detect the emotion.")

iface.launch()
import pandas as pd
from preprocess import clean_text
from model import train_model
from joblib import dump

# Load and preprocess your dataset (assume it has 'text' and 'label' columns)
df = pd.read_csv("social_media_emotions.csv")
df['text'] = df['text'].apply(clean_text)

model = train_model(df)

# Save model
dump(model, 'sentiment_model.joblib')

