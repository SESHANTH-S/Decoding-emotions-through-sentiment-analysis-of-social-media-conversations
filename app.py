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