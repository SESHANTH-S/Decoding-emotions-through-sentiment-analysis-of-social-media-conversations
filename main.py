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