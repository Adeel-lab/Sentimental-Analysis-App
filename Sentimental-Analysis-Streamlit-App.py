#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

lm = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
def load_data():
    train_df = pd.read_csv(r"C:\Users\adeel\Desktop\GitHub\Data_Sets\Emotions dataset for NLP\train.txt", delimiter=";", names=['names', 'label'])
    val_df = pd.read_csv(r"C:\Users\adeel\Desktop\GitHub\Data_Sets\Emotions dataset for NLP\val.txt", delimiter=";", names=['names', 'label'])
    df = pd.concat([train_df, val_df])
    df.reset_index(inplace=True, drop=True)
    return df

# Custom encoder function
def custom_encoder(label):
    if label in ["surprise", "love", "joy"]:
        return 1
    else:
        return 0

# Text transformation function
def text_transformation(item):
    new_item = re.sub('[^a-zA-Z]', ' ', str(item))
    new_item = new_item.lower()
    new_item = new_item.split()
    new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
    return ' '.join(str(x) for x in new_item)

# Main function
def main():
    st.title("Emotions Analysis with Streamlit")

    # Load data
    df = load_data()

    # Display raw data
    st.subheader("Raw Data")
    st.write(df.head())

    # Encode labels
    df['label'] = df['label'].apply(custom_encoder)

    # Display label count plot
    st.subheader("Label Count Plot")
    fig, ax = plt.subplots()
    sns.countplot(x=df['label'], ax=ax)
    st.pyplot(fig)

    # Text transformation
    df['transformed_names'] = df['names'].apply(text_transformation)

    # Word Cloud
    st.subheader("Word Cloud")
    wordcloud = WordCloud(width=800, height=400, max_words=200, min_font_size=12, background_color="white").generate(' '.join(df['transformed_names']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()

    # Model Training
    st.subheader("Model Training")
    cv = CountVectorizer(ngram_range=(1, 2))
    X = cv.fit_transform(df['transformed_names'])
    y = df['label']
    rfc = RandomForestClassifier()
    rfc.fit(X, y)
    st.success("Model training complete!")

    # Prediction
    st.subheader("Prediction")
    text_input = st.text_input("Enter text for sentiment prediction:")
    if st.button("Predict"):
        transformed_input = cv.transform([text_transformation(text_input)])
        prediction = rfc.predict(transformed_input)
        if prediction == 0:
            st.write("Input statement has Negative Sentiment.")
        elif prediction == 1:
            st.write("Input statement has Positive Sentiment.")

if __name__ == "__main__":
    main()
