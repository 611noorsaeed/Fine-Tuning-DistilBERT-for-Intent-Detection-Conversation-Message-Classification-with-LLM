import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification  # HuggingFace Transformers for tokenization and model
import torch
# clean function
import nltk
from nltk.corpus import stopwords


# load save model
model = DistilBertForSequenceClassification.from_pretrained("distilbert_model")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert_model")

# Download NLTK stopwords (only need to do this once)
nltk.download('stopwords')
# Load the list of stopwords
stop_words = set(stopwords.words('english'))


# Preprocessing function: convert text to lowercase and remove stopwords
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove stopwords: split the text, filter out stopwords, and join back
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

# Function to make a prediction
def predict(text, model, tokenizer, max_length=21):
    # Preprocess the input text
    text = preprocess_text(text)
    # Tokenize the input text
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")

    # Make prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)  # Get model output
        logits = outputs.logits  # Get logits from the output

    # Get the predicted label (highest logit)
    predicted_class_id = torch.argmax(logits, dim=-1).item()  # Get the index of the max logit
    return predicted_class_id


# UI app...........
st.title("Fine Tuned Distil Bert For Intent detection")

text = st.text_input("type your message here")

if st.button("Predict Intent"):
    if text:
        id = predict(text, model, tokenizer)

        # our classes names
        # Define the intent labels
        id_to_label = {
            0: 'get weather',
            1: 'search creative work',
            2: 'search screening event',
            3: 'add to playlist',
            4: 'book restaurant',
            5: 'rate book',
            6: 'play music'
        }

        intent = id_to_label.get(id, "unknown intent")

        st.write("Predicted Intent  :", intent)
    else:
        st.write("Type your message....")

