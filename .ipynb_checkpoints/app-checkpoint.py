import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

st.set_page_config(page_title='Fake News Prediction App')

def get_model():
    model_nn = load_model('deployment_nn.h5')
    return model_nn

def predict(model, data):
    # Tokenize and pad the input text
    tokenizer = Tokenizer(num_words=20000)  # Assuming 20000 was used during training
    tokenizer.fit_on_texts(data['text'])
    sequences = tokenizer.texts_to_sequences(data['text'])
    padded_data = pad_sequences(sequences, padding='pre', truncating='pre', maxlen=7218)  # Assuming maxlen was used during training

    # Make prediction
    prediction = np.rint(model.predict(padded_data))[0][0]
    return prediction

model = get_model()

st.title("Fake News Detection App")

# Collect user input
text = st.text_area("Enter the text for prediction:", "")
form = st.form('Outcome')
predict_button = form.form_submit_button('Predict')

input_dict = {'text': text}
input_df = pd.DataFrame([input_dict])

if predict_button:
    output = predict(model, input_df)

    if output == 0:
        st.write("Legitimate Information")
    else:
        st.write("Fake Information")