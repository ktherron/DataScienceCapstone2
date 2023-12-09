{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e912a961-e555-4e54-a3fc-0a4855ff09f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "from keras.preprocessing.sequence import pad_sequences\n",
    "from keras.models import load_model\n",
    "from keras.preprocessing.text import Tokenizer\n",
    "\n",
    "# Load the trained neural network model\n",
    "model_nn = load_model('neural_network_model.h5')\n",
    "\n",
    "# Load the Tokenizer used during training\n",
    "tokenizer = Tokenizer(num_words=20000)  # Adjust to vocabulary size\n",
    "tokenizer.fit_on_texts([])  # Provide empty texts to fit on\n",
    "\n",
    "# Streamlit app\n",
    "def main():\n",
    "    st.title(\"Fake News Detection App\")\n",
    "\n",
    "    # User input for text\n",
    "    user_input = st.text_area(\"Enter the text for prediction:\", \"\")\n",
    "\n",
    "    if st.button(\"Predict\"):\n",
    "        if user_input:\n",
    "            # Tokenize and pad the user input\n",
    "            sequences = tokenizer.texts_to_sequences([user_input])\n",
    "            data = pad_sequences(sequences, padding='pre', truncating='pre', maxlen=5000)\n",
    "\n",
    "            # Make prediction\n",
    "            prediction = np.rint(model_nn.predict(data))[0][0]\n",
    "\n",
    "            # Display the result\n",
    "            if prediction == 1:\n",
    "                st.success(\"This text is predicted as Fake News.\")\n",
    "            else:\n",
    "                st.success(\"This text is predicted as Not Fake News.\")\n",
    "        else:\n",
    "            st.warning(\"Please enter some text for prediction.\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
