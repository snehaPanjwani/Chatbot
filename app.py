import os
import json
import datetime
import random
import nltk
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup SSL and nltk data path
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Step 1: Create the vectorizer and classifier
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english')
classifier = LogisticRegression()

# Step 2: Preprocess the data
def preprocess_data(intents):
    X = []
    y = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            X.append(pattern)
            y.append(intent['tag'])
    return X, y

# Step 3: Train the model
def train_model(X, y):
    X_vectors = vectorizer.fit_transform(X)
    classifier.fit(X_vectors, y)

X, y = preprocess_data(intents)
train_model(X, y)

# Step 4: Create the chatbot function
def chatbot_response(user_input):
    X_input = vectorizer.transform([user_input])
    tag = classifier.predict(X_input)[0]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "I'm sorry, I couldn't understand that. Could you please rephrase?"

# Step 5: Create the main function
def main():
    st.title("Museum Booking Chatbot")
    st.write("Hello! I'm here to assist you with your museum visit.")

    # Conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    user_query = st.text_input("Enter your query:")

    if user_query:
        response = chatbot_response(user_query)
        st.session_state.conversation.append(("User", user_query))
        st.session_state.conversation.append(("Bot", response))

    # Display conversation history
    st.subheader("Conversation History")
    for speaker, message in st.session_state.conversation:
        st.write(f"**{speaker}:** {message}")

# Step 6: Conversation history menu
def conversation_history_menu():
    if st.sidebar.button("Clear History"):
        st.session_state.conversation = []
        st.sidebar.write("Conversation history cleared.")

# Run the app
if __name__ == "__main__":
    conversation_history_menu()
    main()
