import os
import ssl
import nltk
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import json

with open("intents.json", "r") as f:
    data = json.load(f)
    intents = data['intents']



ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

vectorizer = TfidfVectorizer()
#classification
clf = LogisticRegression(random_state=0, max_iter=1000)

#processing the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)  
y = tags
clf.fit(x, y)

#Create Nerdbot 

def nerdbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            responses = random.choice(intent['responses'])
            return responses
    return "Sorry, I don't understand that."


# Streamlit app
counter = 0
def main():
    st.title("NerdBot")
    st.write("Ask me anything!")

    if "counter" not in st.session_state:
        st.session_state.counter = 0

    st.session_state.counter += 1
    user_input = st.text_input("You:",key=f"user_input_{counter}")

    if user_input:
        response = nerdbot(user_input)
        st.text_area(f"NerdBot:",value=response,height=100,max_chars=1000,key=f"nerdbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye', 'exit']:
            st.text_area("NerdBot:",value="Goodbye! Have a great day!",height=100,max_chars=1000,key=f"nerdbot_response_{counter+1}")
            st.stop()

if __name__ == "__main__": 
    main()   
    
