import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf= pickle.load(open('vectorizer.pkl','rb'))

model=pickle.load(open('model.pkl','rb'))

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')
ps = PorterStemmer

def transform_message(message):
     message=message.lower()
     message=nltk.word_tokenize(message)

     message=(word for word in message if word.isalnum())

     message=[word for word in message if word not in stopwords.words('english') and word not in string.punctuation]

     message=[ps.stem(word) for word in message]
     return " ".join(message)
     
st.title("Email Spam Classifier")
input_email=st.text_area("Enter Email")

if st.button('predict'):
    transformed_email= transform_message(input_email)
    vector_input=tfidf.transform([transformed_email])
    result=model.predict(vector_input)[0]
    if result==1:
         st.header("spam")
    else:
         st.header("Not Spam")


