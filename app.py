import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')

data = pd.read_csv('news.csv')
data.head()

##Preprocessing
data = data.drop(["Unnamed: 0"],axis=1)
data.head(5)

##data encoding
le = preprocessing.LabelEncoder()
data['label'] = le.fit_transform(data['label'])


ps = PorterStemmer()
def stemming(title):
    if title is None:
        return ""

    stem_title = re.sub('[^a-zA-Z]', " ", title)
    stem_title = stem_title.lower()
    stem_title = stem_title.split()
    stem_title = [ps.stem(word) for word in stem_title 
                  if word not in stopwords.words('english')]
    return ' '.join(stem_title)


data['title'] = data['title'].apply(stemming)


X = data['title'].values
y = data['label'].values


vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2 ,stratify=y ,random_state=1)


model = LogisticRegression()
model.fit(X_train , y_train)




#website

# import streamlit as st

# st.title("Fake news Classification:")
# input_text = st.text_input('Enter new Article:')

# def prediction(input_text):
#     input_data = vector.transform([input_text])
#     prediction = model.predict(input_data)
#     return prediction[0]

# if input_text:
#     pred = prediction(input_text)
#     if(pred==1):
#         st.write("The news is fake")
#     else:
#         st.write("the news is real")

import streamlit as st

# ---------- Page Config ----------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #ffffff;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #a1a1aa;
    margin-bottom: 30px;
}
.result-fake {
    background-color: #2a0f14;
    padding: 15px;
    border-radius: 10px;
    color: #ff4b4b;
    font-size: 20px;
    text-align: center;
}
.result-real {
    background-color: #0f2a1a;
    padding: 15px;
    border-radius: 10px;
    color: #4bff88;
    font-size: 20px;
    text-align: center;
}
textarea {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("<div class='title'>📰 Fake News Classification</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered news authenticity checker</div>", unsafe_allow_html=True)

# ---------- Input ----------
input_text = st.text_area(
    "Enter the news article below:",
    height=180,
    placeholder="Paste or type the news article here..."
)

# ---------- Prediction Function ----------
def prediction(text):
    input_data = vector.transform([text])
    pred = model.predict(input_data)
    return pred[0]

# ---------- Button ----------
if st.button("🔍 Analyze News", use_container_width=True):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        result = prediction(input_text)

        if result == 1:
            st.markdown(
                "<div class='result-fake'>🚨 This news is FAKE</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-real'>✅ This news is REAL</div>",
                unsafe_allow_html=True
            )

# ---------- Footer ----------
st.markdown("<br><hr style='border:0.5px solid #333'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>Built with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)
