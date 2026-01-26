# app.py
import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import os

# page config
st.set_page_config(page_title="Email/SMS Spam Classifier", layout="centered", initial_sidebar_state="collapsed")

# simple CSS for card-like look
st.markdown(
    """
    <style>
    .main-card {
        max-width: 800px;
        margin: 40px auto;
        padding: 28px 36px;
        border-radius: 12px;
        background: rgba(255,255,255,0.03);
        box-shadow: 0 6px 20px rgba(0,0,0,0.6);
        color: #e6e6e6;
    }
    .title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .subtitle {
        color: #bfbfbf;
        margin-bottom: 18px;
    }
    .example {
        color: #a8a8a8;
        font-size: 13px;
        margin-top: 6px;
    }
    .stButton>button {
        border-radius: 10px;
        padding: 8px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# header / card container
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="title">📧 Email/SMS Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fast and simple — check if a message is spam or not.</div>', unsafe_allow_html=True)

# NLTK setup (only first time will download)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered = []
    for tok in tokens:
        if tok.isalnum() and tok not in stop_words and tok not in string.punctuation:
            filtered.append(ps.stem(tok))
    return " ".join(filtered)

# load model + vectorizer
VECT_PATH = 'vectorizer.pkl'
MODEL_PATH = 'model.pkl'
if not os.path.exists(VECT_PATH) or not os.path.exists(MODEL_PATH):
    st.error("Missing files: put 'vectorizer.pkl' and 'model.pkl' in the app folder.")
    st.stop()

with open(VECT_PATH, 'rb') as f:
    vectorizer = pickle.load(f)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# check fitted
try:
    check_is_fitted(model)
    model_fitted = True
except (NotFittedError, AttributeError):
    model_fitted = False

# form for input (keeps UI tidy)
with st.form(key="sms_form"):
    sms_input = st.text_area("Enter the message", height=120, placeholder="Type or paste an SMS / email message here...")
    submitted = st.form_submit_button("Predict")

st.markdown('<div class="example"><strong>Example:</strong> Earn ₹50,000/week from home. Click http://scam.link</div>', unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)

# result area
if submitted:
    if not sms_input or sms_input.strip() == "":
        st.warning("Please enter a message to classify.")
    elif not model_fitted:
        st.error("Loaded model is not fitted. Train & save a fitted 'model.pkl'.")
    else:
        with st.spinner("Analyzing message..."):
            processed = transform_text(sms_input)
            if processed.strip() == "":
                st.warning("Message became empty after preprocessing (maybe only stopwords/punctuation). Try another message.")
            else:
                vec = vectorizer.transform([processed])
                pred = model.predict(vec)[0]
                prob = None
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(vec)[0].max()

                if int(pred) == 1:
                    st.success("🚨 Predicted: SPAM")
                    if prob is not None:
                        st.write(f"Confidence: **{prob*100:.1f}%**")
                    st.markdown("> Tip: contains links/activation words like 'claim', 'win', 'free', 'click'.")
                else:
                    st.info("✅ Predicted: NOT SPAM")
                    if prob is not None:
                        st.write(f"Confidence: **{prob*100:.1f}%**")

# small footer and sample quick-test buttons
st.markdown('<hr>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("Sample Spam"):
        st.experimental_set_query_params()  # no-op; using to avoid rerun confusion
        st.write("Use the message below in the box:")
        st.code("Congratulations! You have won ₹5,00,000. Click http://claim-prize.example to receive.")
with col2:
    if st.button("Sample Ham"):
        st.write("Use the message below in the box:")
        st.code("Hey, are we meeting today at 6pm? Call me when you're free.")

st.markdown('</div>', unsafe_allow_html=True)
