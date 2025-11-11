import streamlit as st
import re
import nltk
import joblib
import base64
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Set page config and design
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Cached model loader
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# ✅ Preprocessing class
class Preprocessing:
    def __init__(self, text):
        self.text = text

    def clean_text(self):
        lm = WordNetLemmatizer()
        review = re.sub('[^a-zA-Z0-9]', ' ', self.text)
        review = review.lower().split()
        review = [lm.lemmatize(word) for word in review if word not in stopwords.words('english')]
        return ' '.join(review)

# ✅ Prediction class
class Prediction:
    def __init__(self, text, model, vectorizer):
        self.text = text
        self.model = model
        self.vectorizer = vectorizer

    def predict(self):
        clean_text = Preprocessing(self.text).clean_text()
        vectorized_text = self.vectorizer.transform([clean_text])
        pred = self.model.predict(vectorized_text)
        return "Real" if pred[0] == 1 else "Fake"


# =============== 🌆 Custom Styling Section ===============

@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Image file '{bin_file}' not found. Please make sure it's in the same folder as app.py.")
        return None

# Load the image
img_b64 = get_base64_of_bin_file("news.png")

# Apply styling only if the image was found
if img_b64:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{img_b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: "Georgia", serif;
        }}

        /* --- CARD 1 (Main) --- */
        div[data-testid="stContainer"]:nth-of-type(1) {{
            background: rgba(255, 255, 255, 0.9); /* Faded background */
            padding: 30px;
            border-radius: 18px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
            margin-top: 40px;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }}
        div[data-testid="stContainer"]:nth-of-type(1):hover {{
            transform: translateY(-6px);
            box-shadow: 0 18px 40px rgba(0,0,0,0.35);
        }}

        /* --- CARD 2 (Examples) --- */
        div[data-testid="stContainer"]:nth-of-type(2) {{
            background: rgba(255, 255, 255, 0.9); /* Faded background */
            border-radius: 12px;
            padding: 18px;
            margin-top: 40px;
            box-shadow: 0 6px 25px rgba(0,0,0,0.15);
        }}
        
        /* --- 🌟 NEW FIXES 🌟 --- */

        /* 1. Fix "News Headline" label color */
        label[data-testid="stWidgetLabel"] p {{
            color: #1e1e1e !important;
            font-weight: 600 !important;
        }}
        
        /* 2. Fix Text Area input color */
        textarea[aria-label="News Headline"] {{
            border-radius:10px !important;
            background:#fafafa !important;
            border:1px solid #aaa !important;
            font-size:16px !important;
            padding:12px !important;
            color: #1e1e1e !important; /* Input text color */
        }}

        /* 3. Fix "Example..." title color */
        div[data-testid="stContainer"]:nth-of-type(2) h3 {{
             color: #111 !important;
        }}
        
        /* --- Other Styles (No changes needed) --- */
        h1 {{
            text-align:center;
            color:#1e1e1e;
            margin-bottom:4px;
        }}
        .subtitle {{
            text-align:center;
            color:#444;
            margin-bottom:20px;
        }}
        div.stButton > button:first-child {{
            display:block;
            margin: 18px auto 0 auto;
            background: linear-gradient(180deg,#b22222,#8b1a1a);
            color:white;
            border:none;
            padding:12px 42px;
            border-radius:12px;
            font-size:18px;
            cursor:pointer;
            transition: all 0.25s ease;
            box-shadow: 0 6px 16px rgba(178,34,34,0.3);
        }}
        div.stButton > button:first-child:hover {{
            background:#ff4040;
            transform:scale(1.06);
            box-shadow:0 10px 25px rgba(0,0,0,0.3);
        }}
        .result-text {{
            text-align:center;
            font-weight:bold;
            font-size:20px;
            margin-top:16px;
        }}
        .fake-item {{
            display:flex;
            justify-content: space-between;
            align-items:center;
            padding:10px 12px;
            border-radius:8px;
            margin-bottom:8px;
            transition: all 0.2s ease;
            text-decoration:none;
            color:inherit;
        }}
        .fake-item:hover {{
            background:#fff6f6;
            transform: translateY(-3px);
            box-shadow:0 6px 15px rgba(0,0,0,0.1);
        }}
        .fake-title {{
            color:#b22222;
            font-weight:700;
            font-size:15px;
        }}
        .fake-desc {{
            color:#333;
            font-size:13px;
        }}
        .fake-link {{
            color:#1a73e8;
            font-size:13px;
            text-decoration:underline;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============== 📰 Frontend Layout ===============
# Create the first container (for the main card)
container_1 = st.container()
container_1.markdown("<h1>📰 Fake News Detection</h1>", unsafe_allow_html=True)
container_1.markdown('<p class="subtitle">Enter a news headline below and check if it’s Real or Fake.</p>', unsafe_allow_html=True)

user_input = container_1.text_area("News Headline", placeholder="Type or paste your news headline here...")

if container_1.button("Predict"):
    if user_input.strip() == "":
        container_1.warning("⚠ Please enter a news headline!")
    else:
        result = Prediction(user_input, model, vectorizer).predict()
        if result == "Fake":
            container_1.markdown('<div class="result-text" style="color:#b22222">❌ The news appears to be FAKE.</div>', unsafe_allow_html=True)
        else:
            container_1.markdown('<div class="result-text" style="color:#2e7d32">✅ The news appears to be REAL.</div>', unsafe_allow_html=True)

# =============== 🕵 Fake News Section ===============
# Create the second container (for the fake news examples)
container_2 = st.container()
container_2.markdown("<h3 style='margin-bottom:14px;'>🕵 Example Fake News (Click to read full story)</h3>", unsafe_allow_html=True)

fake_news_links = [
    (
        "Aliens landed near Eiffel Tower!",
        "Claim debunked by NASA & Paris officials.",
        "https://www.snopes.com/fact-check/aliens-eiffel-tower"
    ),
    (
        "Chocolate cures all diseases!",
        "Satire post spread on social media.",
        "https://www.bbc.com/future/article/20171128-the-fake-news-machine"
    ),
    (
        "Government announces free flying cars by 2025!",
        "Edited meme circulated as real news.",
        "https://www.factcheck.org/"
    ),
]

# Loop with corrected variables
for title, desc, url in fake_news_links:
    container_2.markdown(
        f"""
        <a class="fake-item" href="{url}" target="_blank">
            <div>
                <div class="fake-title">{title}</div>
                <div class="fake-desc">{desc}</div>
            </div>
            <div class="fake-link">Open</div>
        </a>
        """,
        unsafe_allow_html=True,
    )

# =============== <footer> ===============
# 🌟 NEW FIX: Footer text color changed to white with shadow 🌟
st.markdown("<br><div style='text-align:center;color:#FFFFFF;font-size:13px;text-shadow: 0 0 5px #000;'>Made with ❤ using Streamlit</div>", unsafe_allow_html=True)