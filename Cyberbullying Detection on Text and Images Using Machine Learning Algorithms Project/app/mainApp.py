# import streamlit as st
# import pickle
# import pytesseract
# from PIL import Image
# import re

# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# model = pickle.load(open('../models/best_model.pkl', 'rb'))

# vectorizer = pickle.load(open('../models/tfidf_vectorizer.pkl', 'rb'))

# def preprocess_text(text):
#     if not text:
#         return None 
#     text = text.lower()  
#     text = re.sub(r'http\S+', '', text)  
#     text = re.sub(r'<.*?>', '', text)  
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  
#     text = re.sub(r'^RT[\s]+', '', text) 
#     return text.strip() 

# st.title('CYBERBULLYING/HATE SPEECH PREDICTION')

# tweet_input = st.text_input('Enter your tweet')

# image = st.file_uploader('Upload an image', type=['jpg', 'png'])

# submit = st.button('Predict')

# if submit:
#     if tweet_input:
#         preprocessed_text = preprocess_text(tweet_input)
#         vectorized_text = vectorizer.transform([preprocessed_text])
        
#         prediction = model.predict(vectorized_text)
#         st.write('Prediction for text input:', prediction[0])
#         if prediction[0] == 'Negative':
#             st.write('Your text contains cyberbullying keywords!')
#         elif prediction[0] == 'Positive':
#             st.write('Your text is positive and free from cyberbullying content.')
#         elif prediction[0] == 'Neutral':
#             st.write('Your text is neutral and free from cyberbullying content.')
#         else:
#             st.write('Your text is irrelevant regarding cyberbullying detection.')

#     if image:
        
#         st.image(image, caption='Uploaded Image', use_column_width=True)

#         extracted_text = pytesseract.image_to_string(Image.open(image))
        
#         st.write('Extracted text from the image:', extracted_text)

#         preprocessed_text = preprocess_text(extracted_text)
        
#         if preprocessed_text:
#             vectorized_text = vectorizer.transform([preprocessed_text])
            
#             image_prediction = model.predict(vectorized_text)
#             st.write('Prediction for image text:', image_prediction[0])
#             if image_prediction[0] == 'Negative':
#                 st.write('The text extracted from the image contains cyberbullying keywords!')
#             elif image_prediction[0] == 'Positive':
#                 st.write('The text extracted from the image is positive and free from cyberbullying content.')
#             elif image_prediction[0] == 'Neutral':
#                 st.write('The text extracted from the image is neutral and free from cyberbullying content.')
#             else:
#                 st.write('The text extracted from the image is irrelevant regarding cyberbullying detection.')
#         else:
#             st.write('No text extracted from the image. Please try again.')

import streamlit as st
import pickle
import pytesseract
from PIL import Image
import re

# ------------------------------
# Configuration
# ------------------------------
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Page settings
st.set_page_config(
    page_title="Cyberbullying & Hate Speech Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Load Model & Vectorizer
# ------------------------------
@st.cache_resource
def load_resources():
    model = pickle.load(open('../models/best_model.pkl', 'rb'))
    vectorizer = pickle.load(open('../models/tfidf_vectorizer.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_resources()

# ------------------------------
# Preprocessing Function
# ------------------------------
def preprocess_text(text):
    if not text:
        return None
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    return text.strip()

# ------------------------------
# UI Layout
# ------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#4B9CD3;'>🧠 Cyberbullying & Hate Speech Detection</h1>
    <p style='text-align:center; font-size:18px; color:gray;'>
        Detect cyberbullying or hate speech from text or images using ML models.
    </p>
    <hr style="border: 1px solid #ddd;">
    """,
    unsafe_allow_html=True
)

option = st.radio("Choose Input Type:", ["📝 Text", "🖼️ Image"], horizontal=True)

# ------------------------------
# Text Prediction
# ------------------------------
if option == "📝 Text":
    tweet_input = st.text_area("✏️ Enter your text or tweet below:", height=150)
    if st.button("🔍 Analyze Text"):
        if tweet_input:
            with st.spinner("Analyzing text... ⏳"):
                preprocessed = preprocess_text(tweet_input)
                vect_text = vectorizer.transform([preprocessed])
                prediction = model.predict(vect_text)[0]

            st.subheader("🔎 Result")
            if prediction.lower() == "negative":
                st.error("🚨 This text contains cyberbullying or hate speech.")
            elif prediction.lower() == "positive":
                st.success("✅ This text is positive and free from bullying content.")
            elif prediction.lower() == "neutral":
                st.info("😐 This text is neutral and non-offensive.")
            else:
                st.warning("🤔 The text seems irrelevant for cyberbullying detection.")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

# ------------------------------
# Image Prediction
# ------------------------------
elif option == "🖼️ Image":
    st.write("Upload an image containing text (e.g., a screenshot of a tweet).")
    uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("🔍 Analyze Image"):
            with st.spinner("Extracting and analyzing text from image... ⏳"):
                extracted_text = pytesseract.image_to_string(Image.open(uploaded_image))
                preprocessed = preprocess_text(extracted_text)
                st.write("📜 **Extracted Text:**", extracted_text)

                if preprocessed:
                    vect_text = vectorizer.transform([preprocessed])
                    prediction = model.predict(vect_text)[0]

                    st.subheader("🔎 Result")
                    if prediction.lower() == "negative":
                        st.error("🚨 The image text contains cyberbullying or hate speech.")
                    elif prediction.lower() == "positive":
                        st.success("✅ The image text is positive and safe.")
                    elif prediction.lower() == "neutral":
                        st.info("😐 The image text is neutral and harmless.")
                    else:
                        st.warning("🤔 The text seems irrelevant for cyberbullying detection.")
                else:
                    st.warning("⚠️ No readable text found in the image.")
    else:
        st.info("📂 Please upload an image to begin analysis.")

# ------------------------------
# Sidebar Info
# ------------------------------
st.sidebar.markdown("### ⚙️ About this Project")
st.sidebar.info(
    """
    This app uses **Machine Learning** and **OCR** (Optical Character Recognition)  
    to detect potential **cyberbullying or hate speech** in both text and images.
    
    - 🧠 Model: Trained on Twitter dataset  
    - 🗣️ Technique: TF-IDF + Classification  
    - 🖼️ OCR Engine: Tesseract
    
    👨‍💻 Developed by: *Manohar Singh*
    """
)

st.sidebar.markdown("---")
st.sidebar.write("🌐 **Cyberbullying Detection App v1.0**")
