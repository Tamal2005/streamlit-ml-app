import streamlit as st 
from streamlit_option_menu import option_menu
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from PIL import Image
from io import BytesIO
import librosa
from IPython.display import Audio, display 
import requests
import json
import sys

sys.stdout.reconfigure(line_buffering=True)

def load_css():
    st.markdown(
        """
        <style>
            /* Disable sidebar resizing by hiding the resize handle */
            [data-testid="stSidebar"] div[style*="cursor: col-resize"] {
                display: none !important;
            }
            
            /* Alternative: Make the resize handle non-functional */
            [data-testid="stSidebar"] div[style*="cursor: col-resize"] {
                pointer-events: none !important;
                cursor: default !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def load_model_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(BytesIO(response.content))
    else:
        raise Exception(f"Failed to download model from Google Drive. Status code: {response.status_code}")


# Preprocessing
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Fake News Prediction endpoint

def predict_news(news):
    # models for fake news prediction
    lr_file_id = '1J9rL9Y0l2LHlCrODI8whsWCut-9rNTtI'
    lr = load_model_from_drive(lr_file_id)

    vectorizer_file_id = '1nK5vdINYUVjsfZCu-Yp1j-KGXcHG1ZW5'
    vectorizer = load_model_from_drive(vectorizer_file_id)

    try:
        data = news
        cleaned_text = stemming(data.get("title", ""))
        
        vector_input = vectorizer.transform([cleaned_text])
        
        prediction = lr.predict(vector_input)

        result = "Fake News" if prediction[0] == 1 else "Real News"
        return result
    except Exception as e:
        return str(e)



# Spam Mails Prediction endpoint    

def predict_mails(mail):
    loaded_package = joblib.load('./model/spamMailModel/all_spam_mail_model.pkl')
    spamModel = loaded_package['model']
    spamvectorizer = loaded_package['vectorizer']

    try:
        data = mail
        print(data)
        input_mail = data.get("mail", "")

        result = 'Spam Mail' if spamModel.predict(spamvectorizer.transform([input_mail])) == 0 else "Not Spam"
        return result
    except Exception as e:
        return str(e)




# Face Mask Prediction endpoint    
def predict_facemask(image):
    masksModel = joblib.load('./model/faceMaskModel/face_masks_model_compressed.pkl')
    try:
        # Read image bytes
        image_bytes = image.read()

        # Convert to image array using PIL + OpenCV
        img = Image.open(BytesIO(image_bytes)).convert("")
        img = img.resize((128, 128))
        img_np = np.array(img)
        img_np = img_np / 255.0
        img_np = np.reshape(img_np, (1, 128, 128, 3))

        prediction = masksModel.predict(img_np)

        prediction_label = np.argmax(prediction)

        result = "The person in the image is not wearing a mask" if prediction_label == 0 else "The person in the image is wearing a mask"
        return result
    except Exception as e:
        return str(e)
    



# Phishing Url Prediction endpoint   
def predict_Urls(urls):
    phishing_model_file_id = '1ikdw85kNEmIDW4zaPUcWEXpkSvHNOuZQ'
    phishing_model = load_model_from_drive(phishing_model_file_id)
    url_vectorizer = joblib.load('./model/phishingUrlModel/url_Vectorizer.pkl')
    try:
        data = urls
        input_url = data.get("link", "").strip()
        
        feature_extraction = url_vectorizer.transform([input_url])
        
        prediction = phishing_model.predict(feature_extraction)
        

        result = "Safe URL" if prediction[0] == 0 else "Phishing URL"
        return result
    except Exception as e:
        return str(e)


# Deepfake voice Prediction endpoint   
def predict_voice(voice):
    deepfake_voice_model = joblib.load('./model/deepfakeVoiceModel/deepfake_voice_detection_model.pkl')
    try:
        data = voice
        audio, sr = librosa.load(data)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        # Combine features
        features = np.concatenate([mfccs.mean(axis=1), chroma.mean(axis=1), spectral_contrast.mean(axis=1)])
        # Scale the features (reshape for scaler)
        features_reshaped = features.reshape(1, -1)  # Reshape to (1, n_features)

        # Make prediction
        input_prediction = deepfake_voice_model.predict(features_reshaped)
        print(f"Raw prediction: {input_prediction}")

        input_pred_label = np.argmax(input_prediction)

        # Display result
        result = "The voice is fake (Edited or AI generated!)" if input_pred_label == 1 else "The voice is real"
        return result

    except Exception as e:
        return str(e)

# Apply CSS
load_css()

# --- Sidebar Menu ---
with st.sidebar:
    page = option_menu(
        None,  # No title
        ["Home", "ML Projects","Overview", "About"],  # Menu options
        icons=["house", "cpu","grid-fill", "info-circle"],  # Optional: icons
        default_index=0,
        orientation="vertical"
    )
# Show sub-menu only if ML Projects is selected
    sub_option = None
    if page == "ML Projects":
        sub_option = option_menu(
            "ML Projects",  # sub-menu title
            ["Fake News Detection", "Spam Mail Detection", "Phishing URL Detection", "Face Mask Detection", "Deepfake Voice Detection"],
            icons=["newspaper", "envelope", "link-45deg", "person-bounding-box", "mic"],
            default_index=0,
            orientation="vertical"
        )


# -----------------------------
# Home Page
# -----------------------------
if page == "Home":
    st.title('🤖 MAchInE LEaRNinG APp')

    st.info('This is a Machine and Deep Learning Model App')

    st.markdown("""
    Welcome to the **AI Detection Hub**! 🚀  
    This app contains 4 different AI-based models:
    
    1. **Fake News Detection** 📰  
       Detects whether a given news text is *fake* or *true*.  

    2. **Spam Mail Detection** 📧  
       Detects whether an email is *spam* or *not spam*.  

    3. **Phishing URL Detection** 🌐  
       Identifies whether a URL is *phishing* or *safe*.  

    4. **Face Mask Detection** 😷  
       Detects whether a person in an uploaded image is wearing a mask.  
    
    5. **Deepfake Voice Detection** 🎙️  
       Detects whether a voice is *real* or *fake* *(edited or AI generated)*.            

    ---
    🔨🤖🔧 *Note: These models are trained on sample datasets and may not always provide 100% accurate results.*  
    """)

elif page == "ML Projects":

    # -----------------------------
    # Fake News Detection
    # -----------------------------
    if sub_option == "Fake News Detection":
        st.header("Fake News Detection 📰")

        text_input = st.text_area("Enter news text:")
        if st.button("Predict"):
            if text_input.strip() == "":
                st.warning("Please enter some text to predict.")
            else:
                # Convert string to dictionary format expected by your function
                text_data = {"news": text_input}
                result = predict_news(text_data)  # Pass dictionary
        
                if result.startswith("Error") or "Traceback" in result:
                    st.error(result)
                else:
                    if result == "Fake News":
                        st.error("❌ Fake News")
                    else:
                        st.success("✅ Real News")
                    st.caption("⚠️ It is trained on a dataset which may not be latest so output may be wrong.")


    # -----------------------------
    # Spam Mail Detection
    # -----------------------------
    elif sub_option == "Spam Mail Detection":
        st.header("Spam Mail Detection 📧")

        email_input = st.text_area("Enter email content:")
        if st.button("Predict"):
            if email_input.strip() == "":
                st.warning("Please enter some text to predict.")
            else:
                # Convert string to dictionary format expected by your function
                email_data = {"mail": email_input}
                result = predict_mails(email_data)  # Pass dictionary
        
                if result.startswith("Error") or "Traceback" in result:
                    st.error(result)
                else:
                    if result == "Spam Mail":
                        st.error("❌ Spam Mail")
                    else:
                        st.success("✅ Valid Mail")
                    st.caption("⚠️ It is trained on a dataset which may not be latest so output may be wrong.")


    # -----------------------------
    # Phishing URL Detection
    # -----------------------------
    elif sub_option == "Phishing URL Detection":
        st.header("Phishing URL Detection 🌐")

        url_input = st.text_input("Enter URL:")
        if st.button("Predict"):
            if url_input.strip() == "":
                st.warning("Please enter a URL to predict.")
            else:
                # Convert string to dictionary format expected by your function
                url_data = {"link": url_input}
                result = predict_Urls(url_data)  # Pass dictionary
        
                if result.startswith("Error") or "Traceback" in result:
                    st.error(result)
                else:
                    if result == "Phishing URL":
                        st.error("❌ Phishing URL Detected (Danger!)")
                    else:
                        st.success("✅ Safe URL")
                    st.caption("⚠️ It is trained on a dataset which may not be latest so output may be wrong.")


    # -----------------------------
    # Face Mask Detection
    # -----------------------------
    elif sub_option == "Face Mask Detection":
        st.header("Face Mask Detection 😷")

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = uploaded_file
            
            col1, col2, col3 = st.columns([1,2,1])  # middle column is wider
            with col2:
                st.image(image, caption="Uploaded Image", width=150)
            
            
            if st.button("Predict"):
                result = predict_facemask(image)
                if result == "The person in the image is not wearing a mask":
                    st.error("❌ The person in the image is not wearing a mask")
                else:
                    st.success("✅ The person in the image is wearing a mask")
                    st.caption("⚠️ It is trained on a dataset which may not be latest so output may be wrong.")

    # -----------------------------
    # Deepfake Voice Detection
    # -----------------------------
    elif sub_option == "Deepfake Voice Detection":
        st.header("Deepfake Voice Detection 🎙️")

        uploaded_file = st.file_uploader("Upload an Voice", type=["wav"])
        if uploaded_file is not None:
            voice = uploaded_file
            
            col1, col2, col3 = st.columns([1,2,1])  # middle column is wider
            with col2:
                st.audio(voice, format="audio/wav")
                st.caption("🎧 Uploaded Voice")
            
            
            if st.button("Predict"):
                result = predict_voice(voice)
                if result == "The voice is fake (Edited or AI generated!)":
                    st.error("❌ The voice is fake (Edited or AI generated!)")
                else:
                    st.success("✅ The voice is real")
                    st.caption("⚠️ It is trained on a dataset which may not be latest so output may be wrong.")

elif page == "Overview":

    st.title("🏠 Overview")

    st.markdown("""
    Welcome to the **AI Detection Hub** 🤖✨  
    
    This app showcases multiple **Machine Learning** and **Deep Learning** models trained to detect and classify different types of data.  

    ---

    ### 🔍 What You Can Do Here
    1. 📰 **Fake News Detection** – Paste a news headline or article to check if it’s real or fake.  
    2. 📧 **Spam Mail Detection** – Test an email text and see if it’s spam or not.  
    3. 🌐 **Phishing URL Detection** – Enter a link to find out if it’s safe or a phishing attempt.  
    4. 😷 **Face Mask Detection** – Upload an image to check if a person is wearing a mask.
    5. 🎙️ **Deepfake Voice Detection** – upload a voive to check if it's real or fake.  

    ---

    ### 📊 Why This Matters
    - Fake news spreads misinformation and impacts society.  
    - Spam mails can carry scams or malware.  
    - Phishing websites steal sensitive information.  
    - Face masks are vital for health safety in many scenarios.
    - Deepfake voices can be misused for fraud, impersonation, and spreading disinformation.
     

    This app helps demonstrate how **AI can tackle real-world problems** 🌍.  

    ---  
    """)

    st.subheader("📑  Model Training Code")

    def notebook(path):
        # Load Jupyter notebook
        with open(path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        # Extract code cells
        code_cells = [
            cell["source"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]

        # Join code into one string
        full_code = "\n".join("".join(cell) for cell in code_cells)

        return full_code

    # Show inside expander
    with st.expander("📰 Fake News Detection Training Code"):
        st.code(notebook('./notebooks/fake_news_prediction.ipynb'), language="python")

    with st.expander("📰 Spam Mail Detection Training Code"):
        st.code(notebook('./notebooks/spam_mail_prediction.ipynb'), language="python")

    with st.expander("📰 Phishing URl Detection Training Code"):
        st.code(notebook('./notebooks/phishing_url_prediction.ipynb'), language="python")

    with st.expander("📰 Face Mask Detection Training Code"):
        st.code(notebook('./notebooks/face_mask_detection.ipynb'), language="python")

    with st.expander("📰 Deepfake Voice Detection Training Code"):
        st.code(notebook('./notebooks/deepfake_voice_detection.ipynb'), language="python")

    st.markdown(
        '''
    ---

    ### 🧭 How to Navigate
    Use the **sidebar** to switch between:  
    - 📂 **ML Projects** → Try out each model interactively.  
    - ℹ️ **About** → Learn more about the technologies behind the app.  

    ---

    🔨🤖🔧 *Note: These models are prototypes for educational purposes and not production-level security tools.*'''
    )

elif page == "About":
    st.title("ℹ️ About")

    st.markdown("""
    Welcome to the **AI Detection Hub** 🚀  

    This app brings together multiple **Machine Learning and Deep Learning models** to demonstrate how AI can be applied to real-world problems:  

    - 📰 **Fake News Detection** – Classifies whether a news article is fake or real.  
    - 📧 **Spam Mail Detection** – Identifies spam vs. non-spam messages.  
    - 🌐 **Phishing URL Detection** – Detects unsafe links.  
    - 😷 **Face Mask Detection** – Uses computer vision to check if a person is wearing a mask.
    - 🎙️ **Deepfake Voice Detection** – Analyzes audio to detect AI-generated or manipulated voices.

    ---

    ### 🔧 Technologies Used
    - **Python** (Streamlit, scikit-learn, TensorFlow/Keras, NLTK, etc.)  
    - **Machine Learning Models**: Logistic Regression, Naive Bayes, etc.  
    - **Deep Learning Models**:  
      - CNNs for image classification (Face Mask Detection)  
      - Conv1D networks for audio classification (Deepfake Voice Detection)  
    - **Vectorizers & Feature Extraction**:  
      - TF-IDF for text (Fake News, Spam Mail)  
      - Custom URL feature extraction (Phishing Detection)  
      - MFCCs (Mel-Frequency Cepstral Coefficients) for voice analysis  
    - **Deployment**: Streamlit app with Google Drive model storage  
  

    ---

    ### 📌 Disclaimer
    These models are trained on **public datasets** for demonstration and educational purposes.  
    They may not always provide 100% accurate predictions.  

    ---

    ### 👨‍💻 Author
    Developed by **Tamal Debnath**  
    📧 Email: tamalcoder@email.com  
    🌐 GitHub: [github.com/tamal2005](https://github.com/tamal2005)  
    💼 LinkedIn: [linkedin.com/in/tamal-debnath-35823b312](https://linkedin.com/in/tamal-debnath-35823b312)  

    ---
    """)