import streamlit as st 
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime
from tensorflow.keras.models import load_model

# ====================
# Page Configuration
# ====================
st.set_page_config(
    page_title="🎯 Sentiment Analysis Dashboard",
    layout="wide",
    page_icon="💬"
)

# ====================
# Custom CSS Styling
# ====================
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        h1, h2, h3 {
            color: #3b3b3b;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stDownloadButton>button {
            background-color: #6c63ff;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ====================
# Load Model and Vectorizer
# ====================
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

model = load_model("best_model.h5")

# ====================
# Logging Function
# ====================
LOG_FILE = "predictions_log.csv"

def log_prediction(review, prediction, confidence, prediction_type):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "type": prediction_type,
        "review": review,
        "prediction": prediction,
        "confidence": confidence
    }])
    if os.path.exists(LOG_FILE):
        log_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(LOG_FILE, mode='w', header=True, index=False)

# ====================
# Prediction Functions
# ====================
def predict_sentiment(review):
    vec = vectorizer.transform([review]).toarray().astype('float32')
    prob = model.predict(vec)[0][0]
    label = "Positive" if prob > 0.5 else "Negative"
    log_prediction(review, label, float(prob), prediction_type="single")
    return label, float(prob)

def batch_predict(df, text_column):
    reviews = df[text_column].astype(str).tolist()
    vectors = vectorizer.transform(reviews).toarray().astype('float32')
    probs = model.predict(vectors).flatten()
    labels = ["Positive" if p > 0.5 else "Negative" for p in probs]
    df["Prediction"] = labels
    df["Confidence"] = probs
    for review, label, prob in zip(reviews, labels, probs):
        log_prediction(review, label, float(prob), prediction_type="batch")
    return df

# ====================
# App Title and Header
# ====================
st.title("💬 AI-Powered Sentiment Analysis")
st.markdown("Analyze customer reviews in real-time using Deep Learning 🧠")
st.markdown("---")

# ====================
# Single Review Analysis
# ====================
st.header("🔍 Single Review Analysis")
input_text = st.text_area("📝 Enter a review to analyze sentiment", height=150)

if st.button("✨ Predict Sentiment"):
    if input_text.strip():
        label, prob = predict_sentiment(input_text)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Sentiment", value=f"{label} {'😊' if label == 'Positive' else '😞'}")
        with col2:
            st.metric(label="Confidence", value=f"{prob:.2f}")
        
        if label == "Positive":
            st.balloons()
            st.success("This seems to be a positive review! 🎉")
        else:
            st.warning("Looks like a negative sentiment detected. ⚠️")

    else:
        st.error("❗ Please enter some review text.")

st.markdown("---")

# ====================
# Batch Prediction
st.header("📄 Batch Prediction from CSV")

uploaded_file = st.file_uploader("📤 Upload CSV with a 'review' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'review' not in df.columns:
        st.error("❌ CSV must contain a 'review' column.")
    else:
        result_df = batch_predict(df, 'review')
        
        st.subheader("📊 Predictions Table")
        st.dataframe(result_df[['review', 'Prediction', 'Confidence']])

        # Summary Statistics
        st.subheader("📈 Report Summary")
        sentiment_counts = result_df['Prediction'].value_counts()
        avg_confidence = result_df['Confidence'].mean()
        positive_conf = result_df[result_df['Prediction'] == 'Positive']['Confidence'].mean()
        negative_conf = result_df[result_df['Prediction'] == 'Negative']['Confidence'].mean()

        st.markdown(f"""
        - ✅ **Total Reviews:** {len(result_df)}
        - 😀 **Positive Reviews:** {sentiment_counts.get('Positive', 0)} (Avg Conf: {positive_conf:.2f})
        - 😐 **Negative Reviews:** {sentiment_counts.get('Negative', 0)} (Avg Conf: {negative_conf:.2f})
        - 📊 **Overall Avg Confidence:** {avg_confidence:.2f}
        """)

        # Bar chart
        st.subheader("📉 Sentiment Distribution - Bar Chart")
        st.bar_chart(sentiment_counts)

        # Pie chart using plotly
        import plotly.express as px
        pie_data = result_df['Prediction'].value_counts().reset_index()
        pie_data.columns = ['Sentiment', 'Count']
        fig = px.pie(pie_data, values='Count', names='Sentiment', title='Sentiment Breakdown')
        st.plotly_chart(fig, use_container_width=True)

        # Download CSV
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name='sentiment_predictions.csv',
            mime='text/csv',
        )
        st.success("✅ All batch predictions and logs saved successfully.")


# ====================
# Logs Viewer
# ====================
if os.path.exists(LOG_FILE):
    with st.expander("📁 View Recent Prediction Logs"):
        log_df = pd.read_csv(LOG_FILE)
        st.dataframe(log_df.tail(10))

# ====================
# Footer
# ====================
st.markdown("""
    <hr>
    <div style='text-align: center; padding-top: 30px; font-size: 16px; color: #6c63ff;'>
        Built with ❤️ using <strong>Streamlit</strong> | Model: <strong>LSTM Sentiment Classifier</strong><br>
        <a href='https://github.com/vaibhavgaikwad01' target='_blank' style='text-decoration: none; color: inherit;'>
            <img src='https://img.icons8.com/ios-glyphs/30/000000/github.png' alt='GitHub' style='vertical-align: middle; margin-top: 10px;'/> 
            <span style='font-size: 14px; margin-left: 5px;'>View on GitHub</span>
        </a>
    </div>
""", unsafe_allow_html=True)
