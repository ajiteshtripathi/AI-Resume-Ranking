import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Enhanced CSS
st.markdown("""
    <style>
        body {
            background-color: #0D1117;
            color: #C9D1D9;
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
        }
        .stApp {
            background-color: #161B22;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.7);
            min-height: 100vh;
        }
        h1, h2 {
            color: #58A6FF;
            text-align: center;
        }
        .card {
            background-color: #21262D;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.7);
            margin-bottom: 20px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(88, 166, 255, 0.4);
        }
        .stTextArea, .stFileUploader {
            background-color: #21262D;
            color: #C9D1D9;
            border: 1px solid #58A6FF;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 20px;
        }
        .hover-button {
            background-color: #58A6FF;
            color: #FFFFFF;
            padding: 16px 32px;
            font-size: 18px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.2s ease;
            width: 100%;
            margin-top: 10px;
        }
        .hover-button:hover {
            background-color: #1F6FEB;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("AI Resume Ranking System")

# Layout
st.header("Upload Resumes")
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.header("Job Description")
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    job_description = st.text_area("Paste the job description here...")
    st.markdown('</div>', unsafe_allow_html=True)

# Extract Text
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = "".join([page.extract_text() or "" for page in pdf.pages])
    return text.strip()

# Rank Resumes
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_desc_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity([job_desc_vector], resume_vectors).flatten()

# AI Tips
def generate_resume_tips(score):
    if score > 80:
        return "Excellent match!"
    elif score > 60:
        return "Good match. Add more keywords."
    else:
        return "Needs improvement! Focus on key terms."

# Button to Rank Resumes
if st.button("Rank Resumes", key="rank_button", help="Click to rank resumes", use_container_width=True):
    if uploaded_files and job_description:
        st.header("Resume Rankings")

        resumes = [extract_text_from_pdf(file) for file in uploaded_files]

        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

        scores = rank_resumes(job_description, resumes)

        results_df = pd.DataFrame({
            "Resume": [file.name for file in uploaded_files],
            "Match Score (%)": (scores * 100).round(2),
            "AI Tips": [generate_resume_tips(score * 100) for score in scores]
        }).sort_values(by="Match Score (%)", ascending=False)

        # Display Results
        st.dataframe(results_df)

        # Success Modal
        st.markdown(f"""
            <div class="card">
                Top Match: <strong>{results_df.iloc[0]['Resume']}</strong> with {results_df.iloc[0]['Match Score (%)']}%!
            </div>
        """, unsafe_allow_html=True)

        # Download Results
        csv = results_df.to_csv(index=False)
        st.download_button("Download Results", data=csv, file_name="resume_rankings.csv", mime="text/csv")

    else:
        st.error("Please upload resumes and enter a job description.")
