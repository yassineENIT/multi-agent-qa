import streamlit as st
import PyPDF2
import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from PIL import Image
client = OpenAI(api_key="your-api-key-here"
) 

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
st.title("📚 MCQ Generator from PDF")

uploaded_files = st.file_uploader("📤 Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("📄 Uploaded files:")
    full_text = ""

    for uploaded_file in uploaded_files:
        st.write(f"✅ {uploaded_file.name}")
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

    with st.expander("📖 Preview of extracted text"):
        st.text(full_text[:2000] + "..." if len(full_text) > 2000 else full_text)

    sentences = [sent.text.strip() for sent in nlp(full_text).sents if len(sent.text.strip()) > 20]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    num_clusters = min(4, len(sentences))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    labels = kmeans.labels_

    mind_map = {}
    for label in set(labels):
        cluster_sentences = [sentences[i] for i in range(len(sentences)) if labels[i] == label]
        joined_text = " ".join(cluster_sentences[:5])

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Tu es un assistant pédagogique."},
                {"role": "user", "content": f"Résume ce contenu en 2-3 phrases :\n{joined_text}"}
            ]
        )
        summary = response.choices[0].message.content
        mind_map[label] = {
            "summary": summary,
            "sentences": cluster_sentences
        }

    qcm_data = []
    for label, content in mind_map.items():
        summary = content["summary"]
        prompt = f"""
Read the following summary and generate one multiple-choice question (MCQ) in english.
Summary: {summary}

Format the response like this:
Question: ...
A. ...
B. ...
C. ...
D. ...
Correct answer: A/B/C/D
"""

        gpt_qcm = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content

        try:
            lines = gpt_qcm.split("\n")
            question = lines[0].replace("Question:", "").strip()
            options = [l.strip() for l in lines[1:5]]
            correct = lines[5].split(":")[-1].strip()
            qcm_data.append({
                "question": question,
                "option_A": options[0],
                "option_B": options[1],
                "option_C": options[2],
                "option_D": options[3],
                "correct_answer": correct
            })
        except Exception as e:
            st.warning(f"Erreur parsing QCM: {e}")

    if qcm_data:
        df = pd.DataFrame(qcm_data)
        st.success("✅ MCQs successfully generated!")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download MCQs in CSV format", data=csv, file_name="qcm_questions.csv", mime='text/csv')
    else:
        st.warning("No MCQs generated.")
else:
    st.info("🧾 No PDF file uploaded.")
