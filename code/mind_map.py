import fitz  # PyMuPDF
import openai
import spacy
import time
import networkx as nx
from sentence_transformers import SentenceTransformer, util

# Configuration des modèles et clés API
openai.api_key = "your-api-key-here"
embedder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

# 📄 Étape 1 : Extraire le texte d’un PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ✂️ Étape 2 : Chunker le texte en petits morceaux (évite dépassement de tokens)
def chunk_text(text, max_chars=800):  # max_chars équilibré pour éviter les erreurs
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + "\n\n"
        else:
            chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks

# 🤖 Étape 3 : Résumer chaque chunk avec GPT-3.5
def summarize_chunk(chunk):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You summarize academic content clearly in 2-3 bullet points."},
                {"role": "user", "content": f"Summarize the following in 2-3 key ideas:\n{chunk}"}
            ],
            max_tokens=200,
            temperature=0.3
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"❌ Erreur lors du résumé : {e}")
        return "[Résumé indisponible pour ce chunk]"

# 🧠 Étape 4 : Extraire les entités et les relations
def extract_entities_and_edges(summary, graph):
    doc = nlp(summary)
    entities = [ent.text for ent in doc.ents]
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            graph.add_edge(entities[i], entities[j], label="related")

# 🔍 Étape 5 : Recherche par similarité dans les résumés
def search_mind_map(summaries, query):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    best_score = 0
    best_summary = None
    for summary in summaries:
        summary_embedding = embedder.encode(summary, convert_to_tensor=True)
        score = util.cos_sim(query_embedding, summary_embedding).item()
        if score > best_score:
            best_score = score
            best_summary = summary
    print(f"\n🔍 Question : '{query}'")
    print(f"✅ Résumé pertinent (score {best_score:.2f}):\n{best_summary}")

# 🚀 Programme principal
def main(pdf_path):
    print("📄 Lecture du fichier PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)

    print("✂️ Découpage du texte...")
    chunks = chunk_text(pdf_text, max_chars=800)  # Ajusté pour GPT-3.5

    mind_map_graph = nx.Graph()
    summaries = []

    print("🤖 Résumé des chunks avec GPT-3.5...")
    for i, chunk in enumerate(chunks[:30]):  # Facultatif : limite les chunks analysés
        print(f"🔹 Chunk {i+1}/{len(chunks)}...")
        summary = summarize_chunk(chunk)
        summaries.append(summary)
        extract_entities_and_edges(summary, mind_map_graph)
        time.sleep(1.2)  # Évite de dépasser les limites d’API

    print("\n🧠 Résumés extraits :")
    for idx, s in enumerate(summaries):
        print(f"\n🔹 Résumé {idx+1}:\n{s}")

    # 🔍 Recherche dans les résumés
    search_mind_map(summaries, "How does Weibull distribution apply to reliability engineering?")

# 🔧 Exécute le programme
if __name__ == "__main__":
    main("file.pdf")
