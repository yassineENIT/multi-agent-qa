import openai
import spacy
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

# 1. Configurer votre clé API OpenAI
#openai.api_key = "your-api-key-here"
# 2. Charge les modèles nécessaires
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

# 3. Chaînes de raisonnement simulées (ou récupérées via GPT)
reasoning_chain = [
    "The monarchy imposed heavy taxes on the third estate.",
    "There was a severe famine that led to food shortages.",
    "Enlightenment philosophers inspired the revolution.",
    "Nobles and clergy enjoyed privileges denied to commoners.",
    "The people demanded equality and justice.",
    "The absolute monarchy ignored the needs of the people."
]

# 4. Extraction d'entités et construction du graphe
graph = nx.Graph()
for sentence in reasoning_chain:
    doc = nlp(sentence)
    entities = [ent.text for ent in doc.ents]
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            graph.add_edge(entities[i], entities[j], sentence=sentence)

# 5. Clustering des phrases pour créer les groupes de la Mind Map
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reasoning_chain)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
labels = kmeans.labels_

# 6. Regroupement et résumé de chaque cluster avec OpenAI
mind_map = {}
for label in set(labels):
    cluster_sentences = [reasoning_chain[i] for i in range(len(reasoning_chain)) if labels[i] == label]
    joined_text = " ".join(cluster_sentences)

    # Appel à l'API OpenAI pour résumer ce cluster
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who summarizes ideas clearly."},
            {"role": "user", "content": f"Summarize the following ideas in 2-3 sentences:\n{joined_text}"}
        ]
    )
    summary = response['choices'][0]['message']['content']
    mind_map[label] = {
        "summary": summary,
        "sentences": cluster_sentences
    }

# 7. Affichage de la carte mentale
print("🧠 MIND MAP:")
for label, content in mind_map.items():
    print(f"\n🔹 Cluster {label}:")
    for s in content["sentences"]:
        print(f" - {s}")
    print(f"📝 Résumé: {content['summary']}")

# 8. Recherche dans la Mind Map (question ciblée)
def search_mind_map(query):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    best_score = 0
    best_summary = None

    for content in mind_map.values():
        summary_embedding = embedder.encode(content["summary"], convert_to_tensor=True)
        score = util.cos_sim(query_embedding, summary_embedding).item()
        if score > best_score:
            best_score = score
            best_summary = content["summary"]

    print(f"\n🔎 Résultat pour la question : '{query}'")
    print(f"✅ Meilleure réponse (score {best_score:.2f}): {best_summary}")

# Exemple d’interrogation
search_mind_map("What role did the Enlightenment play?")
