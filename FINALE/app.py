import pandas as pd
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import gradio as gr

# 🔹 Pfade
CHROMA_DIR = r"C:\Unkram123\Full_stack_projects\erkam_netflix\FINALE\chroma_movies_bge"  # Lokaler Chroma-Ordner
CSV_PATH = r"C:\Unkram123\Full_stack_projects\erkam_netflix\FINALE\FINALE_NEW_WITH_IMAGES.csv"  # CSV mit Poster URLs
PLACEHOLDER_IMAGE = r"C:\Unkram123\Full_stack_projects\erkam_netflix\FINALE\placeholder.png"  # Falls Poster fehlt

# 🔹 CSV laden
movies = pd.read_csv(CSV_PATH, encoding="utf-8")
movies["doc_id"] = movies.index

# 🔹 Embeddings laden
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True}
)

# 🔹 Chroma-Index laden
db_movies = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# 🔹 Cross-Encoder laden
reranker = CrossEncoder("cross-encoder/stsb-roberta-large")

# 🔹 Funktion für semantische Empfehlungen
def retrieve_semantic_recommendations(query: str, top_k: int = 10):
    recs = db_movies.similarity_search(query, k=50)
    pairs = [(query, rec.page_content) for rec in recs]
    scores = reranker.predict(pairs)

    scores_dict = {}
    for rec, score in zip(recs, scores):
        doc_id = rec.metadata["doc_id"]
        if doc_id not in scores_dict or score > scores_dict[doc_id]["score"]:
            scores_dict[doc_id] = {"score": score, "rec": rec}

    unique_top_recs = sorted(scores_dict.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    gallery = []
    for item in unique_top_recs:
        rec = item["rec"]
        metadata = rec.metadata

        # Poster aus CSV anhand doc_id
        movie_row = movies[movies["doc_id"] == metadata["doc_id"]].iloc[0]
        cover_url = movie_row.get("poster_url", PLACEHOLDER_IMAGE)

        label = f"{movie_row['title']}\n\n{movie_row['description']}"
        gallery.append((cover_url, label))

    return gallery

# 🔹 Gradio-App
with gr.Blocks(theme=gr.themes.Glass()) as demo:
    gr.Markdown("# 🎬 Semantic Movie Recommender (Online Posters)")

    with gr.Row():
        user_query = gr.Textbox(label="Describe your ideal movie", placeholder="e.g., A sci-fi movie about time travel")
        submit_button = gr.Button("🔍 Find recommendations")

    gr.Markdown("## 🍿 Recommended Movies")
    output = gr.Gallery(label="Movies", columns=3, rows=3, show_label=True)

    submit_button.click(fn=retrieve_semantic_recommendations, inputs=user_query, outputs=output)

if __name__ == "__main__":
    demo.launch()
