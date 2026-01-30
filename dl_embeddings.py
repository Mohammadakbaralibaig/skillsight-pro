# services/dl_embeddings.py

from sentence_transformers import SentenceTransformer, util
import numpy as np

_MODEL = None

def load_embedding_model(name: str = "all-MiniLM-L6-v2"):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(name)
    return _MODEL

def encode_text(texts, model_name: str = "all-MiniLM-L6-v2"):
    """
    Encodes text into DL embeddings.
    """
    model = load_embedding_model(model_name)

    if isinstance(texts, str):
        texts = [texts]

    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def cosine_similarity_score(a_vec, b_vec):
    """
    Cosine similarity between 2 vectors.
    """
    return float(util.cos_sim(a_vec, b_vec).item())

def resume_job_match_score(resume_text, job_text, model_name="all-MiniLM-L6-v2"):
    """
    Returns a score 0–100 representing semantic similarity.
    """
    if not resume_text or not job_text:
        return 0

    vecs = encode_text([resume_text, job_text], model_name=model_name)
    sim = cosine_similarity_score(vecs[0], vecs[1])

    score = max(0.0, min(1.0, sim)) * 100
    return int(round(score))
