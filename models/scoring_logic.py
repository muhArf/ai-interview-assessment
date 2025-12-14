import json
import pandas as pd
from sentence_transformers import util, SentenceTransformer
import numpy as np

# --- Thresholds dari Notebook 2 (diambil dari logika) ---
NON_RELEVANT_SIM_THRESHOLD = 0.2
MIN_LENGTH_FOR_SCORE = 5

# --- MODEL CACHING ---
def load_embedder_model():
    """Memuat model SentenceTransformer untuk scoring."""
    # Model yang sama yang digunakan di Notebook 1 dan 2
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 

# --- FUNGSI RELEVANSI (DARI NOTEBOOK 2) ---

def is_non_relevant(text: str) -> bool:
    """Mengecek apakah transkrip cenderung tidak relevan/kosong."""
    t = text.strip().lower()
    if len(t) == 0:
        return True

    if len(t) <= 10 and t.startswith(("em", "uh", "hm")):
        return True

    non_answers = [
        "i don't know", "i dont know", "no idea",
        "i have no idea", "not sure", "i can't answer",
        "i cannot answer", "i don't understand",
        "i dont understand", "how do you answer", "what should i do"
    ]
    if any(na in t for na in non_answers):
        return True

    generic = {
        "learning", "tensorflow", "cnn", "model", "machine", "ai", "ml",
        "nothing", "idk", "question"
    }
    if len(t.split()) <= 3 and any(w in generic for w in t.split()):
        return True

    return False

# --- FUNGSI CONFIDENCE (DARI NOTEBOOK 2) ---

def compute_confidence_score(answer: str, question: str, model_embedder: SentenceTransformer) -> int:
    """Menghitung Confidence Score berdasarkan kelancaran dan relevansi."""
    words = answer.lower().split()
    length = len(words)
    if length == 0:
        return 0

    # Cek relevansi jawaban ke pertanyaan
    sim_score = util.cos_sim(model_embedder.encode(answer.lower()), model_embedder.encode(question.lower())).item()
    if sim_score < NON_RELEVANT_SIM_THRESHOLD:
        return 0

    # Skor dasar dari kelancaran & variasi kata (Logika dari Notebook)
    fillers = {"emmm", "emm", "uh", "uhm", "hmm"}
    filler_ratio = sum(w in fillers for w in words) / max(length, 1)
    unique_ratio = len(set(words)) / max(length, 1)
    
    # Base scoring logic from Notebook 2
    if length < 5:
        base = 15
    elif length < 15:
        base = 35
    elif length < 40:
        base = 60
    else:
        base = 75
        
    base -= int(filler_ratio * 40)
    if unique_ratio < 0.4:
        base -= 15
    elif unique_ratio < 0.6:
        base -= 5

    # Gabungkan dengan relevansi
    confidence = int(base * sim_score)
    return max(0, min(100, confidence))

# --- FUNGSI SCORING RUBRIK (DARI NOTEBOOK 2) ---

def score_with_rubric(question_id, question_text, answer, rubric_data, model_embedder):
    """
    Menghitung skor berdasarkan perbandingan semantik dengan rubrik.
    rubric_data adalah dictionary yang berisi poin ideal: {'4': [...], '3': [...], ...}
    """
    q_key = f"q{question_id}"
    rubric = rubric_data[q_key]["ideal_points"]
    a = answer.strip()

    if is_non_relevant(a) or len(a.split()) < MIN_LENGTH_FOR_SCORE:
        # Mengembalikan skor 0 dan alasan dari rubrik level 0
        return 0, rubric.get(0, ["Unanswered"])[0] 

    embedding_a = model_embedder.encode(a.lower())

    # Fungsi untuk menghitung kecocokan
    def count_matches(indicators, threshold=0.40):
        hits = 0
        local_matches = []
        for ind in indicators:
            sim = util.cos_sim(embedding_a, model_embedder.encode(ind.lower())).item()
            if sim >= threshold:
                hits += 1
                local_matches.append(ind)
        return hits, local_matches

    # Iterasi dari skor tertinggi ke terendah (4, 3, 2, 1)
    # Gunakan string keys karena dimuat dari JSON
    for point_str in ["4", "3", "2", "1"]:
        point = int(point_str)
        indicators = rubric.get(point)
        
        if not indicators: continue

        hits, local_matches = count_matches(indicators)
        
        # Min hits logic dari Notebook 2
        if point == 4:
            min_hits = max(1, int(len(indicators) * 0.6))
        elif point == 3:
            min_hits = max(1, int(len(indicators) * 0.5))
        else: # Untuk point 2 dan 1
            min_hits = 1

        if hits >= min_hits:
            return point, "; ".join(local_matches) if local_matches else indicators[0]

    # Jika tidak mencapai level 1, skor tetap 1 (default minimal)
    return 1, rubric.get("1", ["Minimal or Vague Response"])[0]