import os
import re
import itertools
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
import torch
from faster_whisper import WhisperModel
from spellchecker import SpellChecker
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer, util
from pydub import AudioSegment

# --- KONSTANTA dari Model_STT.ipynb ---
# Didefinisikan di sini agar model dapat diinisialisasi
WHISPER_MODEL_NAME = "large-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
SR_RATE = 16000 # Sample Rate konsisten
SIMILARITY_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ML_TERMS = [
    "tensorflow", "keras", "vgc16", "vgc19", "mobilenet",
    "efficientnet", "cnn", "relu", "dropout", "model",
    "layer normalization", "batch normalization", "attention",
    "embedding", "deep learning", "dataset", "submission"
]
PHRASE_MAP = {
    "celiac" : "cellular", "script" : "skripsi", "i mentioned" : "submission",
    "time short flow": "tensorflow", "eras": "keras", "vic": "vgc16",
    "vic": "vgc19", "va": "vgc16", "va": "vgc19", "mobile net": "mobilenet",
    "data set" : "dataset", "violation laws" : "validation loss"
}
FILLERS = ["umm", "uh", "uhh", "erm", "hmm", "eee", "emmm", "yeah", "ah", "okay", "vic"]

# --- MODEL CACHING --- (Akan dipanggil di app.py menggunakan @st.cache_resource)
def load_stt_model():
    """Memuat Faster Whisper model."""
    print(f"Loading WhisperModel ({WHISPER_MODEL_NAME}) on {DEVICE.upper()}")
    return WhisperModel(WH4ISPER_MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

def load_text_models():
    """Memuat SpellChecker dan SentenceTransformer."""
    spell = SpellChecker(language="en")
    embedder = SentenceTransformer(SIMILARITY_MODEL_NAME)
    english_words = set(spell.word_frequency.words())
    return spell, embedder, english_words

# --- AUDIO UTILITIES (DARI NOTEBOOK 1) ---

def video_to_wav(input_video_path, output_wav_path, sr=SR_RATE):
    """Mengkonversi video ke WAV mono pada 16kHz."""
    try:
        audio = AudioSegment.from_file(input_video_path)
        audio = audio.set_channels(1).set_frame_rate(sr)
        audio.export(output_wav_path, format="wav")
        return True
    except Exception as e:
        print(f"Video to WAV conversion failed: {e}")
        return False

def noise_reduction(in_wav, out_wav, prop_decrease=0.6):
    """Menerapkan Noise Reduction menggunakan noisereduce."""
    try:
        y, sr = librosa.load(in_wav, sr=SR_RATE)
        y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease)
        sf.write(out_wav, y_clean, sr)
        return True
    except Exception as e:
        print(f"Noise reduction failed: {e}")
        return False

# --- TEXT CLEANING LOGIC (DARI NOTEBOOK 1) ---

def correct_ml_terms(word, spell, english_words):
    """Koreksi domain-specific terms (ML_TERMS) dari Notebook."""
    w = word.lower()
    if w in english_words:
        return word

    match, score, _ = process.extractOne(w, ML_TERMS)
    dist = Levenshtein.distance(w, match.lower())

    if dist <= 3 or score >= 65:
        return match
    return word

def fix_context_outliers(text, model_embedder):
    """Koreksi kata yang tidak sesuai konteks menggunakan embedding (Experimental)."""
    words = text.split()
    if len(words) < 3:
        return text

    try:
        word_embeds = model_embedder.encode(words)
        sent_embed = model_embedder.encode([text])[0]
        sims = util.cos_sim(word_embeds, [sent_embed]).flatten().numpy()
        outlier_idx = sims.argmin()

        match, score, _ = process.extractOne(words[outlier_idx], words)
        if score < 95:
            words[outlier_idx] = match
    except:
        pass

    return " ".join(words)

def remove_duplicate_words(text):
    """Menghilangkan kata duplikat yang berurutan."""
    return " ".join([k for k, g in itertools.groupby(text.split())])

def clean_text(text, spell, model_embedder, english_words, use_embedding_fix=True):
    """Fungsi utama cleaning teks (Final Version dari Notebook 1)."""
    
    # A. hapus filler words
    pattern = r"\b(" + "|".join(FILLERS) + r")\b"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # B. hapus titik & tanda baca yang berlebihan
    text = re.sub(r"\.{2,}", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    # C. rapikan spasi
    text = re.sub(r"\s+", " ", text).strip()

    # D. koreksi frasa
    for wrong, correct in PHRASE_MAP.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", correct, text)

    # E. koreksi word level
    words = []
    for w in text.split():
        # 1. typo correction
        sp = spell.correction(w)
        if sp:
            w = sp
        # 2. ML domain correction
        w = correct_ml_terms(w, spell, english_words)
        words.append(w)

    text = " ".join(words)

    # F. koreksi outlier embedding (jika diaktifkan)
    if use_embedding_fix:
        text = fix_context_outliers(text, model_embedder)

    # G. hilangkan kata duplikat berurutan
    text = remove_duplicate_words(text)

    return text

# --- FUNGSI UTAMA TRANSKRIPSI ---
def transcribe_and_clean(audio_path, whisper_model, spell_checker, embedder, english_words):
    """Melakukan transkripsi, lalu membersihkan teks."""
    try:
        segments, _ = whisper_model.transcribe(
            audio_path, language="en", task="transcribe", beam_size=4, vad_filter=True
        )
        raw_text = " ".join([seg.text for seg in segments])
        
        # Menerapkan seluruh rantai cleaning
        cleaned_text = clean_text(raw_text, spell_checker, embedder, english_words, use_embedding_fix=True)
        return cleaned_text
    except Exception as e:
        print(f"Transcription/Cleaning error: {e}")
        return "ERROR: Failed to transcribe or clean audio."