import librosa
import numpy as np
import os

# --- THRESHOLDS dari Analisis_Non_Verbal.ipynb ---
TEMPO_FAST = 150.0
TEMPO_SLOW = 125.0
PAUSE_TOO_MUCH_PERCENT = 45.0
PAUSE_MINIMAL_PERCENT = 35.0
SILENCE_THRESHOLD_RMS = 0.015 # Nilai RMS default dari Notebook

def interpret_tempo(bpm):
    """Interpretasi kualitatif tempo."""
    if bpm > TEMPO_FAST:
        return "too fast"
    elif bpm >= TEMPO_SLOW:
        return "fast"
    else:
        return "slow"

def interpret_pause_by_percent(pause_percent):
    """Interpretasi kualitatif jeda."""
    if pause_percent > PAUSE_TOO_MUCH_PERCENT:
        return "too many pauses"
    elif pause_percent <= PAUSE_MINIMAL_PERCENT:
        return "minimal pauses"
    else:
        return "normal pauses"

def analyze_non_verbal(file_path):
    """Menganalisis audio untuk Tempo dan Jeda (Sesuai Notebook 3)."""

    try:
        y, sr = librosa.load(file_path, sr=16000)
        total_duration = len(y) / sr

        # Analisis Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo[0]

        # Analisis Jeda (Menggunakan RMS)
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        frame_duration = hop_length / sr

        silent_frames_count = np.sum(rms < SILENCE_THRESHOLD_RMS)
        total_silent_time_sec = silent_frames_count * frame_duration

        pause_percentage = (total_silent_time_sec / total_duration) * 100 if total_duration > 0 else 0.0

        # Rangkuman Kualitatif
        tempo_qualitative = interpret_tempo(tempo)
        pause_qualitative = interpret_pause_by_percent(pause_percentage)
        summary = f"{tempo_qualitative} tempo and {pause_qualitative}"

        return {
            "tempo_bpm": f"{tempo:.2f}",
            "total_pause_seconds": f"{total_silent_time_sec:.2f}",
            "pause_percent": f"{pause_percentage:.2f}%",
            "qualitative_summary": summary
        }

    except Exception as e:
        return {
            "error": f"Failed to process non-verbal analysis: {str(e)}"
        }