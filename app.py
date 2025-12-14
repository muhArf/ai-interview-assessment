import streamlit as st
import pandas as pd
import json
import os
import tempfile


# Import logic dari folder models
# PASTIKAN FILE-FILE INI ADA DI FOLDER 'models'
from models.stt_processor import load_stt_model, load_text_models, video_to_wav, noise_reduction, transcribe_and_clean
from models.scoring_logic import load_embedder_model, compute_confidence_score, score_with_rubric
from models.nonverbal_analysis import analyze_non_verbal

# --- Konfigurasi Halaman & Load Data ---

st.set_page_config(
    page_title="AI Interview Assessment",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def get_models():
    """Load semua model berat (hanya sekali)."""
    # Catatan: Jika menggunakan Streamlit Cloud gratis, pastikan STT_MODEL disetel ke 'small'
    # di file stt_processor.py untuk menghindari Out of Memory (OOM) error.
    stt_model = load_stt_model()
    embedder_model = load_embedder_model()
    spell, _, english_words = load_text_models()
    return stt_model, embedder_model, spell, english_words

# Load model di awal
STT_MODEL, EMBEDDER_MODEL, SPELL_CHECKER, ENGLISH_WORDS = get_models()

@st.cache_data
def load_questions():
    """Memuat pertanyaan dari questions.json."""
    try:
        # ASUMSI: File questions.json ada di root folder
        with open('questions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("File questions.json tidak ditemukan! Pastikan file ada.")
        return {}

@st.cache_data
def load_rubric_data():
    """Memuat data rubrik detail dari rubric_data.json."""
    try:
        # ASUMSI: File rubric_data.json ada di root folder
        with open('rubric_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("File rubric_data.json tidak ditemukan! Pastikan file ada.")
        return {}
        
QUESTIONS = load_questions()
RUBRIC_DATA = load_rubric_data()
TOTAL_QUESTIONS = len(QUESTIONS)


# --- FUNGSI NAVIGASI ---
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'candidate_info' not in st.session_state: st.session_state.candidate_info = {}
if 'answers' not in st.session_state: st.session_state.answers = {}
if 'current_q' not in st.session_state: st.session_state.current_q = 1
if 'results' not in st.session_state: st.session_state.results = None

def next_page(target_page):
    st.session_state.page = target_page
    if target_page == 'home': 
        st.session_state.current_q = 1
        st.session_state.results = None # Reset hasil saat kembali ke Home
    st.rerun()

def next_question():
    if st.session_state.current_q < TOTAL_QUESTIONS:
        st.session_state.current_q += 1
    else:
        st.session_state.page = 'processing'
    st.rerun()

# --- Halaman Utama (P1) ---
def render_home_page():
    st.title("🤝 AI Interview Assessment")
    st.markdown("---")
    st.subheader("Selamat Datang, Calon Kandidat!")
    st.info("Sistem ini akan menganalisis jawaban video Anda secara otomatis, mencakup: **Speech to Text**, **Analisis Confidence/Jawaban (Rubrik)**, dan **Analisis Non-Verbal (Tempo/Jeda)**.")
    
    # PERBAIKAN SINTAKS DITERAPKAN DI SINI (Menambahkan 'with')
    with st.container(border=True):
        st.metric(label="Total Pertanyaan", value=f"{TOTAL_QUESTIONS} Soal")
        st.markdown("**Petunjuk:** Anda akan diminta mengunggah satu video jawaban untuk setiap pertanyaan. Harap pastikan video berdurasi singkat (maksimal 1-2 menit per pertanyaan).")

    if st.button("🚀 Start Interview", use_container_width=True, type="primary"):
        next_page('info')

# --- Halaman Input Data Kandidat (P2) ---
def render_info_page():
    st.title("📝 Data Kandidat")
    st.markdown("Mohon lengkapi data diri Anda untuk memulai sesi interview.")

    with st.form("candidate_form"):
        name = st.text_input("Nama Lengkap", placeholder="Contoh: Budi Santoso")
        email = st.text_input("Email", placeholder="Contoh: budi.s@example.com")
        phone = st.text_input("Nomor Telepon", placeholder="Contoh: 0812xxxxxx")

        submitted = st.form_submit_button("Next → Start Interview", use_container_width=True, type="primary")

        if submitted:
            if name and email and phone:
                st.session_state.candidate_info = {'name': name, 'email': email, 'phone': phone}
                next_page('interview')
            else:
                st.error("Mohon lengkapi semua kolom.")

# --- Halaman Interview (P3) ---
def render_interview_page():
    q_id = str(st.session_state.current_q)
    q_num = st.session_state.current_q
    question_data = QUESTIONS.get(f'q{q_id}', {}) # Menggunakan f-string untuk mendapatkan key 'q1', 'q2', dst
    
    if not question_data:
        st.error("Pertanyaan tidak valid.")
        next_page('home')
        return

    st.header(f"Pertanyaan {q_num} dari {TOTAL_QUESTIONS}")
    st.markdown("---")

    # Tampilan Pertanyaan
    st.subheader("Soal:")
    st.info(question_data['question'])
    
    # Kolom Upload Video
    uploaded_file = st.file_uploader(
        "Upload Video Jawaban (Max 50MB)",
        type=['mp4', 'mov', 'webm'],
        key=f"uploader_{q_id}"
    )

    # Pratinjau Video (jika sudah diupload)
    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("Pratinjau Video Anda")
        st.video(uploaded_file, format='video/mp4')
    
    # Simpan file yang sudah ada di session state jika ada
    uploaded_file_in_state = st.session_state.answers.get(q_id)

    col1, col2 = st.columns([1, 1])

    with col1:
        if q_num > 1:
            if st.button("← Previous Question", use_container_width=True):
                st.session_state.current_q -= 1
                st.rerun()

    with col2:
        # Cek apakah ada file yang diupload di sesi saat ini atau di session state sebelumnya
        is_file_available = uploaded_file is not None or uploaded_file_in_state is not None
        
        button_label = "Submit & Finish" if q_num == TOTAL_QUESTIONS else "Submit & Next Question →"
        
        if st.button(button_label, use_container_width=True, type="primary"):
            if is_file_available:
                if uploaded_file is not None:
                    # Simpan file baru ke session state
                    st.session_state.answers[q_id] = uploaded_file
                    st.success(f"Jawaban untuk Pertanyaan {q_num} berhasil di-upload.")
                elif uploaded_file_in_state is not None and uploaded_file is None:
                    # Gunakan file yang sudah ada di state, tanpa perlu menampilkan success message berulang
                    pass

                next_question()
            else:
                st.error("Mohon upload video jawaban Anda sebelum melanjutkan.")

# --- Halaman Pemrosesan dan Hasil (P4) ---
def render_processing_page():
    st.title("⏳ Pemrosesan Jawaban...")
    st.info("Harap tunggu sebentar. Sistem AI sedang menganalisis video yang Anda unggah. Proses ini mungkin memakan waktu 1-3 menit, tergantung pada ukuran model STT dan kecepatan koneksi.")

    if st.session_state.results is None:
        
        results = {}
        progress_bar = st.progress(0, text="Mempersiapkan Model...")
        
        # Gunakan tempfile.TemporaryDirectory untuk penyimpanan sementara
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                
                for i in range(1, TOTAL_QUESTIONS + 1):
                    q_id = str(i)
                    # Menggunakan f-string untuk memastikan key yang benar (misalnya 'q1')
                    q_key = f'q{q_id}' 
                    q_text = QUESTIONS.get(q_key, {}).get('question')

                    if q_id in st.session_state.answers and q_key and q_text:
                        video_file = st.session_state.answers[q_id]
                        
                        # 1. Simpan video yang di-upload ke disk sementara
                        progress_text = f"Q{q_id}: Menyimpan file video sementara..."
                        progress_bar.progress(int((i * 0.1 / (TOTAL_QUESTIONS * 3 + 1)) * 100), text=progress_text)

                        video_path = os.path.join(temp_dir, f"q{q_id}_video.mp4")
                        with open(video_path, "wb") as f:
                            f.write(video_file.getbuffer())

                        # 2. Ekstraksi dan Preprocessing Audio (Noise Reduction)
                        progress_text = f"Q{q_id}: Ekstraksi & Noise Reduction..."
                        progress_bar.progress(int((i * 1 / (TOTAL_QUESTIONS * 3 + 1)) * 100), text=progress_text)
                        
                        audio_raw_path = os.path.join(temp_dir, f"q{q_id}_audio_raw.wav")
                        audio_clean_path = os.path.join(temp_dir, f"q{q_id}_audio_clean.wav")
                        
                        # Convert Video to WAV
                        video_to_wav(video_path, audio_raw_path)

                        # Noise Reduction
                        noise_reduction(audio_raw_path, audio_clean_path) 
                        
                        # 3. Model STT & Cleaning
                        progress_text = f"Q{q_id}: Transkripsi & Cleaning Teks..."
                        progress_bar.progress(int((i * 2 / (TOTAL_QUESTIONS * 3 + 1)) * 100), text=progress_text)
                        transcript = transcribe_and_clean(
                            audio_clean_path, STT_MODEL, SPELL_CHECKER, EMBEDDER_MODEL, ENGLISH_WORDS
                        )
                        
                        # 4. Analisis Non-Verbal
                        progress_text = f"Q{q_id}: Analisis Non-Verbal & Penilaian..."
                        progress_bar.progress(int((i * 3 / (TOTAL_QUESTIONS * 3 + 1)) * 100), text=progress_text)
                        non_verbal_data = analyze_non_verbal(audio_clean_path)

                        # 5. Penilaian Jawaban (Semantik)
                        # Rubric key harus sesuai dengan structure RUBRIC_DATA
                        score, reason = score_with_rubric(
                            q_key, q_text, transcript, RUBRIC_DATA, EMBEDDER_MODEL
                        )
                        
                        # 6. Confidence Score
                        confidence = compute_confidence_score(transcript, q_text, EMBEDDER_MODEL)

                        results[q_key] = {
                            "transcript": transcript,
                            "score_jawaban": score,
                            "rubric_reason": reason,
                            "confidence_score": f"{confidence:.2f}", # Disimpan dengan 2 desimal
                            "non_verbal": non_verbal_data,
                            "pertanyaan": q_text
                        }

                progress_bar.progress(100, text="Analisis Selesai! Menampilkan Hasil...")
                st.session_state.results = results
                st.balloons()
                st.rerun()
        
        except Exception as e:
            st.error(f"Terjadi kesalahan fatal selama pemrosesan: {e}")
            st.warning("Ini mungkin disebabkan oleh keterbatasan memori (OOM) Streamlit Cloud. Coba ubah model STT Anda menjadi 'small' di `models/stt_processor.py`.")
            progress_bar.empty()
            st.session_state.results = {} # Reset untuk menghindari loop tak terbatas

    # Tampilan Hasil
    if st.session_state.results:
        st.title("✅ Hasil Assessment AI")
        st.markdown(f"**Nama Kandidat:** {st.session_state.candidate_info.get('name', 'N/A')}")
        st.markdown(f"**Email:** {st.session_state.candidate_info.get('email', 'N/A')}")
        st.markdown("---")

        # Rangkuman Skor
        total_score = sum(st.session_state.results[q]['score_jawaban'] for q in st.session_state.results)
        max_score = TOTAL_QUESTIONS * 4 # Skor maksimal per pertanyaan adalah 4
        final_percentage = (total_score / max_score) * 100 if max_score > 0 else 0

        col_score1, col_score2 = st.columns(2)
        with col_score1:
            st.metric("Total Skor Jawaban (Maks 4)", f"{total_score} / {max_score}")
        with col_score2:
            st.metric("Persentase Kelulusan", f"{final_percentage:.2f}%")
        
        st.markdown("---")

        # Detail per Pertanyaan
        st.subheader("Detail Analisis per Pertanyaan")
        
        for q_key, res in st.session_state.results.items():
            # Mengambil nomor pertanyaan dari key (misal 'q1' -> 1)
            q_num = q_key[1:] 
            expander = st.expander(f"**Pertanyaan {q_num}:** {res['pertanyaan']}", expanded=False)
            with expander:
                st.markdown(f"#### Hasil Penilaian AI")
                
                # Tentukan warna based on score (misal >= 3: success)
                score_color = "green" if res['score_jawaban'] >= 3 else ("orange" if res['score_jawaban'] >= 1 else "red")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.markdown(f"**Skor Jawaban:** <span style='color:{score_color}; font-size: 24px;'>{res['score_jawaban']}</span>", unsafe_allow_html=True)
                with col_res2:
                    st.metric("Confidence Score", f"{res['confidence_score']}%")
                with col_res3:
                    st.metric("Skor Non-Verbal", res['non_verbal'].get('qualitative_summary', 'N/A'))
                
                st.markdown("---")

                st.markdown("##### Rubrik Penilaian (Alasan Semantik)")
                st.caption(f"**Alasan Pemberian Skor:** {res['rubric_reason']}")

                st.markdown("##### Analisis Audio (Non-Verbal Detail)")
                # Tampilkan detail tempo dan pause
                tempo = res['non_verbal'].get('tempo_bpm', 'N/A')
                pause = res['non_verbal'].get('total_pause_seconds', 'N/A')
                st.markdown(f"* **Tempo Bicara (BPM):** {tempo}")
                st.markdown(f"* **Total Jeda (Detik):** {pause}")

                st.markdown("##### Transkrip Jawaban Bersih")
                st.code(res['transcript'], language='text')

        if st.button("🏠 Selesai & Kembali ke Awal", use_container_width=True):
            # Membersihkan session state untuk memulai ulang aplikasi sepenuhnya
            st.session_state.clear() 
            next_page('home')

# --- Main App Execution Flow ---
if st.session_state.page == 'home':
    render_home_page()
elif st.session_state.page == 'info':
    render_info_page()
elif st.session_state.page == 'interview':
    render_interview_page()
elif st.session_state.page == 'processing':
    render_processing_page()
