import math
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Tomath02/proto4")
LOCAL_MODEL_DIR = os.getenv("LOCAL_INDOBERT_MODEL_DIR", "").strip()
SVM_MODEL_PATH = Path(os.getenv("SVM_MODEL_PATH", APP_DIR / "models" / "svm" / "tfidf_svm_pipeline.joblib"))
INDOBERT_THRESHOLD = float(os.getenv("INDOBERT_THRESHOLD", "0.5"))
SVM_THRESHOLD = float(os.getenv("SVM_THRESHOLD", "0.0"))


st.set_page_config(
    page_title="Deteksi Teks AI Jurnal Indonesia",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {
        max-width: 1180px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    .stTextArea textarea {
        font-size: 15px;
        line-height: 1.55;
        border-radius: 6px;
    }
    .stButton > button {
        width: 100%;
        height: 2.75rem;
        border-radius: 6px;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_text(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_reference_section(text: str) -> str:
    pattern = re.compile(
        r"\n\s*(daftar\s+pustaka|referensi|references|bibliography)\s*\n",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text or "")
    return text[: match.start()].strip() if match else text


def regroup_lines(lines: list[str], target_words: int) -> list[str]:
    groups: list[str] = []
    buffer: list[str] = []
    word_count = 0

    for line in lines:
        line_words = len(line.split())
        if buffer and word_count + line_words > target_words:
            groups.append(" ".join(buffer))
            buffer = [line]
            word_count = line_words
        else:
            buffer.append(line)
            word_count += line_words

    if buffer:
        groups.append(" ".join(buffer))
    return groups


def split_by_words(text: str, target_words: int, overlap_words: int, min_words: int) -> list[str]:
    words = compact_text(text).split()
    if len(words) <= target_words:
        return [compact_text(text)] if len(words) >= min_words else []

    chunks: list[str] = []
    step = max(1, target_words - overlap_words)
    start = 0
    while start < len(words):
        part = words[start : start + target_words]
        if len(part) >= min_words:
            chunks.append(" ".join(part))
        if start + target_words >= len(words):
            break
        start += step
    return chunks


def split_blocks(text: str, min_words: int) -> list[str]:
    text = strip_reference_section(normalize_text(text))
    raw_blocks = [compact_text(part) for part in re.split(r"\n\s*\n+", text) if compact_text(part)]

    if len(raw_blocks) <= 1 and len(text.split()) > 260:
        lines = [compact_text(line) for line in text.splitlines() if compact_text(line)]
        raw_blocks = regroup_lines(lines, target_words=150)

    blocks: list[str] = []
    for block in raw_blocks:
        words = block.split()
        if len(words) < min_words:
            continue
        if len(words) > 230:
            blocks.extend(split_by_words(block, target_words=150, overlap_words=25, min_words=min_words))
        else:
            blocks.append(block)
    return blocks


def extract_pdf_text(uploaded_file) -> str:
    import fitz

    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    pages = [page.get_text("text") or "" for page in doc]
    return "\n\n".join(pages).strip()


def extract_txt_text(uploaded_file) -> str:
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="ignore")
    return str(raw)


def infer_ai_label_id(config) -> int:
    label2id = {str(key).lower(): int(value) for key, value in (getattr(config, "label2id", None) or {}).items()}
    if "ai" in label2id:
        return label2id["ai"]

    id2label = {int(key): str(value).lower() for key, value in (getattr(config, "id2label", None) or {}).items()}
    for idx, label in id2label.items():
        if label == "ai" or "ai" in label:
            return idx
    return 1


def resolve_model_source() -> str:
    if LOCAL_MODEL_DIR and Path(LOCAL_MODEL_DIR).exists():
        return LOCAL_MODEL_DIR
    return HF_MODEL_REPO


@st.cache_resource(show_spinner="Memuat model IndoBERT...")
def load_indobert_model(model_source: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    source_path = Path(model_source)
    use_local = source_path.exists()

    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=use_local)
    model = AutoModelForSequenceClassification.from_pretrained(model_source, local_files_only=use_local)
    ai_label_id = infer_ai_label_id(model.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    id2label = {int(key): str(value).lower() for key, value in (model.config.id2label or {}).items()}
    return tokenizer, model, device, ai_label_id, id2label


@st.cache_resource(show_spinner="Memuat baseline TF-IDF + SVM...")
def load_svm_model(model_path: str):
    import joblib

    return joblib.load(model_path)


def build_token_chunks(
    blocks: list[str],
    tokenizer,
    max_length: int,
    overlap_tokens: int,
    min_chunk_tokens: int = 24,
) -> list[dict]:
    chunk_size = max(32, max_length - 2)
    overlap = min(overlap_tokens, max(0, chunk_size - 8))
    step = max(1, chunk_size - overlap)
    rows: list[dict] = []

    for block_index, block in enumerate(blocks, start=1):
        block = compact_text(block)
        token_ids = tokenizer.encode(block, add_special_tokens=False)
        if not token_ids:
            continue

        if len(token_ids) <= chunk_size:
            rows.append(
                {
                    "block": block_index,
                    "chunk_index": 1,
                    "chunk_count": 1,
                    "token_start": 0,
                    "token_end": len(token_ids),
                    "token_count": len(token_ids),
                    "text": block,
                }
            )
            continue

        ranges: list[tuple[int, int]] = []
        for start in range(0, len(token_ids), step):
            end = min(len(token_ids), start + chunk_size)
            if end - start < min_chunk_tokens and ranges:
                prev_start, _ = ranges[-1]
                ranges[-1] = (prev_start, len(token_ids))
                break
            ranges.append((start, end))
            if end >= len(token_ids):
                break

        for chunk_index, (start, end) in enumerate(ranges, start=1):
            window_ids = token_ids[start:end]
            chunk_text = compact_text(tokenizer.decode(window_ids, skip_special_tokens=True))
            if not chunk_text:
                continue
            rows.append(
                {
                    "block": block_index,
                    "chunk_index": chunk_index,
                    "chunk_count": len(ranges),
                    "token_start": start,
                    "token_end": end,
                    "token_count": len(window_ids),
                    "text": chunk_text,
                }
            )
    return rows


def build_svm_chunks(blocks: list[str]) -> list[dict]:
    rows: list[dict] = []
    for block_index, block in enumerate(blocks, start=1):
        block = compact_text(block)
        if block:
            rows.append(
                {
                    "block": block_index,
                    "chunk_index": 1,
                    "chunk_count": 1,
                    "word_count": len(block.split()),
                    "text": block,
                }
            )
    return rows


def predict_indobert_batch(chunk_rows: list[dict], tokenizer, model, device, ai_label_id: int, max_length: int) -> list[float]:
    import torch

    scores: list[float] = []
    batch_size = 16 if str(device) == "cuda" else 4
    texts = [row["text"] for row in chunk_rows]

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1)[:, ai_label_id]
            scores.extend(float(value) for value in probs.detach().cpu())
    return scores


def predict_svm_batch(chunk_rows: list[dict], svm_model) -> list[float]:
    scores = svm_model.decision_function([row["text"] for row in chunk_rows])
    return [float(value) for value in scores]


def aggregate_score(scores: list[float], mode: str) -> float:
    if not scores:
        return 0.0
    if mode == "max":
        return max(scores)
    if mode == "median":
        ordered = sorted(scores)
        mid = len(ordered) // 2
        return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return sum(scores) / len(scores)


def render_indobert_result(prob_ai: float, threshold: float) -> None:
    label = "AI" if prob_ai >= threshold else "Human"
    color = "#b91c1c" if label == "AI" else "#15803d"
    st.markdown(f"<h1 style='margin-bottom:0;color:{color}'>{label}</h1>", unsafe_allow_html=True)
    st.metric("Probabilitas AI", f"{prob_ai:.2%}")
    st.progress(min(1.0, max(0.0, prob_ai)))


def render_svm_result(score_ai: float, threshold: float) -> None:
    label = "AI" if score_ai >= threshold else "Human"
    color = "#b91c1c" if label == "AI" else "#15803d"
    indicator = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, score_ai - threshold))))
    st.markdown(f"<h1 style='margin-bottom:0;color:{color}'>{label}</h1>", unsafe_allow_html=True)
    st.metric("Skor SVM AI", f"{score_ai:.4f}")
    st.caption("Skor SVM adalah decision score, bukan probabilitas.")
    st.progress(min(1.0, max(0.0, indicator)))


st.title("Deteksi Teks Hasil Generative AI")
st.caption("Model IndoBERT dan baseline TF-IDF + SVM untuk paper jurnal berbahasa Indonesia.")

with st.sidebar:
    st.header("Konfigurasi")
    model_choice = st.selectbox("Model", ["IndoBERT", "TF-IDF + SVM"], index=0)
    min_words = st.slider("Minimal kata per blok", 10, 120, 35, 5)
    aggregate_mode = st.selectbox("Agregasi skor", ["mean", "median", "max"], index=0)

    if model_choice == "IndoBERT":
        model_source = st.text_input("Model IndoBERT", value=resolve_model_source())
        threshold = st.slider("Batas keputusan AI", 0.01, 0.99, INDOBERT_THRESHOLD, 0.01)
        max_length = st.select_slider("Max token length", options=[96, 128, 256, 384, 512], value=256)
        overlap_tokens = st.slider("Token overlap", 0, 96, 32, 8)
        st.caption("Default deploy memakai Hugging Face repo. Untuk lokal, bisa pakai LOCAL_INDOBERT_MODEL_DIR.")
    else:
        svm_model_path = st.text_input("Path model SVM", value=str(SVM_MODEL_PATH))
        svm_threshold = st.slider("Batas skor SVM", -2.0, 2.0, SVM_THRESHOLD, 0.05)
        st.caption("Batas default 0 berarti skor positif diprediksi AI dan skor negatif diprediksi human.")

left, right = st.columns([1.35, 1], gap="large")

with left:
    st.subheader("Input")
    input_mode = st.radio("Sumber input", ["Teks", "File PDF/TXT"], horizontal=True)
    text = ""

    if input_mode == "Teks":
        text = st.text_area(
            "Teks paper jurnal",
            height=340,
            placeholder="Tempel paragraf artikel ilmiah berbahasa Indonesia di sini.",
        )
    else:
        uploaded_file = st.file_uploader("Upload file", type=["pdf", "txt"])
        if uploaded_file is not None:
            try:
                suffix = Path(uploaded_file.name).suffix.lower()
                text = extract_pdf_text(uploaded_file) if suffix == ".pdf" else extract_txt_text(uploaded_file)
                st.caption(f"Teks terambil: {len(text.split())} kata")
            except Exception as exc:
                st.error(f"Gagal membaca file: {exc}")

    analyze = st.button("Jalankan Deteksi", type="primary")

with right:
    st.subheader("Hasil")
    if not analyze:
        st.info("Masukkan teks atau file, lalu jalankan deteksi.")
    elif len((text or "").split()) < 10:
        st.warning("Input terlalu pendek untuk dianalisis.")
    else:
        try:
            blocks = split_blocks(text, min_words=min_words)
            if not blocks and text.strip():
                blocks = [compact_text(text)]

            if model_choice == "IndoBERT":
                tokenizer, model, device, ai_label_id, id2label = load_indobert_model(model_source)
                chunk_rows = build_token_chunks(
                    blocks,
                    tokenizer,
                    max_length=max_length,
                    overlap_tokens=overlap_tokens,
                )
                if not chunk_rows:
                    raise ValueError("Tidak ada chunk valid setelah preprocessing.")

                scores = predict_indobert_batch(chunk_rows, tokenizer, model, device, ai_label_id, max_length)
                final_score = aggregate_score(scores, aggregate_mode)
                render_indobert_result(final_score, threshold)

                stat_cols = st.columns(3)
                stat_cols[0].metric("Blok", len(blocks))
                stat_cols[1].metric("Token-chunk", len(chunk_rows))
                stat_cols[2].metric("Threshold", f"{threshold:.2f}")
                st.caption(f"Device: {device} | AI label id: {ai_label_id} | id2label: {id2label} | agregasi: {aggregate_mode}")

                chunk_df = pd.DataFrame(chunk_rows)
                chunk_df["prob_ai"] = scores
                chunk_df["prediksi"] = ["AI" if score >= threshold else "Human" for score in scores]
                score_column = "prob_ai"
                score_label = "Probabilitas AI"
                download_name = "hasil_deteksi_indobert.csv"
                display_columns = [
                    "block",
                    "chunk_index",
                    "chunk_count",
                    "token_count",
                    score_column,
                    "prediksi",
                    "preview",
                ]
            else:
                svm_path = Path(svm_model_path)
                if not svm_path.exists():
                    raise FileNotFoundError(f"Model SVM tidak ditemukan: {svm_path}")

                svm_model = load_svm_model(str(svm_path))
                chunk_rows = build_svm_chunks(blocks)
                if not chunk_rows:
                    raise ValueError("Tidak ada chunk valid setelah preprocessing.")

                scores = predict_svm_batch(chunk_rows, svm_model)
                final_score = aggregate_score(scores, aggregate_mode)
                render_svm_result(final_score, svm_threshold)

                stat_cols = st.columns(3)
                stat_cols[0].metric("Blok", len(blocks))
                stat_cols[1].metric("Chunk teks", len(chunk_rows))
                stat_cols[2].metric("Batas SVM", f"{svm_threshold:.2f}")
                st.caption(f"Model: TF-IDF + LinearSVC | agregasi: {aggregate_mode}")

                chunk_df = pd.DataFrame(chunk_rows)
                chunk_df["svm_score_ai"] = scores
                chunk_df["prediksi"] = ["AI" if score >= svm_threshold else "Human" for score in scores]
                score_column = "svm_score_ai"
                score_label = "Skor SVM AI"
                download_name = "hasil_deteksi_svm.csv"
                display_columns = [
                    "block",
                    "chunk_index",
                    "chunk_count",
                    "word_count",
                    score_column,
                    "prediksi",
                    "preview",
                ]

            chunk_df["preview"] = chunk_df["text"].str.slice(0, 190)
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Agregasi": "mean", score_label: aggregate_score(scores, "mean")},
                        {"Agregasi": "median", score_label: aggregate_score(scores, "median")},
                        {"Agregasi": "max", score_label: aggregate_score(scores, "max")},
                    ]
                ),
                hide_index=True,
                width="stretch",
                column_config={score_label: st.column_config.NumberColumn(score_label, format="%.6f")},
            )

            st.dataframe(
                chunk_df[display_columns],
                hide_index=True,
                width="stretch",
                column_config={
                    score_column: st.column_config.NumberColumn(score_label, format="%.6f"),
                    "preview": st.column_config.TextColumn("preview", width="large"),
                },
                row_height=86,
            )

            csv_bytes = chunk_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download hasil chunk",
                data=csv_bytes,
                file_name=download_name,
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Gagal menjalankan deteksi: {exc}")
            if model_choice == "IndoBERT" and model_source == HF_MODEL_REPO:
                st.info(
                    "Pastikan repo Hugging Face berisi file model IndoBERT, seperti config.json, "
                    "model.safetensors, tokenizer.json, dan tokenizer_config.json."
                )

st.caption("Output model digunakan sebagai indikator probabilistik dan tetap perlu dibaca sesuai konteks penggunaan.")
