import json
import math
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = APP_DIR.parent
WORKSPACE_DIR = PIPELINE_DIR.parent
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Tomath02/proto4").strip()
HF_MODEL_SUBFOLDER = os.getenv("HF_MODEL_SUBFOLDER", "best_model").strip()
LOCAL_MODEL_DIR = os.getenv("LOCAL_INDOBERT_MODEL_DIR", "").strip()

MODEL_CANDIDATES = [
    Path(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR else None,
    PIPELINE_DIR / "artifacts" / "indobert_final_chatgpt55_clean_p1" / "best_model",
    WORKSPACE_DIR
    / "tmp"
    / "drive_pipeline9_latest"
    / "artifacts"
    / "indobert_final_chatgpt55_clean_p1"
    / "best_model",
]


def is_valid_local_model(candidate: Path | None) -> bool:
    return bool(
        candidate
        and (candidate / "config.json").exists()
        and ((candidate / "model.safetensors").exists() or (candidate / "pytorch_model.bin").exists())
    )


def pick_default_model_source() -> str:
    for candidate in MODEL_CANDIDATES:
        if is_valid_local_model(candidate):
            return str(candidate)
    return HF_MODEL_REPO


DEFAULT_MODEL_SOURCE = pick_default_model_source()
DEFAULT_SVM_MODEL_PATH = Path(
    os.getenv("SVM_MODEL_PATH", APP_DIR / "models" / "svm" / "tfidf_svm_pipeline.joblib")
)


FIXED_INDOBERT_THRESHOLD = 0.5
FIXED_SVM_THRESHOLD = 0.0
FIXED_MAX_LENGTH = 256
FIXED_TOKEN_OVERLAP = 0
TRAINING_MIN_WORDS = 45
TRAINING_MAX_WORDS = 230
SELECTION_STATUS_PASS = "Lolos"
SELECTION_STATUS_SKIP = "Dilewat"

REFERENCE_HEADING_RE = re.compile(
    r"^\s*(daftar\s+pustaka|referensi|references|bibliography)\s*$",
    flags=re.IGNORECASE,
)
NOISE_BLOCK_RE = re.compile(
    r"(^\s*\d+\s*$|issn\s*:?\s*\d+|e-issn|p-issn|vol\.?\s*\d+|"
    r"volume\s+\d+|no\.?\s*\d+|copyright|doi\s*:|email\s*:|e-mail\s*:)",
    flags=re.IGNORECASE,
)


st.set_page_config(
    page_title="Deteksi Teks AI Pipeline9",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    .stTextArea textarea {
        font-size: 15px;
        line-height: 1.5;
        border-radius: 6px;
    }
    .stButton > button {
        width: 100%;
        height: 2.75rem;
        border-radius: 6px;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.55rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w-]+\b", text or "", flags=re.UNICODE))


def normalize_text(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


def clean_candidate_text(text: str) -> str:
    text = compact_text(text)
    text = re.sub(r"^(?:\(?\d+(?:\.\d+)*\)?[.)-]?\s+){1,3}", "", text)
    text = re.sub(r"^[a-z]\s*[\)\.\-]\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:[ivxlcdm]+[.)-]\s+)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:gambar|tabel|table|fig(?:ure)?\.?)\s*\d+\s*[:.\-]?\s*", "", text, flags=re.IGNORECASE)
    return compact_text(text)


def looks_english(text: str) -> bool:
    lowered = f" {(text or '').lower()} "
    markers = [
        " the ",
        " and ",
        " this ",
        " that ",
        " research ",
        " study ",
        " result ",
        " abstract ",
        " keyword ",
    ]
    return sum(1 for marker in markers if marker in lowered) >= 3


def high_numeric_ratio(text: str, ratio_limit: float = 0.20) -> bool:
    tokens = re.findall(r"\b[\w.-]+\b", text or "", flags=re.UNICODE)
    if not tokens:
        return True
    numeric = sum(1 for token in tokens if re.fullmatch(r"\d+(?:[.,]\d+)?", token))
    return (numeric / max(1, len(tokens))) > ratio_limit


def looks_like_list_or_table_residual(text: str) -> bool:
    clean = compact_text(text)
    numbered_items = len(re.findall(r"\b\d{1,3}\)", clean))
    loose_numbered_items = len(re.findall(r"(?:^|\s)\d{1,3}\s+[A-Za-z]", clean))
    item_codes = len(re.findall(r"\b[A-Z]-\d{2,4}\b", clean))
    money_like = len(re.findall(r"\bRp\b|\b\d{1,3}(?:\.\d{3})+\b", clean, flags=re.IGNORECASE))
    brace_rules = len(re.findall(r"[{}|]", clean))
    return (
        numbered_items >= 4
        or loose_numbered_items >= 8
        or item_codes >= 3
        or money_like >= 5
        or brace_rules >= 8
    )


def selection_reason(text: str, is_reference_block: bool, min_words: int, max_words: int) -> str:
    if is_reference_block:
        return "Bagian referensi/daftar pustaka tidak digunakan pada training."
    if not text:
        return "Teks kosong setelah preprocessing."

    count = word_count(text)
    reasons: list[str] = []
    if count < min_words:
        reasons.append(f"Jumlah kata {count}, kurang dari batas training {min_words}.")
    if count > max_words:
        reasons.append(f"Jumlah kata {count}, melebihi batas training {max_words}.")
    if NOISE_BLOCK_RE.search(text) and count < min_words:
        reasons.append("Terdeteksi sebagai metadata/header/footer.")
    if looks_english(text):
        reasons.append("Terdeteksi dominan bahasa Inggris.")
    if looks_like_list_or_table_residual(text):
        reasons.append("Terdeteksi seperti sisa tabel/list.")
    if high_numeric_ratio(text):
        reasons.append("Proporsi angka terlalu tinggi.")
    return " ".join(reasons)


def build_block_selection_rows(text: str, min_words: int, max_words: int) -> list[dict]:
    normalized = normalize_text(text)
    raw_blocks = [part for part in re.split(r"\n\s*\n+", normalized) if compact_text(part)]

    if len(raw_blocks) <= 1 and len(text.split()) > 260:
        lines = [compact_text(line) for line in text.splitlines() if compact_text(line)]
        raw_blocks = regroup_lines(lines, target_words=150)

    rows: list[dict] = []
    in_reference_section = False
    for block_index, raw_block in enumerate(raw_blocks, start=1):
        raw_text = compact_text(raw_block)
        reference_heading = bool(REFERENCE_HEADING_RE.match(raw_text)) or bool(
            re.match(r"^(daftar\s+pustaka|referensi|references|bibliography)\b", raw_text, flags=re.IGNORECASE)
        )
        if reference_heading:
            in_reference_section = True

        clean_text = clean_candidate_text(raw_text)
        reason = selection_reason(
            clean_text,
            is_reference_block=in_reference_section,
            min_words=min_words,
            max_words=max_words,
        )
        status = SELECTION_STATUS_SKIP if reason else SELECTION_STATUS_PASS
        rows.append(
            {
                "block": block_index,
                "status": status,
                "alasan": reason if reason else "Memenuhi seleksi preprocessing training.",
                "word_count": word_count(clean_text),
                "text": clean_text,
                "preview": clean_text[:190],
            }
        )
    return rows


def extract_pdf_text(uploaded_file) -> str:
    import fitz

    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text("text") or "")
    return "\n\n".join(pages).strip()


def extract_txt_text(uploaded_file) -> str:
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="ignore")
    return str(raw)


def load_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_ai_label_id(config) -> int:
    if not isinstance(config, dict):
        config = {
            "label2id": getattr(config, "label2id", None),
            "id2label": getattr(config, "id2label", None),
        }

    label2id = {str(k).lower(): int(v) for k, v in (config.get("label2id") or {}).items()}
    if "ai" in label2id:
        return label2id["ai"]

    id2label = {int(k): str(v).lower() for k, v in (config.get("id2label") or {}).items()}
    for idx, label in id2label.items():
        if label == "ai" or "ai" in label:
            return idx
    return 1


@st.cache_resource(show_spinner="Memuat model IndoBERT...")
def load_indobert_model(model_source: str, hf_subfolder: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = Path(model_source)
    use_local = path.exists()
    subfolder = None if use_local or not hf_subfolder.strip() else hf_subfolder.strip()

    if use_local:
        config = load_config(path)
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_source, subfolder=subfolder)
        model = AutoModelForSequenceClassification.from_pretrained(model_source, subfolder=subfolder)
        config = model.config

    ai_label_id = infer_ai_label_id(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = {int(k): str(v).lower() for k, v in (model.config.id2label or {}).items()}
    return tokenizer, model, device, ai_label_id, id2label


@st.cache_resource(show_spinner="Memuat model TF-IDF + SVM...")
def load_svm_model(model_path: str):
    import joblib

    return joblib.load(model_path)


def build_token_chunks(
    blocks: list[dict],
    tokenizer,
    max_length: int,
    overlap_tokens: int,
    min_chunk_tokens: int = 24,
) -> list[dict]:
    chunk_size = max(32, max_length - 2)
    overlap = min(overlap_tokens, max(0, chunk_size - 8))
    step = max(1, chunk_size - overlap)
    rows: list[dict] = []

    for block_row in blocks:
        block_index = int(block_row["block"])
        block = compact_text(block_row["text"])
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


def build_svm_chunks(blocks: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for block_row in blocks:
        block_index = int(block_row["block"])
        block = compact_text(block_row["text"])
        if not block:
            continue
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


def predict_batch(chunk_rows: list[dict], tokenizer, model, device, ai_label_id: int, max_length: int) -> list[float]:
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
    texts = [row["text"] for row in chunk_rows]
    scores = svm_model.decision_function(texts)
    return [float(value) for value in scores]


def average_score(scores: list[float]) -> float:
    if not scores:
        return 0.0
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
    st.caption("Skor >= batas berarti AI. Skor ini bukan probabilitas.")
    st.progress(min(1.0, max(0.0, indicator)))


def render_score_explanation(
    model_name: str,
    score_label: str,
    final_score: float,
    threshold: float,
    selected_count: int,
    skipped_count: int,
    chunk_count: int,
    label: str,
) -> None:
    st.dataframe(
        pd.DataFrame(
            [
                {"Komponen": "Blok lolos seleksi", "Nilai": selected_count},
                {"Komponen": "Blok dilewat", "Nilai": skipped_count},
                {"Komponen": "Chunk/blok yang diprediksi", "Nilai": chunk_count},
                {"Komponen": score_label, "Nilai": f"{final_score:.6f}"},
                {"Komponen": "Batas keputusan", "Nilai": f"{threshold:.2f}"},
                {"Komponen": "Hasil akhir", "Nilai": label},
            ]
        ),
        hide_index=True,
        width="stretch",
    )


st.title("Deteksi Teks Hasil Generative AI")
st.caption("Pipeline9 IndoBERT dan baseline TF-IDF + SVM untuk paper jurnal berbahasa Indonesia")

with st.sidebar:
    st.header("Konfigurasi")
    model_choice = st.selectbox("Model", ["IndoBERT", "TF-IDF + SVM"], index=0)
    if model_choice == "IndoBERT":
        model_dir = st.text_input("Path model IndoBERT", value=str(DEFAULT_MODEL_SOURCE))
        hf_subfolder = st.text_input("Subfolder Hugging Face", value=HF_MODEL_SUBFOLDER)
        threshold = FIXED_INDOBERT_THRESHOLD
        st.caption(f"Batas keputusan IndoBERT: probabilitas AI >= {threshold:.2f}.")
        max_length = FIXED_MAX_LENGTH
        overlap_tokens = FIXED_TOKEN_OVERLAP
        st.caption(f"Batas masukan IndoBERT: {max_length} token per chunk.")
    else:
        svm_model_path = st.text_input("Path model SVM", value=str(DEFAULT_SVM_MODEL_PATH))
        svm_threshold = FIXED_SVM_THRESHOLD
        st.caption(f"Batas keputusan SVM: skor >= {svm_threshold:.2f}.")
    min_words = TRAINING_MIN_WORDS
    max_words = TRAINING_MAX_WORDS
    st.markdown("**Seleksi blok**")
    st.caption(
        f"Mengikuti data training: {min_words}-{max_words} kata per blok. "
        "Blok yang tidak lolos tetap ditampilkan sebagai dilewat."
    )
    if model_choice == "IndoBERT":
        st.caption(
            "Hasil akhir dihitung dari rata-rata skor blok/chunk yang lolos seleksi."
        )
    else:
        st.caption(
            "SVM memakai decision score. Hasil akhir dihitung dari rata-rata skor blok yang lolos seleksi."
        )

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
                if suffix == ".pdf":
                    text = extract_pdf_text(uploaded_file)
                else:
                    text = extract_txt_text(uploaded_file)
                st.caption(f"Teks terambil: {len(text.split())} kata")
            except Exception as exc:
                st.error(f"Gagal membaca file: {exc}")

    analyze = st.button("Jalankan Deteksi", type="primary")

with right:
    st.subheader("Hasil")
    if not analyze:
        st.info("Masukkan teks atau file, lalu jalankan deteksi.")
    elif model_choice == "IndoBERT" and not model_dir.strip():
        st.error(f"Model IndoBERT tidak ditemukan: {model_dir}")
    elif model_choice == "TF-IDF + SVM" and not Path(svm_model_path).exists():
        st.error(f"Model SVM tidak ditemukan: {svm_model_path}")
    elif len((text or "").split()) < 10:
        st.warning("Input terlalu pendek untuk dianalisis.")
    else:
        try:
            block_rows = build_block_selection_rows(text, min_words=min_words, max_words=max_words)
            selected_blocks = [row for row in block_rows if row["status"] == SELECTION_STATUS_PASS]
            skipped_count = len(block_rows) - len(selected_blocks)

            selection_df = pd.DataFrame(
                block_rows,
                columns=["block", "status", "alasan", "word_count", "text", "preview"],
            )
            selection_columns = ["block", "status", "word_count", "alasan", "preview"]

            if not selected_blocks:
                st.warning(
                    "Tidak ada blok yang lolos seleksi preprocessing training. "
                    "Blok tetap ditampilkan agar bagian yang dilewat dapat diperiksa."
                )
                st.dataframe(
                    selection_df[selection_columns],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "preview": st.column_config.TextColumn("preview", width="large"),
                        "alasan": st.column_config.TextColumn("alasan", width="large"),
                    },
                    row_height=86,
                )
                st.stop()

            if model_choice == "IndoBERT":
                tokenizer, model, device, ai_label_id, id2label = load_indobert_model(model_dir, hf_subfolder)
                chunk_rows = build_token_chunks(
                    selected_blocks,
                    tokenizer,
                    max_length=max_length,
                    overlap_tokens=overlap_tokens,
                )
                if not chunk_rows:
                    raise ValueError("Tidak ada chunk valid setelah preprocessing.")

                scores = predict_batch(
                    chunk_rows,
                    tokenizer,
                    model,
                    device,
                    ai_label_id=ai_label_id,
                    max_length=max_length,
                )
                score_ai = average_score(scores)
                final_label = "AI" if score_ai >= threshold else "Human"
                render_indobert_result(score_ai, threshold)

                stat_cols = st.columns(4)
                stat_cols[0].metric("Blok lolos", len(selected_blocks))
                stat_cols[1].metric("Dilewat", skipped_count)
                stat_cols[2].metric("Token-chunk", len(chunk_rows))
                stat_cols[3].metric("Threshold", f"{threshold:.2f}")
                st.caption(f"Device: {device} | AI label id: {ai_label_id} | id2label: {id2label}")

                chunk_df = pd.DataFrame(chunk_rows)
                chunk_df["prob_ai"] = scores
                chunk_df["prediksi"] = ["AI" if score >= threshold else "Human" for score in scores]
                score_column = "prob_ai"
                score_label = "Probabilitas AI"
                explanation_model_name = "IndoBERT"
                download_name = "hasil_deteksi_indobert_pipeline9.csv"
                display_columns = [
                    "block",
                    "chunk_index",
                    "chunk_count",
                    "token_count",
                    score_column,
                    "prediksi",
                    "preview",
                ]
                caption = "Output IndoBERT berupa probabilitas kelas AI."
            else:
                svm_model = load_svm_model(svm_model_path)
                chunk_rows = build_svm_chunks(selected_blocks)
                if not chunk_rows:
                    raise ValueError("Tidak ada chunk valid setelah preprocessing.")

                scores = predict_svm_batch(chunk_rows, svm_model)
                score_ai = average_score(scores)
                final_label = "AI" if score_ai >= svm_threshold else "Human"
                render_svm_result(score_ai, svm_threshold)

                stat_cols = st.columns(4)
                stat_cols[0].metric("Blok lolos", len(selected_blocks))
                stat_cols[1].metric("Dilewat", skipped_count)
                stat_cols[2].metric("Chunk teks", len(chunk_rows))
                stat_cols[3].metric("Batas SVM", f"{svm_threshold:.2f}")
                st.caption("Model: TF-IDF + LinearSVC")

                chunk_df = pd.DataFrame(chunk_rows)
                chunk_df["svm_score_ai"] = scores
                chunk_df["prediksi"] = ["AI" if score >= svm_threshold else "Human" for score in scores]
                score_column = "svm_score_ai"
                score_label = "Skor SVM AI"
                explanation_model_name = "TF-IDF + SVM"
                download_name = "hasil_deteksi_svm_pipeline9.csv"
                display_columns = [
                    "block",
                    "chunk_index",
                    "chunk_count",
                    "word_count",
                    score_column,
                    "prediksi",
                    "preview",
                ]
                caption = "Output SVM berupa decision score, bukan probabilitas."

            chunk_df["preview"] = chunk_df["text"].str.slice(0, 190)

            render_score_explanation(
                model_name=explanation_model_name,
                score_label=score_label,
                final_score=score_ai,
                threshold=threshold if model_choice == "IndoBERT" else svm_threshold,
                selected_count=len(selected_blocks),
                skipped_count=skipped_count,
                chunk_count=len(chunk_rows),
                label=final_label,
            )

            st.caption(
                "Seleksi blok mengikuti preprocessing training. "
                "Blok berstatus dilewat tidak masuk proses prediksi, tetapi tetap ditampilkan."
            )
            st.dataframe(
                selection_df[selection_columns],
                hide_index=True,
                width="stretch",
                column_config={
                    "preview": st.column_config.TextColumn("preview", width="large"),
                    "alasan": st.column_config.TextColumn("alasan", width="large"),
                },
                row_height=86,
            )

            st.caption(caption)
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

            selection_csv_bytes = selection_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download seleksi blok",
                data=selection_csv_bytes,
                file_name="hasil_seleksi_blok_pipeline9.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Gagal menjalankan deteksi: {exc}")

st.caption("Output model digunakan sebagai indikator probabilistik dan tetap perlu dibaca sesuai konteks penggunaan.")
