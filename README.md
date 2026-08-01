# Deteksi Teks AI Jurnal Indonesia

Streamlit app untuk mendeteksi teks hasil Generative AI pada paper jurnal berbahasa Indonesia.

## Model

- IndoBERT default: `Tomath02/proto4`, subfolder `best_model`
- Baseline lokal: `models/svm/tfidf_svm_pipeline.joblib`

Repo Hugging Face `Tomath02/proto4` perlu berisi file model IndoBERT di root repo atau subfolder `best_model`:

- `config.json`
- `model.safetensors` atau `pytorch_model.bin`
- `tokenizer.json`
- `tokenizer_config.json`

## Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Kalau model IndoBERT belum tersedia di Hugging Face, jalankan dengan model lokal:

```bash
$env:LOCAL_INDOBERT_MODEL_DIR="path\to\best_model"
streamlit run app.py
```

## Konfigurasi Env

- `HF_MODEL_REPO`: repo model Hugging Face, default `Tomath02/proto4`
- `HF_MODEL_SUBFOLDER`: subfolder model Hugging Face, default `best_model`; kosongkan jika file model berada di root repo
- `LOCAL_INDOBERT_MODEL_DIR`: path model lokal untuk testing
- `SVM_MODEL_PATH`: path file baseline SVM
- `INDOBERT_THRESHOLD`: default `0.5`
- `SVM_THRESHOLD`: default `0.0`
