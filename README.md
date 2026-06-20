# Cat Translator AI

Cat Translator AI is a small full-stack project that classifies short cat vocalizations into likely intent labels. It does not translate exact cat words. The app predicts broad sound-pattern intent such as `Food`, `Isolation`, or `Brushing`.

The project uses only free/open-source tooling:

- Training: Google Colab, TensorFlow, TensorFlow Hub YAMNet, librosa
- Backend: FastAPI, TensorFlow, TensorFlow Hub, FFmpeg, librosa
- Frontend: Angular standalone components, HttpClient, browser MediaRecorder API

## Repository Layout

```text
cat-translator-ai/
  training/
    cat_training_colab.ipynb
    label_map.json
    requirements-training.txt
  backend/
    main.py
    model_loader.py
    audio_utils.py
    label_map.json
    requirements.txt
    test_client.py
  frontend/
    src/app/
```

## Important Model Note

`cat_model.keras` is intentionally not committed. It is a generated training artifact and can be large. Generate it from `training/cat_training_colab.ipynb`, then place it locally at:

```text
backend/cat_model.keras
```

The backend will not make predictions until that local model file exists.

## Backend Setup

The backend decodes browser-recorded audio with FFmpeg. MediaRecorder often sends WebM/Opus, so FFmpeg converts uploads to mono 16 kHz WAV before librosa loads them.

Install FFmpeg:

```bash
# Windows
winget install Gyan.FFmpeg

# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg
```

Verify FFmpeg:

```bash
ffmpeg -version
```

Create and activate a Python environment from `backend/`:

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn main:app --reload
```

Health check:

```text
GET http://localhost:8000/health
```

Prediction endpoint:

```text
POST http://localhost:8000/translate
multipart/form-data field: file
```

You can test with:

```bash
python test_client.py sample.wav
```

## Frontend Setup

From `frontend/`:

```bash
npm install
npm start
```

Open:

```text
http://localhost:4200
```

The Angular app asks for microphone permission, records exactly 3 seconds, sends `cat-recording.webm` to the backend, and displays the predicted intent, confidence, and message.

## Training Notebook Workflow

Use Google Colab for training:

1. Open `training/cat_training_colab.ipynb` in Colab.
2. Upload or mount `archive.zip`.
3. The notebook unzips the dataset and parses Kaggle-style filenames.
4. It builds `manifest.csv` with label, cat ID, breed, sex, owner, session, sample rate, and duration metadata.
5. It splits train/validation/test by `cat_id` to reduce leakage.
6. It loads audio with `librosa`, resamples to 16 kHz, and creates 3-second windows.
7. It extracts YAMNet embeddings with TensorFlow Hub.
8. It trains a small Keras classifier on top of those embeddings.
9. It reports accuracy, confusion matrix, classification report, and dataset balance.
10. It saves `cat_model.keras`, a timestamped model copy, `label_map.json`, `manifest.csv`, and the training history plot.

Copy the generated `cat_model.keras` and `label_map.json` into `backend/` for local inference. Do not commit `cat_model.keras`, datasets, manifests, or training zips.

## Safety Before First Push

The root `.gitignore` excludes:

- Python virtual environments
- `node_modules`
- Angular build output
- `.env` files
- `archive.zip`
- dataset folders
- generated manifests
- large model files such as `.keras`, `.h5`, `.onnx`, and `.tflite`

Before pushing, check:

```bash
git status --short
git check-ignore -v backend/cat_model.keras backend/manifest.csv backend/venv frontend/node_modules
```

If any dataset, virtual environment, `node_modules`, secret, or model artifact appears as staged/tracked, remove it from Git before pushing.

## Limitations

This is an intent classifier, not a literal cat-language translator. Predictions depend heavily on dataset quality, label quality, number of samples, recording conditions, and whether the model has seen similar cats and sounds during training. It is not a veterinary or behavioral diagnosis tool.
