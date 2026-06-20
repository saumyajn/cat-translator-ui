# Cat Translator AI

A cat voice to human intent translator AI project skeleton.

This project is designed to classify short cat vocalizations into broad intent labels such as food request, isolation distress, greeting, play, or discomfort. It does **not** claim to translate exact cat words or sentences. The goal is practical intent classification from audio patterns using free and open-source tools.

## Project Status

This repository currently contains a placeholder project skeleton only. The full model training pipeline, API inference code, and Angular UI behavior are intentionally not implemented yet.

## Folder Structure

```text
cat-translator-ai/
  README.md
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
  frontend/
    package.json
    src/
      app/
        app.component.ts
        app.component.html
        app.component.css
        cat-translator.service.ts
```

## What This App Should Do

The finished app will let a user record or upload a cat sound, send the audio to a FastAPI backend, and receive a predicted intent label with confidence scores.

Example supported intent labels:

- `food_request`: The cat may be asking for food or treats.
- `isolation_distress`: The cat may be calling because it is separated, lonely, or seeking attention.
- `happy_greeting`: The cat may be greeting or showing friendly engagement.
- `play_request`: The cat may want play or stimulation.
- `discomfort`: The cat may be stressed, uncomfortable, or in pain.
- `unknown`: The audio does not confidently match a known class.

These are behavioral intent categories, not exact translations.

## Tech Stack

All tools are free and open source.

- Training: Google Colab
- Audio loading and preprocessing: librosa
- Embeddings: TensorFlow Hub YAMNet
- Model training: TensorFlow / Keras
- Backend API: FastAPI
- Frontend: Angular
- Recording: Browser MediaRecorder API
- Browser audio decoding: FFmpeg
- Data format: WAV audio files
- Model artifact format: TensorFlow SavedModel or Keras `.keras` file

## Full Workflow

### 1. Collect Audio Data

Create a dataset of short cat vocalization recordings. Each file should contain a single clear cat sound when possible.

Recommended recording rules:

- Use WAV files when available.
- Keep recordings short, ideally 1 to 5 seconds.
- Avoid background music, TV, human speech, and loud household noise.
- Record multiple cats if possible to reduce overfitting to one animal.
- Capture different rooms, microphones, and distances.
- Do not include private conversations in training audio.

A simple naming convention can encode labels:

```text
F_001.wav
I_002.wav
H_003.wav
P_004.wav
D_005.wav
U_006.wav
```

Suggested filename prefixes:

- `F`: food_request
- `I`: isolation_distress
- `H`: happy_greeting
- `P`: play_request
- `D`: discomfort
- `U`: unknown

The same label mapping should be stored in both `training/label_map.json` and `backend/label_map.json`.

### 2. Train in Google Colab

The notebook at `training/cat_training_colab.ipynb` will eventually handle training.

Planned training steps:

1. Mount Google Drive or upload a zipped dataset.
2. Install training dependencies from `requirements-training.txt`.
3. Load `.wav` files using `librosa`.
4. Resample each file to 16,000 Hz, which is required by YAMNet.
5. Pad or trim audio clips to a consistent duration.
6. Pass audio through TensorFlow Hub YAMNet to extract embeddings.
7. Train a small classifier on top of the YAMNet embeddings.
8. Evaluate the classifier with a train/validation/test split.
9. Export the trained model artifact.
10. Save the final label map used during training.

YAMNet is useful here because it already understands many general audio features. Instead of training a large audio model from scratch, this project can train a smaller classifier using YAMNet embeddings.

### 3. Export the Model

The trained classifier should be exported from Colab into a backend-loadable format.

Possible artifact layout:

```text
backend/models/cat_intent_classifier.keras
backend/label_map.json
```

The backend should use the exact same label order as training. Changing label order after training will produce incorrect predictions.

### 4. Run the FastAPI Backend

The backend exposes:

```text
POST /translate
```

Expected behavior:

1. Receive an audio file from the frontend.
2. Validate file type and size.
3. Convert browser audio such as WebM/Opus to mono 16 kHz WAV with FFmpeg.
4. Load the converted WAV with `librosa`.
5. Pad or trim to the expected duration.
6. Extract YAMNet embeddings.
7. Run the trained classifier.
8. Return the predicted intent and confidence scores.

Example future response:

```json
{
  "predicted_label": "food_request",
  "confidence": 0.82,
  "scores": {
    "food_request": 0.82,
    "isolation_distress": 0.08,
    "happy_greeting": 0.04,
    "play_request": 0.03,
    "discomfort": 0.02,
    "unknown": 0.01
  },
  "disclaimer": "This is an intent classification, not an exact translation."
}
```

### 5. Build the Angular Frontend

The Angular app will eventually provide:

- A record button using the browser MediaRecorder API.
- A stop button to finish recording.
- A playback preview of the captured audio.
- A submit button to send the audio to FastAPI.
- A result area showing the predicted intent and confidence.
- A visible disclaimer that predictions are approximate intent classifications.

The frontend should not claim that the model can translate exact cat words. It should use careful wording such as:

- "Likely intent"
- "Possible meaning"
- "Confidence"
- "This is not a medical or behavioral diagnosis"

## Development Setup

### Training Environment

Use Google Colab for model training.

Install dependencies in Colab:

```bash
pip install -r training/requirements-training.txt
```

### Backend Environment

The backend uses FFmpeg to decode browser-recorded audio such as WebM/Opus from
the MediaRecorder API. FFmpeg must be installed and available on your PATH before
running the backend.

Install FFmpeg:

Windows with winget:

```bash
winget install Gyan.FFmpeg
```

Windows with Chocolatey:

```bash
choco install ffmpeg
```

macOS with Homebrew:

```bash
brew install ffmpeg
```

Linux with apt:

```bash
sudo apt update
sudo apt install ffmpeg
```

Verify installation:

```bash
ffmpeg -version
```

From the `backend` folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The backend will typically run at:

```text
http://127.0.0.1:8000
```

### Frontend Environment

From the `frontend` folder:

```bash
npm install
npm start
```

The frontend will typically run at:

```text
http://localhost:4200
```

## API Design

Prediction endpoint:

```text
POST /translate
Content-Type: multipart/form-data
Field: file
```

The frontend records with the browser MediaRecorder API, which commonly creates
`webm` audio. The backend saves the upload to a temporary file, converts it with
FFmpeg, and then loads the converted WAV with `librosa`:

```bash
ffmpeg -y -i input.webm -ac 1 -ar 16000 output.wav
```

Response shape:

```json
{
  "intent": "Food",
  "confidence": 0.87,
  "all_predictions": [
    { "label": "Food", "confidence": 0.87 },
    { "label": "Isolation", "confidence": 0.08 },
    { "label": "Brushing", "confidence": 0.05 }
  ],
  "message": "Your cat is probably asking for food.",
  "disclaimer": "This predicts likely intent only; it is not a literal cat-language translation."
}
```

## Model Limitations

This project should be honest about what it can and cannot do.

Limitations:

- Cat vocalization meaning depends heavily on context.
- The same sound can mean different things depending on posture, environment, history, and body language.
- A model trained on one cat may not generalize well to another cat.
- Background noise can strongly affect predictions.
- The app should not be used as a veterinary diagnostic tool.
- Signs of pain, distress, breathing trouble, or sudden behavior changes should be handled by a veterinarian.

## Privacy Notes

Audio can accidentally capture people, addresses, conversations, or other private information.

Recommended privacy approach:

- Keep raw training data local or in the user's private Google Drive.
- Do not upload user recordings to third-party services except the user's own training environment.
- Make the backend process audio locally during development.
- Delete temporary uploaded files after inference.

## Next Implementation Steps

1. Fill in the Colab notebook with dataset loading and YAMNet embedding extraction.
2. Train a baseline classifier on top of embeddings.
3. Export a model artifact and label map.
4. Implement backend audio preprocessing in `backend/audio_utils.py`.
5. Implement backend model loading in `backend/model_loader.py`.
6. Implement the FastAPI `/predict` route in `backend/main.py`.
7. Implement MediaRecorder recording in the Angular component.
8. Implement the Angular service call to the backend.
9. Add validation, confidence thresholds, and friendly unknown-state handling.
10. Test with real cat recordings and refine labels.

## License

Placeholder: choose an open-source license before publishing.
