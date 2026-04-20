# Emotion Detection from Face - Final Project

This is a beginner-friendly AI final project that detects facial emotions from an image.
It uses a Convolutional Neural Network (CNN) trained on a folder-based facial expression dataset.

## Project goal
Build a model that predicts emotions such as:
- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

## Dataset options
You can use either of these Kaggle datasets:
1. FER-2013: https://www.kaggle.com/datasets/msambare/fer2013
2. Face Expression Recognition Dataset: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

The easiest option for this project is the FER-2013 dataset because it is already organized in folders like:

```text
data/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  test/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
```

## Folder structure

```text
emotion-detection-final-project/
│
├── app.py
├── predict_image.py
├── train.py
├── requirements.txt
├── .gitignore
├── README.md
├── data/
├── models/
├── notebooks/
├── assets/
└── src/
    ├── config.py
    ├── data_utils.py
    └── model_utils.py
```

## Step-by-step setup in VS Code

### 1. Create the project folder
Create a folder named:

```text
emotion-detection-final-project
```

Open that folder in VS Code.

### 2. Create a virtual environment
Open the VS Code terminal and run:

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset
Download the Kaggle dataset and place the folders inside `data/` so that you have:

```text
data/train/...
data/test/...
```

### 5. Train the model
```bash
python train.py
```

After training, these files will appear in `models/`:
- `emotion_model.keras`
- `class_names.txt`
- `training_plot.png`

### 6. Test one image
```bash
python predict_image.py path/to/your/image.jpg
```

### 7. Run the demo app
```bash
streamlit run app.py
```

## GitHub steps

### Option A: Upload using GitHub website
1. Create a new repository on GitHub.
2. Name it something like `emotion-detection-final-project`.
3. Click **Add file** > **Upload files**.
4. Upload all project files except `.venv` and dataset images.
5. Commit the files.

### Option B: Upload with Git commands
Run these commands inside your project folder:

```bash
git init
git add .
git commit -m "Initial commit - emotion detection final project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/emotion-detection-final-project.git
git push -u origin main
```

## What to show in your presentation
1. Problem statement: detect human emotion from face images.
2. Dataset overview: number of classes and sample images.
3. Preprocessing: grayscale, resize to 48x48.
4. Model: simple CNN.
5. Results: test accuracy and training plot.
6. Live demo: upload one image in Streamlit.
7. Limitations: lighting, face angle, low image quality.
8. Future improvement: webcam input or stronger model.

## Beginner tips
- Do not put the dataset inside GitHub because it is too large.
- Keep `.venv` and trained model files out of GitHub unless your teacher asks.
- If the terminal says your environment is not activated, activate it again before running commands.
- If `python` does not work, try `py` on Windows or `python3` on Mac/Linux.

## Possible project title
**Facial Emotion Detection Using Deep Learning**
