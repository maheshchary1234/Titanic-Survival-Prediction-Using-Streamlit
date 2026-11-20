# Project 05 – Streamlit Interactive ML Web App
## Titanic Survival Prediction

This app wraps the Project 04 Decision Tree classifier in an interactive Streamlit interface. Users choose passenger class, sex, age and fare, then receive a predicted class and estimated survival probability.

## Project files

- `app.py` – Streamlit user interface and prediction logic
- `train_model.py` – downloads the Titanic dataset, trains the Project 04 model and saves `model.joblib`
- `requirements.txt` – Python dependencies
- `model.joblib` – generated model artifact (create it by running the training script)

## 1. Setup locally

Use Python 3.10 or newer. Open a terminal in this project folder.

### Windows
```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If `py` is unavailable, use `python` instead. If activation is blocked in PowerShell, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python train_model.py
streamlit run app.py
```

The training script needs internet access to download the public Titanic CSV. It saves the fitted model as `model.joblib` in this folder. Keep this file alongside `app.py`.

## 2. Test the app

1. Open the local URL printed by Streamlit (usually `http://localhost:8501`).
2. Choose passenger details in the sidebar.
3. Click **Predict survival probability**.
4. Confirm the probability and predicted class appear.
5. Capture a screenshot showing the inputs and prediction.

The app loads the model with `@st.cache_resource`. The feature order is fixed as `pclass`, `sex`, `age`, `fare`; sex is encoded female = 1 and male = 0, matching training.

## 3. Push to GitHub

Create a new public repository on GitHub, for example `titanic-survival-streamlit`.

In the project folder:
```bash
git init
git add app.py train_model.py requirements.txt README.md .gitignore model.joblib
git commit -m "Build Titanic survival prediction Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/titanic-survival-streamlit.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. If Git says a remote named `origin` already exists, use:
```bash
git remote set-url origin https://github.com/YOUR-USERNAME/titanic-survival-streamlit.git
```

**Important:** Run `python train_model.py` before `git add` so `model.joblib` exists. Include `model.joblib` in the repository for this small educational project. Do not upload passwords, API keys or private data.

## 4. Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ using your GitHub account.
2. Choose **Create app**.
3. Select the repository, branch `main`, and main file path `app.py`.
4. Deploy the app.
5. Wait for installation and startup to finish.
6. Open the public app URL and test a prediction.
7. Test the URL on a phone or in a private/incognito browser window.

If deployment reports a missing model, confirm `model.joblib` is committed to the repository in the same folder as `app.py`. If package installation fails, check that `requirements.txt` is in the repository root.

## 5. What to submit to InternCircle

According to the Project 05 learner guide, submit:
- Public deployed Streamlit app URL
- GitHub repository URL
- Screenshot of a successful prediction

Make sure both links are accessible to anyone with the link. The screenshot should show the selected passenger inputs and the prediction result.

## 6. Submission notes

I developed an interactive Titanic Survival Prediction web application using Streamlit and my Project 04 Decision Tree classifier. The application loads a saved machine-learning model using joblib, accepts passenger class, sex, age and fare as inputs, and displays the predicted survival class and estimated probability. The model is cached for efficient loading, and the app includes a friendly message if the model file is missing. The project is version-controlled on GitHub and deployed using Streamlit Community Cloud.

App URL: [PASTE DEPLOYED STREAMLIT URL]
GitHub URL: [PASTE GITHUB REPOSITORY URL]
Prediction screenshot: [ATTACH SCREENSHOT]

## Educational-use note

This is a learning demonstration based on historical Titanic passenger data. Predictions are not advice and should not be treated as reliable real-world risk estimates.
"# Titanic-Survival-Prediction-Using-Streamlit" 
