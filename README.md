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

The repository is configured for:
`https://github.com/maheshchary1234/Titanic-Survival-Prediction-Using-Streamlit.git`

To stage and push your commits:
```bash
git add .
git commit -m "Build and optimize Titanic survival prediction Streamlit app"
git push origin main
```

**Important:** Both `model.joblib` and `requirements.txt` are included in the repository so Streamlit Community Cloud can load the model immediately upon deployment.

## 4. Deploy on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) using your GitHub account (`maheshchary1234`).
2. Click **Create app** (or **New app**).
3. Select:
   - **Repository**: `maheshchary1234/Titanic-Survival-Prediction-Using-Streamlit`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy**.
5. Streamlit will install dependencies from `requirements.txt` and launch the application.
6. Copy the public deployment URL (e.g., `https://titanic-survival-prediction-using-streamlit.streamlit.app/`).
7. Test the link on your desktop and phone.

## 5. What to submit to InternCircle

According to the Project 05 learner guide, submit:
1. **Public deployed Streamlit app URL**: (e.g. `https://titanic-survival-prediction-using-streamlit.streamlit.app/`)
2. **GitHub repository URL**: `https://github.com/maheshchary1234/Titanic-Survival-Prediction-Using-Streamlit`
3. **Screenshot of a successful prediction**: Showing inputs and the estimated chance of survival.

## 6. Submission template

> **Project 05 Submission: Streamlit Interactive ML Web App**
>
> I developed and deployed an interactive Titanic Survival Prediction web application using Streamlit and my Decision Tree classifier trained on the historical Titanic dataset. The application loads the serialized model using `joblib` with `@st.cache_resource` for fast performance, validates user inputs (Ticket Class, Sex, Age, and Fare), and displays real-time prediction metrics including estimated survival probability and class outcome. Friendly error handling is implemented if files are missing.
>
> - **Deployed App URL**: [PASTE YOUR STREAMLIT APP LINK HERE]
> - **GitHub Repository**: https://github.com/maheshchary1234/Titanic-Survival-Prediction-Using-Streamlit
> - **Screenshot**: [Attach screenshot of prediction]

## Educational-use note

This is a learning demonstration based on historical Titanic passenger data. Predictions are not advice and should not be treated as reliable real-world risk estimates.
