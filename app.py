from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"
FEATURES = ["pclass", "sex", "age", "fare"]

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Enter passenger details to see the Decision Tree model's estimated "
    "survival probability."
)
st.info(
    "Educational demonstration using a historical Titanic dataset. "
    "It is not a real-world prediction or advice."
)

@st.cache_resource
def load_model():
    """Load the trained scikit-learn model once per app process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.joblib was not found. Run `python train_model.py` first."
        )
    artifact = joblib.load(MODEL_PATH)
    # train_model.py stores the estimator together with reproducibility metadata.
    if isinstance(artifact, dict):
        if artifact.get("features") != FEATURES:
            raise ValueError("Saved model feature order does not match the app.")
        return artifact["model"]
    # Also accept a plain estimator saved directly with joblib.
    return artifact

with st.sidebar:
    st.header("Passenger details")
    pclass = st.selectbox("Ticket class", options=[1, 2, 3], index=2,
                          help="1 = First class, 2 = Second class, 3 = Third class")
    sex_label = st.radio("Sex", options=["Female", "Male"], horizontal=True)
    age = st.slider("Age", min_value=1, max_value=80, value=30, step=1)
    fare = st.slider("Fare paid", min_value=0.0, max_value=500.0,
                     value=30.0, step=1.0)

st.subheader("Your selected inputs")
input_row = pd.DataFrame(
    [[int(pclass), 1 if sex_label == "Female" else 0, float(age), float(fare)]],
    columns=FEATURES
)
display_row = input_row.copy()
display_row["sex"] = sex_label
st.dataframe(display_row, hide_index=True, use_container_width=True)

if st.button("Predict survival probability", type="primary", use_container_width=True):
    try:
        model = load_model()
        if list(input_row.columns) != FEATURES:
            st.error("Input feature order is invalid. Please refresh the app.")
        elif input_row.isnull().any().any():
            st.error("Please complete all passenger inputs.")
        elif not hasattr(model, "predict_proba"):
            st.error("The saved model does not support probability predictions.")
        else:
            probabilities = model.predict_proba(input_row)[0]
            classes = list(model.classes_)
            if 1 not in classes:
                st.error("The saved model does not contain the survival class.")
            else:
                survival_probability = float(probabilities[classes.index(1)])
                predicted_class = int(model.predict(input_row)[0])
                st.metric(
                    "Estimated chance of survival",
                    f"{survival_probability:.1%}"
                )
                if predicted_class == 1:
                    st.success("Model prediction: Survived")
                else:
                    st.warning("Model prediction: Did not survive")
                st.caption(
                    "This probability is produced by a Decision Tree trained on "
                    "historical Titanic passenger records. It is an educational "
                    "example and should not be interpreted as a reliable real-world "
                    "risk estimate."
                )
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python train_model.py", language="bash")
    except (ValueError, OSError, EOFError) as exc:
        st.error("The model could not be loaded or the prediction could not be made.")
        st.caption(f"Technical detail: {exc}")
    except Exception as exc:
        st.error("Something went wrong while generating the prediction.")
        st.caption(f"Technical detail: {exc}")

st.divider()
st.caption("Project 05 | InternCircle Data Science & AI")
