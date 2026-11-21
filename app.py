"""Project 05 | InternCircle Data Science & AI
Interactive Titanic Survival Prediction Web App
Deployed using Streamlit Community Cloud
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"
if not MODEL_PATH.exists() and Path("model.joblib").exists():
    MODEL_PATH = Path("model.joblib")

FEATURES = ["pclass", "sex", "age", "fare"]

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Model Loading with Caching
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained model artifact once and cache it in memory."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The model file 'model.joblib' is missing. "
            "Please run `python train_model.py` to train and save the model first."
        )
    
    artifact = joblib.load(MODEL_PATH)
    
    # Support both dictionary artifact (with metadata) and plain estimator
    if isinstance(artifact, dict):
        if artifact.get("features") != FEATURES:
            raise ValueError(
                f"Saved model feature order {artifact.get('features')} "
                f"does not match expected app features {FEATURES}."
            )
        return artifact["model"], artifact
    return artifact, {}

# ---------------------------------------------------------
# Sidebar Information & Documentation
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/320px-RMS_Titanic_3.jpg",
        caption="RMS Titanic (1912)",
        use_container_width=True
    )
    st.header("Project Details")
    st.markdown(
        """
        **Project 05 | Data Science & AI**  
        *Streamlit Interactive ML Web App*  
        **InternCircle Internship Program**
        """
    )
    
    st.divider()
    st.subheader("Model Information")
    st.markdown(
        """
        - **Algorithm**: Decision Tree Classifier
        - **Max Depth**: 4 (pruned to prevent overfitting)
        - **Test Accuracy**: ~79.7%
        - **Features Used**:
          1. `pclass` (1 = 1st, 2 = 2nd, 3 = 3rd)
          2. `sex` (0 = Male, 1 = Female)
          3. `age` (Passenger age in years)
          4. `fare` (Ticket fare in British pounds)
        """
    )
    
    st.divider()
    st.caption("Developed by Mahesh Chary • Ready for Streamlit Community Cloud")

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title("🚢 Titanic Survival Predictor")
st.markdown(
    "Would a passenger with your selected characteristics have survived the "
    "historic RMS Titanic voyage? Enter passenger details below and test the "
    "machine learning classification model in real time."
)

st.info(
    "ℹ️ **Educational Demonstration**: This model was trained on historical passenger "
    "records from the 1912 Titanic tragedy. It is strictly an educational exercise, "
    "not real-world advice.",
    icon="ℹ️"
)

# ---------------------------------------------------------
# Preset Profiles for Quick Testing
# ---------------------------------------------------------
st.markdown("##### ⚡ Quick Example Profiles")
preset_cols = st.columns(3)

if "pclass_val" not in st.session_state:
    st.session_state.pclass_val = 3
if "sex_val" not in st.session_state:
    st.session_state.sex_val = "Female"
if "age_val" not in st.session_state:
    st.session_state.age_val = 28
if "fare_val" not in st.session_state:
    st.session_state.fare_val = 15.0

with preset_cols[0]:
    if st.button("👑 1st Class Female", use_container_width=True):
        st.session_state.pclass_val = 1
        st.session_state.sex_val = "Female"
        st.session_state.age_val = 29
        st.session_state.fare_val = 100.0

with preset_cols[1]:
    if st.button("💼 2nd Class Male", use_container_width=True):
        st.session_state.pclass_val = 2
        st.session_state.sex_val = "Male"
        st.session_state.age_val = 30
        st.session_state.fare_val = 26.0

with preset_cols[2]:
    if st.button("🎒 3rd Class Male", use_container_width=True):
        st.session_state.pclass_val = 3
        st.session_state.sex_val = "Male"
        st.session_state.age_val = 22
        st.session_state.fare_val = 7.75

# ---------------------------------------------------------
# Passenger Inputs (Mobile-friendly 2-Column Layout)
# ---------------------------------------------------------
st.subheader("Passenger Information")

col_left, col_right = st.columns(2)

with col_left:
    pclass_options = [1, 2, 3]
    pclass_labels = {1: "1st Class (Upper)", 2: "2nd Class (Middle)", 3: "3rd Class (Lower)"}
    pclass = st.selectbox(
        "Ticket Class",
        options=pclass_options,
        index=pclass_options.index(st.session_state.pclass_val),
        format_func=lambda x: pclass_labels[x],
        help="Passenger cabin / ticket booking class"
    )
    
    sex_label = st.radio(
        "Sex",
        options=["Female", "Male"],
        index=0 if st.session_state.sex_val == "Female" else 1,
        horizontal=True,
        help="Biological sex of passenger as recorded in manifest"
    )

with col_right:
    age = st.slider(
        "Age (years)",
        min_value=1,
        max_value=80,
        value=int(st.session_state.age_val),
        step=1,
        help="Age of passenger (1 - 80)"
    )
    
    fare = st.slider(
        "Fare Paid (£)",
        min_value=0.0,
        max_value=500.0,
        value=float(st.session_state.fare_val),
        step=1.0,
        help="Passenger fare in 1912 British pounds"
    )

# Encode sex: female = 1, male = 0 (matching training pipeline)
sex_encoded = 1 if sex_label == "Female" else 0

# Construct one-row dataframe with exact feature ordering
input_df = pd.DataFrame(
    [[int(pclass), int(sex_encoded), float(age), float(fare)]],
    columns=FEATURES
)

# Display summary card
with st.expander("📋 Review Selected Passenger Attributes", expanded=False):
    summary_df = pd.DataFrame([{
        "Ticket Class": pclass_labels[pclass],
        "Sex": sex_label,
        "Age": f"{age} years",
        "Fare Paid": f"£{fare:.2f}"
    }])
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# Prediction & Inference Section
# ---------------------------------------------------------
st.markdown("---")
predict_clicked = st.button("🔮 Predict Survival Probability", type="primary", use_container_width=True)

if predict_clicked:
    try:
        model, metadata = load_model()
        
        # Validations
        if list(input_df.columns) != FEATURES:
            st.error("Error: Input feature order does not match model expectations.")
            st.stop()
            
        if input_df.isnull().any().any():
            st.error("Error: Please provide valid values for all passenger features.")
            st.stop()
            
        if not hasattr(model, "predict_proba"):
            st.error("Error: Loaded model does not support probability prediction.")
            st.stop()
        
        # Calculate probabilities
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        
        if 1 not in classes:
            st.error("Error: The model classes do not contain the positive survival class (1).")
            st.stop()
            
        survival_prob = float(probabilities[classes.index(1)])
        prediction = int(model.predict(input_df)[0])
        
        st.subheader("Prediction Result")
        
        metric_col1, metric_col2 = st.columns([1, 1])
        
        with metric_col1:
            st.metric(
                label="Estimated Chance of Survival",
                value=f"{survival_prob:.0%}"
            )
            
        with metric_col2:
            if prediction == 1:
                st.success("### ✅ Outcome: Survived", icon="✅")
            else:
                st.error("### ❌ Outcome: Did Not Survive", icon="❌")
        
        # Survival Gauge / Progress Bar
        st.markdown("**Survival Probability Spectrum**")
        st.progress(survival_prob)
        st.caption(f"Estimated likelihood: {survival_prob:.1%} survival vs {1 - survival_prob:.1%} perishing.")
        
        # Contextual explanation
        st.caption(
            "A learning exercise. It describes a historical dataset and is not advice."
        )
        st.info(
            f"**Model Insight**: A {pclass_labels[pclass]} {sex_label.lower()} aged {age} "
            f"who paid £{fare:.2f} has a predicted outcome of "
            f"**{'Survived' if prediction == 1 else 'Did Not Survive'}** "
            f"({survival_prob:.1%} probability) according to the Decision Tree model."
        )
        
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        st.info("To generate `model.joblib`, open a terminal and execute:")
        st.code("python train_model.py", language="bash")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        st.caption("Please verify that `model.joblib` is intact and up to date.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888888; font-size: 0.85rem;'>"
    "Project 05 | InternCircle Data Science & AI Internship • "
    "Streamlit & Scikit-Learn Interactive App"
    "</div>",
    unsafe_allow_html=True
)
