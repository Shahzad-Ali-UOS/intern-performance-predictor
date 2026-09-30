import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Intern Performance & Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom header styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-header">🎓 Intern Performance & Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered early triage engine to identify high-performing and at-risk interns based on telemetry.</div>', unsafe_allow_html=True)

# 1. Load Serialized Model Artifacts
model_path = os.path.join("models", "best_model.pkl")

@st.cache_resource
def load_pipeline():
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

artifacts = load_pipeline()

if artifacts is None:
    st.error("⚠️ Model artifact not found! Please run `python train.py` first to generate `models/best_model.pkl`.")
    st.stop()

model = artifacts['model']
model_name = artifacts['model_name']
metrics = artifacts['metrics']

# Display Model Provenance in Sidebar
st.sidebar.header("⚙️ Model Metadata")
st.sidebar.success(f"**Active Model:** {model_name}")

r2_val = metrics.get('R2') or metrics.get('r2', 0.0)
mae_val = metrics.get('MAE') or metrics.get('mae', 0.0)
rmse_val = metrics.get('RMSE') or metrics.get('rmse', 0.0)

st.sidebar.markdown(f"- **R² Score:** `{r2_val:.4f}`")
st.sidebar.markdown(f"- **MAE:** `{mae_val:.4f}`")
st.sidebar.markdown(f"- **RMSE:** `{rmse_val:.4f}`")
st.sidebar.markdown("---")
st.sidebar.caption("Trained on 80/20 train-test split using simulated Internee.pk operational metrics.")

# 2. Main Dashboard Layout
col_inputs, col_results = st.columns([1.1, 1.2], gap="large")

with col_inputs:
    st.subheader("📋 Input Intern Telemetry")
    st.write("Adjust candidate metrics to evaluate predicted performance in real time:")

    task_comp = st.slider(
        "Task Completion Rate (%)",
        min_value=0.0, max_value=100.0, value=75.0, step=1.0,
        help="Percentage of assigned portal tasks marked completed."
    )

    turnaround = st.slider(
        "Average Turnaround Time (Days)",
        min_value=1.0, max_value=14.0, value=3.5, step=0.5,
        help="Average turnaround time to submit assigned assignments."
    )

    attendance = st.slider(
        "Attendance Rate (%)",
        min_value=0.0, max_value=100.0, value=80.0, step=1.0,
        help="Attendance records for meetings and LMS portal activity."
    )

    mentor_feedback = st.slider(
        "Mentor Feedback Rating (1.0 - 5.0)",
        min_value=1.0, max_value=5.0, value=3.8, step=0.1,
        help="Evaluation score assigned by project mentor/supervisor."
    )

    peer_collab = st.slider(
        "Peer Collaboration Score (1.0 - 5.0)",
        min_value=1.0, max_value=5.0, value=3.5, step=0.1,
        help="Score reflecting peer interaction, standups, and team contributions."
    )

with col_results:
    st.subheader("🎯 Performance Forecast & Triage")

    # Construct input payload
    input_payload = pd.DataFrame([{
        'task_completion_rate': task_comp,
        'avg_turnaround_days': turnaround,
        'attendance_rate': attendance,
        'mentor_feedback_rating': mentor_feedback,
        'peer_collaboration_score': peer_collab
    }])

    # Generate inference
    raw_prediction = float(model.predict(input_payload)[0])
    score = np.clip(raw_prediction, 0.0, 100.0)

    # Operational triage logic
    if score >= 80.0:
        tier_title = "Likely to Excel"
        tier_color = "#16A34A"  # Green
        status_badge = "🌟 EXCEL TRACK (HIGH PERFORMER)"
        action_plan = (
            "**Recommended Action:** High probability of completion with distinction. "
            "Candidate is a strong match for recommendation letters, certificate honors, "
            "and potential team-lead or referral opportunities."
        )
    elif score >= 60.0:
        tier_title = "On Track"
        tier_color = "#D97706"  # Orange/Amber
        status_badge = "⚖️ CONSISTENT TRACK (SATISFACTORY)"
        action_plan = (
            "**Recommended Action:** Steady progress meeting core requirements. "
            "Encourage maintaining task submission frequency and maintaining meeting consistency."
        )
    else:
        tier_title = "Likely to Struggle"
        tier_color = "#DC2626"  # Red
        status_badge = "⚠️ AT-RISK TRACK (INTERVENTION NEEDED)"
        action_plan = (
            "**Recommended Action:** Candidate shows high attrition or incomplete submission risk. "
            "Flag for early mentor intervention, a 1-on-1 progress sync, and deadline reminders."
        )

    # Score Cards
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Predicted Score", value=f"{score:.1f} / 100")
    with res_col2:
        st.markdown("**Status Category:**")
        st.markdown(f"<span style='color:{tier_color}; font-weight:700; font-size:1.1rem;'>{status_badge}</span>", unsafe_allow_html=True)

    st.progress(score / 100.0)

    # Actionable Recommendation Box
    st.markdown("---")
    st.markdown("#### 📌 Operations & Mentorship Guidance")
    if score >= 80.0:
        st.success(action_plan)
    elif score >= 60.0:
        st.warning(action_plan)
    else:
        st.error(action_plan)

    # Summary table of input parameters
    st.markdown("#### 🔍 Telemetry Snapshot")
    st.dataframe(input_payload.T.rename(columns={0: "Input Value"}), use_container_width=True)

st.markdown("---")
st.caption("Internee.pk Machine Learning Task Submission • Developed with Streamlit & XGBoost")