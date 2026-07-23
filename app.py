from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "models" / "predict_loan_possibility_model.pkl"

EDUCATION_LEVELS = {
    "SSC": 1,
    "12TH": 2,
    "GRADUATE": 3,
    "UNDER GRADUATE": 3,
    "POST-GRADUATE": 4,
    "OTHERS": 1,
    "PROFESSIONAL": 3,
}
PRODUCTS = ["AL", "CC", "ConsumerLoan", "HL", "PL", "others"]
PREDICTION_DETAILS = {
    0: ("P1", "Suitable for approval", "The model places this application in the strongest approval segment.", "#13795b"),
    1: ("P2", "Approval with caution", "The application is broadly acceptable, but complete the normal lender checks.", "#9a6700"),
    2: ("P3", "Risky to approve", "The application shows elevated risk and should receive additional review.", "#c2410c"),
    3: ("P4", "High risk to approve", "The application is in the highest-risk segment identified by the model.", "#b42318"),
}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def build_model_input(values: dict, model) -> pd.DataFrame:
    row = values.copy()
    row["EDUCATION"] = EDUCATION_LEVELS[row["EDUCATION"]]
    categorical_values = {
        "MARITALSTATUS": (row.pop("MARITALSTATUS"), ["Married", "Single"]),
        "first_prod_enq2": (row.pop("first_prod_enq2"), PRODUCTS),
        "last_prod_enq2": (row.pop("last_prod_enq2"), PRODUCTS),
        "GENDER": (row.pop("GENDER"), ["F", "M"]),
    }
    for prefix, (value, choices) in categorical_values.items():
        for choice in choices:
            row[f"{prefix}_{choice}"] = int(value == choice)
    input_frame = pd.DataFrame([row])
    expected_features = list(getattr(model, "feature_names_in_", input_frame.columns))
    return input_frame.reindex(columns=expected_features, fill_value=0)


def number_field(label: str, key: str, *, minimum=0.0, maximum=None, default=0.0, step=1.0, integer=False):
    if integer:
        numeric_maximum = None if maximum is None else int(maximum)
        return st.number_input(
            label,
            min_value=int(minimum),
            max_value=numeric_maximum,
            value=int(default),
            step=int(step),
            key=key,
        )

    numeric_maximum = None if maximum is None else float(maximum)
    return st.number_input(
        label,
        min_value=float(minimum),
        max_value=numeric_maximum,
        value=float(default),
        step=float(step),
        key=key,
    )


def collect_application() -> dict:
    values = {}
    with st.form("loan_risk_form"):
        st.subheader("Applicant profile")
        profile_left, profile_right = st.columns(2)
        with profile_left:
            values["EDUCATION"] = st.selectbox("Education", list(EDUCATION_LEVELS), key="education")
            values["MARITALSTATUS"] = st.selectbox("Marital status", ["Married", "Single"], key="marital_status")
            values["GENDER"] = st.selectbox("Gender", ["F", "M"], key="gender")
        with profile_right:
            values["NETMONTHLYINCOME"] = number_field("Net monthly income", "income", minimum=0.0, default=50000.0, step=1000.0)
            values["Time_With_Curr_Empr"] = number_field("Time with current employer (months)", "employment_months", minimum=0.0, default=24.0)
            values["Age_Oldest_TL"] = number_field("Age of oldest credit line (months)", "oldest_credit", minimum=0.0, default=60.0)
            values["Age_Newest_TL"] = number_field("Age of newest credit line (months)", "newest_credit", minimum=0.0, default=12.0)

        st.subheader("Credit history")
        credit_left, credit_right = st.columns(2)
        with credit_left:
            for key, label, default in (
                ("pct_tl_open_L6M", "Open accounts ratio, last 6 months", 0.20),
                ("pct_tl_closed_L6M", "Closed accounts ratio, last 6 months", 0.10),
                ("pct_tl_closed_L12M", "Closed accounts ratio, last 12 months", 0.10),
            ):
                values[key] = number_field(label, key, minimum=0.0, maximum=1.0, default=default, step=0.01)
            for key, label, default in (
                ("Tot_TL_closed_L12M", "Accounts closed, last 12 months", 1),
                ("Tot_Missed_Pmnt", "Missed payments", 0),
                ("time_since_recent_payment", "Days since recent payment", 15),
                ("max_recent_level_of_deliq", "Maximum recent delinquency (days)", 0),
                ("recent_level_of_deliq", "Current delinquency level (days)", 0),
            ):
                values[key] = number_field(label, key, minimum=0, default=default, integer=True)
        with credit_right:
            for key, label, default in (
                ("num_deliq_6_12mts", "Delinquencies in last 6-12 months", 0),
                ("num_times_60p_dpd", "Times with 60+ DPD", 0),
                ("num_std_12mts", "Standard accounts, last 12 months", 12),
                ("num_sub", "Sub-standard accounts", 0),
                ("num_sub_6mts", "Sub-standard accounts, last 6 months", 0),
                ("num_sub_12mts", "Sub-standard accounts, last 12 months", 0),
                ("num_dbt", "Doubtful accounts", 0),
                ("num_dbt_12mts", "Doubtful accounts, last 12 months", 0),
                ("num_lss", "Loss accounts", 0),
            ):
                values[key] = number_field(label, key, minimum=0, default=default, integer=True)

        st.subheader("Credit exposure and enquiries")
        exposure_left, exposure_right = st.columns(2)
        with exposure_left:
            for key, label in (("CC_TL", "Credit card accounts"), ("Home_TL", "Home loan accounts"), ("PL_TL", "Personal loan accounts"), ("Secured_TL", "Secured accounts"), ("Unsecured_TL", "Unsecured accounts"), ("Other_TL", "Other accounts")):
                values[key] = number_field(label, key, minimum=0, default=0, integer=True)
        with exposure_right:
            for key, label, default in (
                ("CC_enq_L12m", "Credit card enquiries, last 12 months", 0),
                ("PL_enq_L12m", "Personal loan enquiries, last 12 months", 0),
                ("time_since_recent_enq", "Days since recent enquiry", 30),
                ("enq_L3m", "Enquiries, last 3 months", 0),
            ):
                values[key] = number_field(label, key, minimum=0, default=default, integer=True)
            for key, label in (("pct_PL_enq_L6m_of_ever", "Personal loan enquiry ratio"), ("pct_CC_enq_L6m_of_ever", "Credit card enquiry ratio")):
                values[key] = number_field(label, key, minimum=0.0, maximum=1.0, default=0.0, step=0.01)

        st.subheader("Product history")
        product_left, product_right = st.columns(2)
        with product_left:
            values["first_prod_enq2"] = st.selectbox("First product enquiry", PRODUCTS, key="first_product")
            values["last_prod_enq2"] = st.selectbox("Latest product enquiry", PRODUCTS, key="last_product")
            values["CC_Flag"] = int(st.checkbox("Has a credit card", key="cc_flag"))
        with product_right:
            values["PL_Flag"] = int(st.checkbox("Has a personal loan", key="pl_flag"))
            values["HL_Flag"] = int(st.checkbox("Has a home loan", key="hl_flag"))
            values["GL_Flag"] = int(st.checkbox("Has a group loan", key="gl_flag"))

        st.caption("All fields are used by the trained model. Review the entered information before submitting.")
        submitted = st.form_submit_button("Assess loan risk", type="primary", use_container_width=True)
    return values if submitted else {}


def main():
    st.set_page_config(page_title="Loan Risk Assessment", page_icon=":bar_chart:", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
        [data-testid="stMetricValue"] { color: #0f766e; }
        .result { padding: 1.25rem 1.5rem; border-left: 6px solid var(--result-color); background: #f7faf9; border-radius: 8px; }
        .result h2 { margin: 0; color: var(--result-color); }
        .result p { margin-bottom: 0; font-size: 1.05rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Loan Risk Assessment")
    st.write("Enter an applicant's profile and credit history to classify the application into the model's P1-P4 risk bands.")

    values = collect_application()
    if not values:
        st.info("Complete the application details above, then select **Assess loan risk**.")
        return

    try:
        model = load_model()
        model_input = build_model_input(values, model)
        prediction = int(model.predict(model_input)[0])
    except Exception as error:
        st.error(f"Unable to make a prediction: {error}")
        st.info("Install the project dependencies and ensure the trained model exists at models/predict_loan_possibility_model.pkl.")
        return

    grade, title, description, color = PREDICTION_DETAILS.get(prediction, PREDICTION_DETAILS[3])
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(model_input)[0].max())

    st.divider()
    st.subheader("Assessment result")
    st.markdown(
        f'<div class="result" style="--result-color: {color}"><h2>{grade} · {title}</h2><p>{description}</p></div>',
        unsafe_allow_html=True,
    )
    metric_left, metric_right = st.columns(2)
    metric_left.metric("Risk band", grade)
    metric_right.metric("Model confidence", f"{confidence:.1%}" if confidence is not None else "Unavailable")
    st.warning("This is a decision-support prototype, not a final lending decision. Apply lending policy, affordability checks, fairness review, and regulatory controls before approval.")


if __name__ == "__main__":
    main()