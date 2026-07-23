import joblib
import pandas as pd

MODEL_PATH = "../../models/predict_loan_possibility_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained loan possibility prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model

def predict_loan_possibility(input_data):
    """
    Predict loan possibility for new applications.

    Parameters
    ----------
    input_data: dict

    Returns
    -------
    pd.DataFrame with predicted loan category
    """

    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Loan_Possibility'] = model.predict(input_df.drop(['Approved_Flag'], axis=1)).round()
    return input_df

if __name__ == "__main__":
    
    # Example inference run
    sample_data = {
        "PROSPECTID": [10001, 10002, 10003, 10004],
        "pct_tl_open_L6M": [0.15, 0.40, 0.00, 0.65],
        "pct_tl_closed_L6M": [0.05, 0.20, 0.10, 0.00],
        "Tot_TL_closed_L12M": [1, 3, 0, 2],
        "pct_tl_closed_L12M": [0.10, 0.50, 0.00, 0.33],
        "Tot_Missed_Pmnt": [0, 2, 5, 0],
        "CC_TL": [2, 1, 0, 4],
        "Home_TL": [1, 0, 0, 1],
        "PL_TL": [0, 2, 1, 0],
        "Secured_TL": [1, 0, 0, 2],
        "Unsecured_TL": [2, 3, 1, 4],
        "Other_TL": [0, 0, 1, 0],
        "Age_Oldest_TL": [84, 36, 12, 120],
        "Age_Newest_TL": [12, 4, 8, 2],
        "time_since_recent_payment": [15, 30, 90, 5],
        "max_recent_level_of_deliq": [0, 30, 90, 0],
        "num_deliq_6_12mts": [0, 1, 3, 0],
        "num_times_60p_dpd": [0, 0, 2, 0],
        "num_std_12mts": [12, 8, 2, 12],
        "num_sub": [0, 0, 1, 0],
        "num_sub_6mts": [0, 0, 0, 0],
        "num_sub_12mts": [0, 0, 1, 0],
        "num_dbt": [0, 0, 0, 0],
        "num_dbt_12mts": [0, 0, 0, 0],
        "num_lss": [0, 0, 1, 0],
        "recent_level_of_deliq": [0, 30, 90, 0],
        "CC_enq_L12m": [1, 3, 0, 4],
        "PL_enq_L12m": [0, 2, 1, 0],
        "time_since_recent_enq": [45, 10, 120, 2],
        "enq_L3m": [0, 2, 0, 3],
        "EDUCATION": [3, 2, 1, 4],
        "NETMONTHLYINCOME": [75000, 45000, 18000, 150000],
        "Time_With_Curr_Empr": [48, 12, 6, 96],
        "CC_Flag": [1, 1, 0, 1],
        "PL_Flag": [0, 1, 1, 0],
        "pct_PL_enq_L6m_of_ever": [0.0, 0.8, 1.0, 0.0],
        "pct_CC_enq_L6m_of_ever": [0.2, 0.5, 0.0, 0.9],
        "HL_Flag": [1, 0, 0, 1],
        "GL_Flag": [0, 0, 0, 0],
        "Approved_Flag": ["P1", "P2", "P3", "P0"],
        "MARITALSTATUS_Married": [1, 0, 0, 1],
        "MARITALSTATUS_Single": [0, 1, 1, 0],
        "first_prod_enq2_AL": [0, 0, 1, 0],
        "first_prod_enq2_CC": [1, 0, 0, 1],
        "first_prod_enq2_ConsumerLoan": [0, 1, 0, 0],
        "first_prod_enq2_HL": [0, 0, 0, 0],
        "first_prod_enq2_PL": [0, 0, 0, 0],
        "first_prod_enq2_others": [0, 0, 0, 0],
        "last_prod_enq2_AL": [0, 0, 0, 0],
        "last_prod_enq2_CC": [0, 1, 0, 1],
        "last_prod_enq2_ConsumerLoan": [1, 0, 0, 0],
        "last_prod_enq2_HL": [0, 0, 0, 0],
        "last_prod_enq2_PL": [0, 0, 1, 0],
        "last_prod_enq2_others": [0, 0, 0, 0],
        "GENDER_F": [0, 1, 0, 1],
        "GENDER_M": [1, 0, 1, 0]
    }

    prediction = predict_loan_possibility(sample_data)
    print(prediction)