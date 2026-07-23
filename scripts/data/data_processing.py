# import dependencies
import logging
import os

import pandas as pd
import sqlite3

from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create logs directory if it doesn't exist
os.makedirs("../../logs", exist_ok=True)

logging.basicConfig(
    filename="../../logs/data_processing.log",
    level=logging.DEBUG,
    force=True,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode="a",
)

def drop_columns_and_rows_with_high_missing_values(df, threshold=10000, null_value=-99999):
    columns_to_drop = []
    df_dropped = df.copy()
    
    for column in df.columns:
        if df.loc[df[column] == null_value].shape[0] > threshold:
            columns_to_drop.append(column)

    # Drop the identified columns
    if columns_to_drop:
        logging.info(f"Dropping columns: {columns_to_drop}")
        df_dropped = df.drop(columns=columns_to_drop)
    
    for column in df_dropped.columns:
        df_dropped = df_dropped.loc[df_dropped[column] != null_value]
    
    return df_dropped


def drop_categorical_by_chi2(df, target_column, p_value_threshold=0.05):
    """
    Performs a Chi-square hypothesis test between all string-type categorical 
    columns and a target column. Drops columns where the p-value is below the threshold.
    """
    columns_to_drop = []
    
    # Identify categorical columns (pandas usually stores strings as 'object' or 'string')
    # We exclude the target column so it doesn't test against itself
    categorical_cols = [
        col for col in df.columns 
        if col != target_column and df[col].dtype in ['str']
    ]

    logging.info(f"Categorical columns to test: {categorical_cols}")
    
    for col in categorical_cols:
        # Create the contingency table
        contingency_table = pd.crosstab(df[col], df[target_column])
        
        # Run the test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        logging.info(f"Tested '{col}': p-value = {p_value}")
        
        # Check against threshold
        if p_value > p_value_threshold:
            columns_to_drop.append(col)
            logging.info(f" -> Dropping '{col}' (p-value > {p_value_threshold})")
            
    # Drop the flagged columns
    df_dropped = df.drop(columns=columns_to_drop)
    
    return df_dropped


def filter_numerical_features(df, target_column='Approved_Flag', vif_threshold=6.0, anova_threshold=0.05):
    """
    Evaluates numerical columns by sequentially removing those with high multicollinearity (VIF > threshold),
    then testing the remaining features against a categorical target using ANOVA.
    """
    # 1. Identify numeric columns (excluding the target if it happens to be numeric)
    numeric_columns = [
        col for col in df.columns 
        if df[col].dtype in ['int64', 'float64']
    ]
    
    # 2. VIF Sequential Check
    logging.info("--- Starting VIF Check ---")
    vif_data = df[numeric_columns].copy()
    columns_after_vif = []
    column_index = 0
    
    for col in numeric_columns:
        # Calculate VIF for the column currently at 'column_index'
        vif_value = variance_inflation_factor(vif_data.values, column_index)
        
        if vif_value <= vif_threshold:
            columns_after_vif.append(col)
            logging.info(f"Kept: {col} (VIF: {vif_value})")
            column_index += 1
        else:
            logging.info(f"Dropped: {col} (VIF: {vif_value} > {vif_threshold})")
            vif_data = vif_data.drop(columns=[col])
            
    # 3. ANOVA Test
    logging.info("\n--- Starting ANOVA Check ---")
    final_numerical_columns = []
    
    for col in columns_after_vif:
        # Dynamically separate the column into lists based on the target column's classes
        # This replaces the hardcoded zip(a, b) logic for P1, P2, P3, P4
        category_groups = [group.values for name, group in df.groupby(target_column)[col]]
        
        # Unpack the groups into f_oneway
        f_statistic, p_value = f_oneway(*category_groups)
        
        if p_value < anova_threshold:
            final_numerical_columns.append(col)
            logging.info(f"Kept: {col} (p-value: {p_value:.4f})")
        else:
            logging.info(f"Dropped: {col} (p-value: {p_value:.4f} >= {anova_threshold})")
            
    # 4. Drop the rejected columns and return the new DataFrame
    # Find all numeric columns that didn't make the final cut
    rejected_columns = set(numeric_columns) - set(final_numerical_columns)
    
    df_filtered = df.drop(columns=list(rejected_columns))
    logging.info(f"\nTotal columns dropped: {len(rejected_columns)}")
    
    return df_filtered


def encode_categorical_features(df):
    """
    Applies custom ordinal encoding to the EDUCATION column and 
    one-hot encoding to specified nominal columns.
    """
    # Create a copy to avoid modifying the original dataframe warning
    df_encoded = df.copy()
    
    # 1. Ordinal Encoding for EDUCATION (as defined in image_404de4.png)
    education_mapping = {
        'SSC': 1,
        '12TH': 2,
        'GRADUATE': 3,
        'UNDER GRADUATE': 3,
        'POST-GRADUATE': 4,
        'OTHERS': 1,
        'PROFESSIONAL': 3
    }
    
    if 'EDUCATION' in df_encoded.columns:
        # Map the string values to integers
        df_encoded['EDUCATION'] = df_encoded['EDUCATION'].map(education_mapping)
        
        # Convert the column to integer type as requested in the image
        df_encoded['EDUCATION'] = df_encoded['EDUCATION'].astype(int)
        
    # 2. One-Hot Encoding for nominal columns
    dummy_columns = ['MARITALSTATUS', 'first_prod_enq2', 'last_prod_enq2', 'GENDER']
    
    # Filter only columns that exist in the dataframe to prevent KeyError
    existing_dummy_cols = [col for col in dummy_columns if col in df_encoded.columns]
    
    if existing_dummy_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=existing_dummy_cols, dtype='uint8')
        
    return df_encoded


import pandas as pd

def push_dataframe_to_database(df, conn, table_name='credit_risk_load_data'):
    """
    Creates a strict schema table in the database and loads the DataFrame into it.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to push.
    conn: The database connection or SQLAlchemy engine.
    table_name (str): The name of the target table.
    """
    
    create_table_query = f"""
    CREATE TABLE {table_name} (
        pct_tl_open_L6M FLOAT NOT NULL,
        pct_tl_closed_L6M FLOAT NOT NULL,
        Tot_TL_closed_L12M INT NOT NULL,
        pct_tl_closed_L12M FLOAT NOT NULL,
        Tot_Missed_Pmnt INT NOT NULL,
        CC_TL INT NOT NULL,
        Home_TL INT NOT NULL,
        PL_TL INT NOT NULL,
        Secured_TL INT NOT NULL,
        Unsecured_TL INT NOT NULL,
        Other_TL INT NOT NULL,
        Age_Oldest_TL INT NOT NULL,
        Age_Newest_TL INT NOT NULL,
        time_since_recent_payment INT NOT NULL,
        max_recent_level_of_deliq INT NOT NULL,
        num_deliq_6_12mts INT NOT NULL,
        num_times_60p_dpd INT NOT NULL,
        num_std_12mts INT NOT NULL,
        num_sub INT NOT NULL,
        num_sub_6mts INT NOT NULL,
        num_sub_12mts INT NOT NULL,
        num_dbt INT NOT NULL,
        num_dbt_12mts INT NOT NULL,
        num_lss INT NOT NULL,
        recent_level_of_deliq INT NOT NULL,
        CC_enq_L12m INT NOT NULL,
        PL_enq_L12m INT NOT NULL,
        time_since_recent_enq INT NOT NULL,
        enq_L3m INT NOT NULL,
        EDUCATION INT NOT NULL,
        NETMONTHLYINCOME BIGINT NOT NULL,
        Time_With_Curr_Empr INT NOT NULL,
        CC_Flag INT NOT NULL,
        PL_Flag INT NOT NULL,
        pct_PL_enq_L6m_of_ever FLOAT NOT NULL,
        pct_CC_enq_L6m_of_ever FLOAT NOT NULL,
        HL_Flag INT NOT NULL,
        GL_Flag INT NOT NULL,
        Approved_Flag VARCHAR(50) NOT NULL,
        MARITALSTATUS_Married TINYINT NOT NULL,
        MARITALSTATUS_Single TINYINT NOT NULL,
        first_prod_enq2_AL TINYINT NOT NULL,
        first_prod_enq2_CC TINYINT NOT NULL,
        first_prod_enq2_ConsumerLoan TINYINT NOT NULL,
        first_prod_enq2_HL TINYINT NOT NULL,
        first_prod_enq2_PL TINYINT NOT NULL,
        first_prod_enq2_others TINYINT NOT NULL,
        last_prod_enq2_AL TINYINT NOT NULL,
        last_prod_enq2_CC TINYINT NOT NULL,
        last_prod_enq2_ConsumerLoan TINYINT NOT NULL,
        last_prod_enq2_HL TINYINT NOT NULL,
        last_prod_enq2_PL TINYINT NOT NULL,
        last_prod_enq2_others TINYINT NOT NULL,
        GENDER_F TINYINT NOT NULL,
        GENDER_M TINYINT NOT NULL
    );
    """
    
    try:
        cursor = conn.cursor()
        
        # 1. Drop the table if it already exists to prevent duplication/errors on re-runs
        logging.info(f"Dropping table '{table_name}' if it exists...")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # 2. Create the table with strict constraints
        logging.info(f"Creating table '{table_name}'...")
        cursor.execute(create_table_query)
        
        # Commit the DDL changes if using a standard DBAPI connection
        if hasattr(conn, 'commit'):
            conn.commit()
            
        # 3. Push the data using 'append' to respect the newly created schema
        logging.info(f"Pushing {len(df)} rows to the database...")
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        logging.info("Data load successful!")
        
    except Exception as e:
        logging.info(f"An error occurred: {e}")
        # Rollback in case of failure
        if hasattr(conn, 'rollback'):
            conn.rollback()

if __name__ == '__main__':
    logging.info("Initiating database connection.")
    conn = sqlite3.connect('credit_modelling.db')
    
    logging.info("Fetching tables from database")
    internal_data = pd.read_sql_query("SELECT * FROM internal_product", conn)
    cibil_data = pd.read_sql_query("SELECT * FROM cibil_score", conn)
    
    logging.info("Droping columns containg missing values")
    internal_data = drop_columns_and_rows_with_high_missing_values(internal_data)
    cibil_data = drop_columns_and_rows_with_high_missing_values(cibil_data)
    
    logging.info("Combining two tables accross common column")
    combined_dataframe = pd.merge(internal_data, cibil_data, how='inner', left_on=['PROSPECTID'], right_on=['PROSPECTID'])
    combined_dataframe.drop(columns=['PROSPECTID'], inplace=True)
    
    logging.info("Performing Hypothesis testing")
    combined_dataframe = drop_categorical_by_chi2(combined_dataframe, target_column='Approved_Flag')
    combined_dataframe = filter_numerical_features(combined_dataframe)

    logging.info("Performing label encoding for categorical columns")
    encoded_combined_dataframe = encode_categorical_features(combined_dataframe)

    logging.info("Pushing new dataframe to database")
    push_dataframe_to_database(encoded_combined_dataframe, conn=conn)