import streamlit as st
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier

st.title("Credit Risk Analysis")

# Numeric input
age = st.number_input("Enter your age:", min_value=0, max_value=100, step=1)
income = st.number_input("Enter your income:")
emp_length = st.number_input("Enter your length of employment:", format="%.1f")
loan_amount = st.number_input("Enter your loan amount:")
int_rate = st.number_input("Enter your interest rate:", format="%.2f")
per_income = st.number_input("Enter your loan percent income:", format="%.2f")
cred_history = st.number_input("Enter your credit history length:")

# Another text input
home_ownership_type = st.selectbox("Pick the home ownership type", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
loan_intent = st.selectbox("Pick the loan intent", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT',
       'DEBTCONSOLIDATION'])
loan_grade = st.selectbox("Pick the loan grade", ['A','B','C','D','E','F','G'])
default = st.radio("Pick one", ["Y", "N"])
click = st.button("Submit")

if click:
    new_data = pd.DataFrame([{'person_age':age, 
                         'person_income':income, 
                         'person_emp_length':emp_length,
                         'loan_amnt':loan_amount,
                         'loan_int_rate':int_rate,
                         'loan_percent_income':per_income,
                         'cb_person_cred_hist_length': cred_history,
                         'person_home_ownership': home_ownership_type, 
                         'loan_intent': loan_intent, 
                         'loan_grade': loan_grade, 
                         'cb_person_default_on_file': default}])

    all_columns = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
       'loan_int_rate', 'loan_status', 'loan_percent_income',
       'cb_person_cred_hist_length', 'person_home_ownership_OTHER',
       'person_home_ownership_OWN', 'person_home_ownership_RENT',
       'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
       'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
       'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 'loan_grade_E',
       'loan_grade_F', 'loan_grade_G', 'cb_person_default_on_file_Y']
    
    new_data_encoded = pd.get_dummies(new_data, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], dtype=int)
    new_data_encoded = new_data_encoded.reindex(columns=all_columns, fill_value=0)
    print(new_data_encoded)
    new_data_encoded.drop('loan_status', axis=1, inplace=True)
    # Load the saved model
    model = pickle.load(open('credit_risk_model.pkl', 'rb'))
    result = model.predict(new_data_encoded)
    print(result)
    if result == 1:
        st.text("Loan Approved")
    else:
        st.text("Loan Rejected")


