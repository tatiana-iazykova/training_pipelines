import streamlit as st
import pandas as pd
from eda.eda_utils import check_value_counts, compute_percentage_of_suitable_data
from eda.data_utils import return_text_and_targets, analyse_data_annotation


def check_state():
    proceed = st.checkbox("Proceed?", key="checkbx")
    return proceed

st.set_page_config(
    page_title="Data Modeling"
)
st.sidebar.header("Data Modeling")

st.header("Data Modeling")

df = st.file_uploader(label="Upload your Excel file", type=["csv"])
if df is not None:  
    df = pd.read_csv(df)
    st.dataframe(data=df.head())

    with st.form("Information about data"):
        text_columns = st.text_input(label="Text column, if multiple separate them with semicolumn (;) delimeter")
        text_columns = text_columns.split(';')
        target_column = st.text_input(label="There should be only one target column per session")

        submitted = st.form_submit_button("Submit")

    if submitted:
        threshold = 2
        v_c = check_value_counts(df=df, target_column=target_column, threshold=threshold)
        na_check = compute_percentage_of_suitable_data(df=df, target_column=target_column)
        res = pd.DataFrame(
                {
                    "text_columns": text_columns,
                    "target_column": target_column,
                    "percentage of data suitable data": f"{na_check}%",
                    f"percentage of target_classes, that have more observations than {threshold}": f"{v_c}%"
                }
            ).T

        res.columns = ['stats']
        st.dataframe(res)
        
        X, y = return_text_and_targets(df, text_columns=text_columns, target_column=target_column)

        score, report = analyse_data_annotation(X, y, threshold)

        potential_corrupt_score =  1 - score
        num_corrupt = round(len(y) * potential_corrupt_score)

        st.write(
            f"Data Quality: {score * 100}. There may be potential issues with {potential_corrupt_score * 100}% " \
                f"or {num_corrupt} out of {len(y)} examples"
                )

        st.dataframe(report)
            
if check_state():
    
    st.text("Hi there, bitch")
