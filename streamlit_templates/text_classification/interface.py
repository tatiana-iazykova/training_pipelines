import os
from io import BytesIO

import pandas as pd
import streamlit as st

from Model import Model


@st.cache(hash_funcs={Model: id})
def load():
    """
    load model
    """
    model = Model()
    return model


@st.cache
def to_excel(df: pd.DataFrame) -> bytes:
    """
    converts dataframe to an excel bytefile
    """
    outputs = BytesIO()
    writer = pd.ExcelWriter(outputs, engine="openpyxl")
    df.to_excel(writer, index=False, sheet_name="ModelPredictions")
    writer.save()
    processed_data = outputs.getvalue()
    return processed_data


if __name__ == '__main__':
    model = load()

    uploaded_file = st.file_uploader("Choose file for inference", type=[".xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        text_columns = st.multiselect(
            label="Select text columns", options=df.columns)
        target_column = st.text_input(
            label="Choose name for target columns", value='target')

        if st.checkbox("All set"):
            file_name = os.path.splitext(uploaded_file.name)[0] + "_done.xlsx"
            df = model.predict(
                df=df, 
                text_columns=text_columns,
                target_column=target_column
                )
            result = to_excel(df)
            st.download_button(
                label="Download model resuls",
                data=result,
                file_name=file_name,
                mime="text/xlsx"
            )
