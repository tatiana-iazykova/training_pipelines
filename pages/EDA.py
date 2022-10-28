import pandas as pd
import streamlit as st
from utils.constants import EDA_NAME, MINIMAL_NUMBER_OF_OBSERVATIONS, THRESHOLD, MIN_CLASS_NUMBER, ANNOTATION_THRESHOLD, BASE_CHECK_THRESHOLD
from utils.data_utils import (analyse_data_annotation, clean_duplicates,
                            clean_relevant_duplicates, return_text_and_targets)
from core.Dataset import PandasDataset
from utils.eda_utils import (check_value_counts,
                           compute_percentage_of_suitable_data,
                           render_pie_chart)


st.set_page_config(
    page_title=EDA_NAME
)
st.sidebar.header(EDA_NAME)

st.header(EDA_NAME)

score = v_c = na_check = 0

df = st.file_uploader(label="Upload your file", type=["csv", "xls", "xlsx", "tsv"])
if df is not None:
    df = PandasDataset(df)
    df = df.data
    st.dataframe(data=df.head())

    with st.form("Information about data"):
        text_columns = st.multiselect(
            label="Select text columns", options=df.columns)

        target_column = st.selectbox(
            label="Select target column. There can be only one target column per session",
            options=df.columns
        )

        accepted = True

        if target_column in text_columns:
            accepted = False
            st.error("Target column can't be in text columns")

        submitted = st.form_submit_button("Submit")

    if submitted and accepted:

        full_length, cnt_duplicates, df = clean_duplicates(df)

        relevant_length, cnt_relevant_duplicates, df = clean_relevant_duplicates(
            df=df,
            text_columns=text_columns,
            target_column=target_column
        )

        df, v_c = check_value_counts(
            df=df, target_column=target_column, threshold=THRESHOLD)

        if df[target_column].nunique() < MIN_CLASS_NUMBER:
            st.warning(
                f"Your data has less than {MIN_CLASS_NUMBER} classes eligible for modeling. We cannot proceed with this data :(")
            st.session_state["features"] = None
            st.session_state["targets"] = None
            st.session_state["data"] = None
        else:
            na_check = compute_percentage_of_suitable_data(
                df=df, target_column=target_column, full_length=full_length)
            res = pd.DataFrame(
                {
                    "text_columns": ", ".join(text_columns),
                    "target_column": target_column,
                    "percentage of duplicated data": f"{(cnt_duplicates/full_length) * 100:.2f}%",
                    "percentage of duplicated data in relevant columns": f"{(cnt_relevant_duplicates/relevant_length) * 100:.2f}%",
                    f"percentage of target classes, that have more observations than {THRESHOLD}": f"{v_c}%",
                    "percentage of suitable labels": f"{na_check}%",
                    "number of relevant observations": len(df),
                },
                index=[0]
            ).T

            res.columns = ['stats']
            st.session_state["base_data_statistics"] = res
            st.dataframe(res.style.format(precision=2))

            if len(df) < MINIMAL_NUMBER_OF_OBSERVATIONS:
                st.warning(f"WARNING! Number of observations in you data ({len(df)}) is less "
                           f"than minimal number of observations ({MINIMAL_NUMBER_OF_OBSERVATIONS})")

            X, y = return_text_and_targets(
                df=df, text_columns=text_columns, target_column=target_column)

            score, report = analyse_data_annotation(
                X=X, y=y, threshold=THRESHOLD)

            pie_chart = render_pie_chart(df=df, column_name=target_column)
            st.pyplot(pie_chart)

            st.session_state["data_quality_result"] = report
            st.session_state["data_quality_score"] = score

            potential_corrupt_score = 1 - score
            num_corrupt = round(len(y) * potential_corrupt_score)

            st.write(
                f"Data Quality: {score * 100 :.2f}. There may be potential issues with {potential_corrupt_score * 100:.2f}% "
                f"or {num_corrupt} out of {len(y)} examples"
            )
            st.dataframe(report.style.format(precision=2))

            if score > ANNOTATION_THRESHOLD and na_check > BASE_CHECK_THRESHOLD \
                and v_c > BASE_CHECK_THRESHOLD and len(df) > MINIMAL_NUMBER_OF_OBSERVATIONS:

                st.text("Data is good")
                st.session_state["features"] = X
                st.session_state["targets"] = y
                st.session_state["data"] = df
            else:
                st.text("Hi there, bitch. Your data is shit")
                st.session_state["features"] = X
                st.session_state["targets"] = y
                st.session_state["data"] = df
                st.session_state["trash_data"] = True
