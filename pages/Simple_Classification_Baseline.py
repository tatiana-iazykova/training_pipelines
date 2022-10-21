import streamlit as st
from constants import TFIDF_NAME
from home_utils import create_app
from models.build_tfidf_logreg import build_tfidf_logreg
from pathlib import Path

st.set_page_config(
    page_title=TFIDF_NAME
)
st.sidebar.header(TFIDF_NAME)

st.header(TFIDF_NAME)

proceed = False

if "X" not in st.session_state.keys() and "y" not in st.session_state.keys() and "data" not in st.session_state.keys():
    st.session_state["X"] = None
    st.session_state["y"] = None
    st.session_state["data"] = None
    st.warning("There is no data yet")

else:
    X = st.session_state["X"]
    y = st.session_state["y"]
    df = st.session_state["data"]

    if X is None and y is  None and df is None:
        st.warning("There is no data yet")
    
    else:
        st.write("Your data")
        st.dataframe(df.head())

        if "trash_data" in st.session_state.keys():
            proceed = st.checkbox("I accept that my data is trash and I take the consequences of it")

        if proceed or "trash_data" not in st.session_state.keys():
            train_scores, overall_scores, df_classification_report, model = build_tfidf_logreg(X=X, y=y)

            st.write(f"Train scores: {train_scores}")
            st.write(f"Overall scores: {overall_scores}")

            st.write("Metrics")
            st.dataframe(df_classification_report.style.format(precision=2))

            log = {
                "data": {
                    "base_data_statistics": st.session_state["base_data_statistics"].to_dict("index"),
                    "data_quality_result": st.session_state["data_quality_result"].to_dict("index"),
                    "data_quality_score": st.session_state["data_quality_score"]
                },
                "model": {
                    "train cross_val_score": list(train_scores),
                    "full data cross_val_score": list(overall_scores),
                    "metrics": df_classification_report.to_dict("index")
                }
            }

            path_to_template = Path(__file__).parent.parent / "streamlit_templates" / "text_classification"
            create_app(model=model, log=log, path_to_template=path_to_template.as_posix())

            with open("application.zip", "rb") as fp:
                btn = st.download_button(
                    label="Download ZIP",
                    data=fp,
                    file_name="application.zip",
                    mime="application/zip"
                )