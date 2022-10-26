import streamlit as st
from utils.constants import TFIDF_NAME
from utils.home_utils import create_app
from pipelines.build_tfidf_logreg import build_tfidf_logreg
from pathlib import Path
import eli5
import numpy as np

st.set_page_config(
    page_title=TFIDF_NAME
)
st.sidebar.header(TFIDF_NAME)

st.header(TFIDF_NAME)

proceed = False

if "features" not in st.session_state.keys() and "targets" not in st.session_state.keys() and "data" not in st.session_state.keys():
    st.session_state["features"] = None
    st.session_state["targets"] = None
    st.session_state["data"] = None
    st.warning("There is no data yet")

else:
    X = st.session_state["features"]
    y = st.session_state["targets"]
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

            st.write(f"Train scores: {np.mean(train_scores)}")
            st.write(f"Overall scores: {np.mean(overall_scores)}")

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

            if st.checkbox("Explain model"):
                r = eli5.explain_weights_df(
                        model['logreg'], 
                        top=10, 
                        feature_names=model['vectoriser'].get_feature_names_out()
                        )
                st.dataframe(r)

            if st.checkbox("Save model"):

                path_to_template = Path(__file__).parent.parent / "streamlit_templates" / "text_classification"
                path_to_core = Path(__file__).parent.parent / "core"
                
                create_app(
                    model=model, 
                    log=log, 
                    path_to_template=path_to_template.as_posix(),
                    path_to_core=path_to_core.as_posix()
                    )

                with open("application.zip", "rb") as fp:
                    btn = st.download_button(
                        label="Download ZIP",
                        data=fp,
                        file_name="application.zip",
                        mime="application/zip"
                    )