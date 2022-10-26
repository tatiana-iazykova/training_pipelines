import streamlit as st
import shutil
import os
import joblib
import json
from typing import List, Dict, Any
import sklearn


def generate_requirements() -> List[str]:
    """
    creates a set of requirements crucial for running an app with the model trained with this pipeline

    :return list of librarires and their versions

    Example

    >>> generate_requirements()
    ["scikit-learn==1.1.2", "streamlit==1.13.0", "openpyxl==3.0.10", "pandas==1.5.1"]

    """
    requirements = []

    import sklearn
    sklearn_version = "scikit-learn=="+sklearn.__version__

    streamlit_version = 'streamlit=='+st.__version__ 

    import openpyxl
    openpyxl_version = "openpyxl==" + openpyxl.__version__

    import pandas
    pandas_version = "pandas==" + pandas.__version__

    requirements.append(sklearn_version)
    requirements.append(streamlit_version)
    requirements.append(openpyxl_version)
    requirements.append(pandas_version)   

    return requirements


def create_app(model: sklearn.pipeline.Pipeline, log: Dict[str, Any], path_to_template: str, path_to_core: str) -> None:
    """
    assembles zip archive with working streamlit application with your model

    :param model: trained sklearn pipeline that can be saved with the help of joblib
    :param log: data analysis result as well as model metrics
    :param path_to_template: path to the folder where relevant streamlit template and utility files are stored
    :param path_to_core: path to the folder where core files are stored
    """

    temporary_dir = "temp/"

    if os.path.exists(temporary_dir):
        shutil.rmtree(temporary_dir)
        
    shutil.copytree(src=path_to_template, dst=temporary_dir)
    shutil.copytree(src=path_to_core, dst=temporary_dir)
    joblib.dump(model, f"{temporary_dir}model.joblib")
    requirements = generate_requirements()
    with open(f"{temporary_dir}requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    with open(f"{temporary_dir}log.json", "w") as f:
        json.dump(log, f, indent=4, ensure_ascii=False)
    
    shutil.make_archive("application", "zip", temporary_dir)
    shutil.rmtree(temporary_dir)
