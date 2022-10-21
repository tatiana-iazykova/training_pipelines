import streamlit as st
import shutil
import os
import joblib
import json


def generate_requirements():
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


def create_app(model, log, path_to_template):

    if os.path.exists("temp/"):
        shutil.rmtree("temp/")
        
    shutil.copytree(src=path_to_template, dst='temp/')
    joblib.dump(model, "temp/model.joblib")
    requirements = generate_requirements()
    with open("temp/requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    with open("temp/log.json", "w") as f:
        json.dump(log, f, indent=4, ensure_ascii=False)
    
    shutil.make_archive("application", "zip", "temp/")
    shutil.rmtree("temp/")