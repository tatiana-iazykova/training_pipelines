#!/bin/bash
python3 -m venv venv 
source venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
streamlit run interface.py 