# training_pipelines

Streamlit app for easy classification modeling
It's so easy that you should only have to clone this repo, install requirements and run the streamlit app. We'll handle the rest :)



0. Clone the repository, install dependencies and run the app

Clone the repo and go to that folder in the terminal

```bash
git clone https://github.com/tatiana-iazykova/training_pipelines.git
cd training_pipelines
```

Optional (create virtual environment):

```bash
python3 -m venv venv 
```
    
For Unix-like OS:

```bash
source venv/bin/activate
```

For Windows:

```bash
source venv/Scripts/activate
```

Upgrade pip, wheel and setuptools, install dependencies and the run streamlit app

```bash
pip install -U pip wheel setuptools && pip install -r requirements.txt
streamlit run Home_Page.py
```

1. Head to EDA page to get you data assessed
2. Head to the page with the model you want to try. For now you can choose only the baseline model.
3. After you train you model you can donwload it neatly packed in a zip archive with all the necessary
     things for running you own streamlit inference application
4. Unzip the archive as a folder, open the folder directory in terminal and run 
    
    1.  ```bash
        chmod 777 run.sh
        ```
    2.  ```bash
        ./run.sh
        ```

5. Or you can do all the things specified in run.sh manually

    1. create virtual environment: 

        ```bash
        python3 -m venv venv
        ```

    2. activate the created virtual environment
    
        For Unix-like systems:
        
        ```bash
            source venv/bin/activate
        ```
        
        For Windows:
        
        ```bash
            source venv/Scripts/activate
        ```
        
    3. upgrade pip, wheel and setuptools
    
        ```bash
        pip install -U pip wheel setuptools
        ```
    
    4. install the requirements
    
        ```bash
        pip install -r requirements.txt
        ```
    
    5. run streamlit app
    
        ```bash
        streamlit run interface.py
        ```
