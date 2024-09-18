## Requirements
* Python 3.12
* Docker
 

## Preprocessing
1. Clone Directory:
```
git clone git@github.com:niladi/kdd-seminar-code.git
git submodule init 
git submodule update
```
2. Set Up Python Env
```
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$PWD/src
``` 
3. Create GraphDB
a. Run Compose File
```
docker compose up -d 
```
b. Set Up Graph under: `http://localhost:7200`
4. Download Clit Results and Datasets `https://bwsyncandshare.kit.edu/s/cnDLM4gKMLed8JA?path=%2`
5. Extract the Results and Datasets and set Paths in `src/clit_recommender/__init__.py`
6. Create Med Mentions NIF
```
python src/clit_recommender/data/med_mentions.py
```
7. Upload Datasets turtle nif files over the `Import` Tab in `http://localhost:7200`
8. Create Full Results NIF
```
python src/clit_recommender/data/create_complete_nif.py
```
9. Upload CLiT Result turtle nif files over the `Import` Tab in `http://localhost:7200`
10. Create Eval Test Split of Datasets and Upload it over the `Import` Tab in `http://localhost:7200`

## Analysis and Experiments
Checkout the `experiments.ipynb` Jupyter Notebook.



## TODO 
* Preprocessing Doku/Pipeline

