# TODO: does not work
conda create -n pd python=3.8
conda activate pd
pip install --upgrade pip
pip install -r requirements.txt
mkdir data data/raw data/intermediate data/final
python -m ipykernel install --user --name=pd
