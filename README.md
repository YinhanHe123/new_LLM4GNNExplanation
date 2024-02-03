# LLM4GNNExplanation
Using LLMs to guide GNN Counterfactual Explanation

## How to run
Add an OpenAI api key in utils/pre_defined.py and run
```bash
python main.py
```

## Directory Layout
```bash
./data                          # Datasets used in the paper            
|---- AIDS/
|---- BBBP/ 
|---- Mutagenicity/ 
|---- SIDER/ 
|---- Tox21/ 
./exp_results                   # Results used in the paper

./model                         
|----gnn.py                     # Contains the model and functions for ground truth GNN (GT-GNN)
|----explainer.py               # Contains the model and functions for the LLM autoencoder
|----model.utils.py             # Contains any functions related to models
|----model_evaluation.py        # Code to evaluate models
./saved_models                  # Weights of trained models
./utils
|----data_load.py               # Used to preprocess the datasets
|----datasets.py                # Contains Dataset class
|----pre_defined.py             # Predefined string formatting for querying OpenAI API and the API key
|----smiles.py                  # Functions to convert graph data to and from SMILES representation
```

## Environments
The dependencies needed to run the code can be installed using
```shell
pip install -r requirements.txt
```

## Experiments
### AIDS
```bash
python main.py
```
### BBBP

### Mutagenicity

### SIDER
```bash
python -m pdb main.py -d SIDER -glr 0.01 -gwd 0.0005
test_loss: 1.2117120948704807 | test_acc : 0.6438746438746439
```
### Tox21