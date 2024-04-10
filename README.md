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

## Complementary Experiment Results

### (1) (R1, R2, R5) Model efficiency and scalability evaluation.

Below are runtime metrics for training for 100 epochs and generating counterfactual graphs for AIDS and ClinTox. The metrics are in seconds. All of these experiments were conducted on a compute cluster on a single Nvidia RTX 2080 Ti and two 18-core Intel processors.
|         | AIDS | ClinTox | 
|---------|------|---------|
| GNN_Explainer    | 454.66  | 37.75  |
| CF_GNNExplainer |  1272.21 | 95.46  |   
| RegExplainer | 1385.05  | 99.91  |   
| CLEAR        | 908.70  | 398.91  |

Here, we see that GNN_Explainer is the fastest baseline with CLEAR and RegExplainer trading places as the slowest baselines between AIDS and ClinTox.

### (2) (R1) More case studies should be done.

### (3) (R1) Parameter sensitivity of $\alpha$ and $\beta$.

|         | 5 | 2 | 1 | 0.5 | 0.2 |
|---------|---|---|---|-----|-----|
| AIDS    |   |   |   |     |     |
| Clintox |   |   |   |     |     |

### (4) (R2) LLM can be hard to generalize to graphs other than molecules. 

### (5) (R2) Measure the performance of counterfactual explanation with solely GPT-4, i.e., SMILES in, SMILES out, calculate accuracy.

### (6) (R3, R5) Use different LLMs for prompt provider, for example GPT-MolBERTa/ChemGPT.
