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
#### Efficiency
Below are runtime metrics for training for 100 epochs and generating counterfactual graphs for AIDS and ClinTox. The metrics are in seconds. All of the baseline experiments were conducted on a compute cluster on a single Nvidia RTX 2080 Ti and two 18-core Intel processors. We conducted an efficiency study of our method (LLM-GCE) on a compute server equipped with 6 RTX A6000 GPUs. The reported result represents the average of five separate experiments.
|         | AIDS | ClinTox | 
|---------|------|---------|
| GNN_Explainer    | 454.66  | 37.75  |
| CF_GNNExplainer |  1272.21 | 95.46  |   
| RegExplainer | 1385.05  | 99.91  |   
| CLEAR        | 908.70  | 398.91  |
| LLM-GCE      | 7925.60 | 3441.08 |

Here, we see that GNN_Explainer is the fastest baseline with CLEAR and RegExplainer trading places as the slowest baselines between AIDS and ClinTox.

#### Scalability
We further evaluate our method on the Peptides-func dataset from [LRGB](https://github.com/vijaydwivedi75/lrgb). The Peptides-func dataset has an average node count of 150.94, significantly higher than the average node numbers in the five datasets evaluated in our paper, which range from 15.69 to 27.74. To adapt to a binary classification task, we only consider the fifth label type (antiviral) from the original multi-labeled dataset.
|         | Proximity | Validity | 
|---------|------|---------|
| GNN_Explainer    |   |   |
| CF_GNNExplainer |   |   |   
| RegExplainer |   |   |   
| CLEAR        |   |   |
| LLM-GCE      |  |  |

### (2) (R1) More case studies should be done.

We strengthen our claims regarding LLM-GCE's more feasible counterfactuals with another example from BBBP. Below, we compare CF_GNNExplainer with LLM-GCE on molecule 273 from BBP.
|![original_mol](https://github.com/YinhanHe123/new_LLM4GNNExplanation/blob/main/original_mol.png) | ![our_mol](https://github.com/YinhanHe123/new_LLM4GNNExplanation/blob/main/our_mol.png) | ![baseline_mol](https://github.com/YinhanHe123/new_LLM4GNNExplanation/blob/main/baseline_mol.png)
|:--:|:--:|:--:|
| *Original molecule* | *LLM-GCE's output* | *CF_GNNExplainer's output* |

We see that LLM-GCE is successfully able to produce a counterfactual with minimal changes to the original input, compared to CF_GNNExplainer, which removes a large portion of the original molecule. Notice the subtle removal of part of one of the side chains. Further, LLM-GCE's output has a superior proximity of 19.26 versus CF_GNNExplainer's 24.76.

In addition, we inspect the generated smiles for various invalid smiles from baseline outputs and compare with outputs from LLM-GCE. Below, we show an example of molecule 450 from AIDS.
| `CSC1OC(C)(C)OC1=O` | `O=COCCSC1OC1`  |  `[AsH3].[As]#B[AsH]#Cl12(=[AsH])#[As](=[As]1)=[AsH]=2` |
|:--:|:--:|:--:|
| *Original SMILES* |  *LLM-GCE's SMILES*  | *CLEAR's SMILES* |

LLM-GCE's SMILES output passes RDKit's valence theory validity checks, while CLEAR struggles to produce a SMILES string which is chemically feasible. Next, we show an example of a failure case of LLM-GCE, but which still demonstrates improvement. Below are the SMILES strings for molecule 1737 from Tox21.
| `c1ccc2cc(CC3=NCCN3)ccc2c1` | `OClSNCCCNCCNCCNCCN` | `C=CC.C=CC.CCC.O=CO`|
|:--:|:--:|:--:|
| *Original SMILES* | *LLM-GCE's SMILES*  | *CF_GNNExplainer's SMILES* |

In this example, we see that, while LLM-GCE is unable to construct a valid counterfactual, and it indeed hallucinates a sulfur and oxygen atom, its generated molecule is an improvement over CF_GNNExplainer, adding nitrogen atoms and refraining from hallucinated double bonds.


### (3) (R1) Parameter sensitivity of $\alpha$ and $\beta$.

|         | 5 | 2 | 1 | 0.5 | 0.2 |
|---------|---|---|---|-----|-----|
| AIDS    |   |   |   |     |     |
| Clintox |   |   |   |     |     |

### (4) (R2) LLM can be hard to generalize to graphs other than molecules. 

### (5) (R2) Measure the performance of counterfactual explanation with solely GPT-4, i.e., SMILES in, SMILES out, calculate accuracy.

### (6) (R3, R5) Use different LLMs for prompt provider, for example GPT-MolBERTa/ChemGPT.
Unfortunately, the majority of current domain-specific models in this field, such as ChemGPT and MolGen, are solely trained for SMILES completion. As a result, they are not equipped to accurately process our prompts. A case is shown in follows:
```bash=
# Input
User: In CN1CCC(CC1)CNC2=NN3C(=NC=C3C4=CC(=CC=C4)OC(F)(F)F)C=C2, amine may be the most influential for failing clinical trials for toxicity reasons; what can we change amine to to increase the likelihood of it being causing toxicity in clinical trials? Please find the best substitution functional group for amine that can replace the "__" in the last sentence (shown below within " "). DO NOT reply with more than 3 words. Reply ONLY the substitution function group. "This molecule contains __, amide, aromatic, and ether functional groups, in which __ may be the most influential for failing clinical trials for toxicity reasons."
# output by GPT-4
GPT-4: sulfonamide
# output by ChemGPT
ChemGPT: [=C] [O] [C] [C] [C] [Branch1_2] [C] [=O] [O] [C] [Branch1_1] [Branch1_2] [C] [=O] [C] [=C] [C] [=C] [C] [Branch1_2] [Branch1_2] [Branch1_2] [Branch1_2] [Branch1_2] [Branch1_2] [C] [=C] [=O] [C] [=C] [C] [=C] [C] [=C] [NHexpl] [Ring2] [Branch1_2] [Branch1_2] [Branch1_2] [Branch1_2] [Branch1_2] [C] [=O] [C] [C] [Branch1_2] [C] [=O] [C] [C] [C] [=C] [C] [=C] [C] [=C] [Branch2_1] [Ring1] [Branch1_1] [C] [=C] [N] [=C] [Branch1_1] [Branch1_1] [Branch1_1] [Branch1_2] [C] [C] [C] [C] [=C] [Branch1_1] [Branch1_1] [Branch1_1] [Branch1_1] [C] [C] [C] [C] [C] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [Expl=Ring1] [Branch1_1] [C] [=C] [Branch1_1] [Branch1_2] [C] [=C] [Branch1_1] [C] [C] [C] [C] [C] [C] [C] [Branch1_2] [C] [=O] [C] [Branch1_2] [C] [=O] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [Branch1_1] [C] [C] [C] [C] [C] [C] [C] [C] [Branch1_1] [Ring1] [Branch1_3] [Ring1] [Branch2_2] [C] [Branch1_2] [Branch1_3] [Ring1] [Branch1_2] [C] [=O] [C] [C] [C] [C] [=O] [C] [=C] [Branch1_1] [C] [=C] [C] [C] [C] [C] [=C] [C] [Branch1_1] [Branch1_3] [Ring1] [Branch1_3] [C] [C] [C] [C] [C] [=C] [Branch1_1] [C] [C] [N] [Branch1_1] [C] [=C] [N] [C] [Branch1_2] [Branch1_2] [Branch1_2] [Branch1_3] [C] [Branch1_2] [C] [=N] [C] [=C] [Branch1_1] [Branch1_2] [Branch1_3] [Ring2] [=N] [C] [C] [=N] [C] [C] [=N] [=N] [=N] [C] [=C] [C] [Branch1_2] [Branch1_2] [C] [Branch1_2] [C] [Branch1_2] [Branch1_2] [C] [=O] [C] [=C] [C] [=C] [N] [N] [C] [Branch1_2] [Branch2_2] [Branch1_1] [C] [C] [=C] [Ring1] [Branch1_3] [C] [=C] [C] [Branch1_2] [Branch1_3] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring1] [Branch1_2] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring1] [Branch1_2] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring1] [Branch1_2] [C] [=C] [C] [=C] [Ring1] [Branch1_2] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [Ring1] [Branch1_2] [C] [Ring1] [Branch2_3] [Ring1] [C] [=C] [C] [=C] [C] [=C] [C] [Ring1] [Branch2_3] [Ring1] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring1] [Branch1_3] [C] [Branch1_2] [C] [=C] [C] [Branch1_2] [Branch1_1] [Branch1_2] [Ring2] [=C] [C] [=C] [C] [=C] [C] [Ring1] [Branch2_3] [Ring1] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring2] [Ring1] [Branch1_2] [C] [C] [Branch1_1] [C] [Branch1_1] [C] [Branch1_1] [C] [C] [Branch1_1] [C] [Ring2] [Ring1] [Branch1_2] [C] [C] [C] [C] [Ring2] [Ring1] [Branch1_2] [Branch1_2] [Branch1_2] [C] [=O] [C] [=O] [C] [C] [C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring1] [Branch1_2] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring1] [P] [=O] [C] [=C] [C] [Ring2] [Ring1] [Branch1_1] [C] [=C] [C] [C] [=C] [C]    
```
We will test these models by direcly generating smiles, and test other general LLMs (e.g. Llama-2) on the original text pair generation task. The results will be updated as soon as possible. 
