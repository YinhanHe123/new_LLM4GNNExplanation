# LLM4GNNExplanation
Using LLMs to guide GNN Counterfactual Explanation

## 1 How to run
Add an OpenAI api key in utils/pre_defined.py and run
```bash
python main.py
```

## 2 Directory Layout
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

## 3 Environments
The dependencies needed to run the code can be installed using
```shell
pip install -r requirements.txt
```


## 4 Model Description
### 4.1 Text Encoder Contrastive Pretraining
Here, we provide a detailed illustration of the contrastive pretraining phase, as depicted in the second box of Figure 3 in our paper.
![image](https://github.com/YinhanHe123/new_LLM4GNNExplanation/assets/44119778/25e346bc-22fc-446d-ae8e-caca1fc92eca)
#### 4.1.1 Intuition
The purpose of this contrastive pretraining phase is to align the graph modality and the text modality within the same embedding space. Specifically, given a graph $G_i$, we generate the corresponding text attribute as the text pair $TP_i$ of $G_i$. Our target is to maximize the alignment between $G_i$ and $TP_i$ for all the $G_i$ in the dataset:
$$\max_\phi P(G_i, TP_i)\text{, i.e. } \max_\phi \text{cosine-similarity}(\text{GT-GNN}(G_i), \phi(TP_i)),$$

where $P(G_i, TP_j)$ denotes the probability score for a pair, $\phi$ represents the parameters of the Bert text encoder. 

#### 4.1.2 Contrastive Pretraining
During the pertaining, each batch of batch size $N$ consists of $N$ pairs of $(G_i, TP_i)$. We construct P/N samples within each randomly sampled batch:
+ Positive samples (pairs): the original pairs in the batch $\lbrace(G_i, TP_i)|0\leq i\leq N\rbrace$.
+ Negative samples (pairs): disordered (graph, text) pairs $\lbrace(G_i, TP_j)|0\leq i, j\leq N, i\neq j\rbrace$.

Intuitively we want to increase the alignment of positive pairs and decrease the alignment of negative pairs. Thus, the contrastive pretraining loss becomes:
$$\mathcal{L}{\text{contr}}=\Sigma_{i\neq j}{\log{P(G_i, TP_j)}} - \Sigma_{k=1}^N\log{P(G_k, TP_k)}.$$


## 5 Complementary Experiment Results

### 5.1 (R1, R2, R5) Model efficiency and scalability evaluation.
#### 5.1.1 Efficiency
Below are runtime metrics for training for 250 epochs and generating counterfactual graphs for AIDS and ClinTox. The metrics are in seconds. All of the experiments were conducted on a single Nvidia RTX A6000 serially on a server equipped with 512GB RAM and dual AMD EPYC 7543 32-core CPUs. The reported result represents the average of five separate experiments.
|         | AIDS | ClinTox | 
|---------|------|---------|
| GNN_Explainer    | 1112.28  | 80.13  |
| CF_GNNExplainer |  2878.85 | 215.40  |   
| RegExplainer | 3457.97  | 201.26  |   
| CLEAR        | 1061.05  | 250.24  |
| LLM-GCE      | 7925.60 | 3441.08 |

Here, we see that GNN_Explainer is one of the faster baselines between the two datasets with RegExplainer being the slowest for AIDS. LLM-GCE is the slowest method compared to the baselines.

#### 5.1.2 Scalability
We further evaluate our method on the Peptides-func dataset from [LRGB](https://github.com/vijaydwivedi75/lrgb). The Peptides-func dataset has an average node count of 150.94, significantly higher than the average node numbers in the five datasets evaluated in our paper, which range from 15.69 to 27.74. To adapt to a binary classification task, we only consider the fifth label type (antiviral) from the original multi-labeled dataset. We will update the results as soon as possible.

|      |  Validity |  Proximity | Validity w/o Feas. | Proximity w/o Feas. |
|------|-----------|------------|--------------------|---------------------|
| GNN_Explainer | $0.0 \pm 0.0$ | na | $0.0 \pm 0.0$ | na |
| CF_GNNExplainer |  $0.0 \pm 0.0$ | na | $0.0 \pm 0.0$ | na |
| RegExplainer |  $0.0 \pm 0.0$ | na | $0.0 \pm 0.0$ | na |
| CLEAR | $0.0 \pm 0.0$ | na| 6.45 $\pm 6.34$ | 43.35 $\pm 31.43 |

### 5.2 (R1) More case studies should be done.

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


### 5.3 (R1) Parameter sensitivity of $\alpha$ and $\beta$.

+ Aids
  | $\beta/\alpha$  | Validity |  Proximity | Validity w/o Feas. | Proximity w/o Feas. |  
  |---------|------|---------|------|---------|
  | 5   |  $0.0\pm0.0$ |  na | $100\pm0.0$  | $834.92 \pm 768.31$  |
  | 2   | $0.0\pm0.0$ |  na |  $100\pm0.0$ | $1203.44 \pm 920.29$  |  
  | 1   | $0.6\pm1.08$  | $7.56\pm10.49$  | $100\pm0.0$  |  $878.64 \pm 741.81$ |  
  | 0.5 | $0.0\pm0.0$ |  na |  $100\pm0.0$ | $464.34 \pm 134.28$  | 
  | 0.2 | $0.15\pm0.3$  | $4.37\pm10.74$  | $100\pm0.0$  |  $829.54 \pm 758.20$ | 
+ Clintox
  | $\beta/\alpha$ | Validity |  Proximity | Validity w/o Feas. | Proximity w/o Feas. |  
  |---------|------|---------|------|---------|
  | 5   |  $0.83\pm1.67$ | $16.98\pm35.96$  | $100\pm0.0$  | $1539.60 \pm 35.55$  |
  | 2   | $0.83\pm1.67$  |  $4.18\pm10.36$ |  $100\pm0.0$ |  $1523.05 \pm 35.44$ |  
  | 1   |  $0.83\pm1.67$  | $14.54\pm31.08$  |  $100\pm0.0$ |  $1532.47 \pm 35.72$ |  
  | 0.5 | $0.00\pm0.00$  |  na | $100\pm0.0$  | $1532.16 \pm 35.72$  | 
  | 0.2 | $1.66\pm2.04$  | $26.15\pm33.57$  |  $100\pm0.0$ | $1531.08 \pm 31.23$  | 

### 5.4 (R2) Measure the performance of counterfactual explanation with solely GPT-4, i.e., SMILES in, SMILES out, calculate accuracy.

Below is a table of chemical validity and proximity scores for generating counterfactuals using solely GPT-4. To reproduce it, please run run_rebuttal.py. 

|      | Validity | Proximity |
|------|--------------|------|
| AIDS |    0        |  na |
|Mutagenicity | 0 | na |
| BBBP | 0 | na |
| ClinTox | 0 | na |
| Tox21 | 0 | na |

### 5.5 (R3, R5) Use different LLMs for prompt provider, for example GPT-MolBERTa/ChemGPT.
#### 5.5.1 Different auto-encoder
We conduct extensive experiments regarding different language models as auto-encoders on the Clintox and Aids datasets. The language models are employed through Huggingface library. 
+ Aids
  |         | Validity |  Proximity | Validity w/o Feas. | Proximity w/o Feas. |  
  |---------|------|---------|------|---------|
  | `bert-base-uncased`   |  $0.13\pm0.18$ | $56.19\pm80.9$  | $100.0\pm0.0$  |  $2187.77\pm456.06$ |
  | `microsoft/deberta-base` |  $0.0\pm0.0$ |  na |   $100.0\pm0.0$  | $925.73\pm874.3$  |  
  | `google/electra-base-discriminator` | $0.35\pm0.44$  | $7.23\pm10.60$  | $100.0\pm0.0$  |  $1498.87\pm827.77$ |  
+ Clintox 
  |         | Validity |  Proximity | Validity w/o Feas. | Proximity w/o Feas. | 
  |---------|------|---------|------|---------|
  | `bert-base-uncased`   | $2.50\pm3.33$  | $18.06\pm29.34$  | $100.0\pm0.0$  | $1692.24 \pm 139.40$ |
  | `microsoft/deberta-base` |  $0.83\pm1.66$ |  $0.28\pm2.56$ |   $100.0\pm0.0$  | $1527.77\pm47.68$  |  
  | `google/electra-base-discriminator` |  $2.50\pm2.04$ | $25.45\pm30.00$  | $100.0\pm0.0$   |  $1568.47\pm52.41$ | 
#### 5.5.2 Different LLMs for prompt provider
Unfortunately, the majority of current domain-specific models in this field, such as ChemGPT and MolGen, are solely trained for SMILES completion. As a result, they are not equipped to accurately process our prompts. A case is shown as follows:
```bash=
# Input
User: In CN1CCC(CC1)CNC2=NN3C(=NC=C3C4=CC(=CC=C4)OC(F)(F)F)C=C2, amine may be the most influential for failing clinical trials for toxicity reasons; what can we change amine to to increase the likelihood of it being causing toxicity in clinical trials? Please find the best substitution functional group for amine that can replace the "__" in the last sentence (shown below within " "). DO NOT reply with more than 3 words. Reply ONLY the substitution function group. "This molecule contains __, amide, aromatic, and ether functional groups, in which __ may be the most influential for failing clinical trials for toxicity reasons."
# output by GPT-4
GPT-4: sulfonamide
# output by ChemGPT
ChemGPT: [=C] [O] [C] [C] [C] [Branch1_2] [C] [=O] [O] [C] [Branch1_1] [Branch1_2] [C] [=O] [C] [=C] [C] [=C] [C] [Branch1_2] [Branch1_2] [Branch1_2] [Branch1_2] [Branch1_2] [Branch1_2] [C] [=C] [=O] [C] [=C] [C] [=C] [C] [=C] [NHexpl] [Ring2] [Branch1_2] [Branch1_2] [Branch1_2] [Branch1_2] [Branch1_2] [C] [=O] [C] [C] [Branch1_2] [C] [=O] [C] [C] [C] [=C] [C] [=C] [C] [=C] [Branch2_1] [Ring1] [Branch1_1] [C] [=C] [N] [=C] [Branch1_1] [Branch1_1] [Branch1_1] [Branch1_2] [C] [C] [C] [C] [=C] [Branch1_1] [Branch1_1] [Branch1_1] [Branch1_1] [C] [C] [C] [C] [C] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [NHexpl] [Expl=Ring1] [Branch1_1] [C] [=C] [Branch1_1] [Branch1_2] [C] [=C] [Branch1_1] [C] [C] [C] [C] [C] [C] [C] [Branch1_2] [C] [=O] [C] [Branch1_2] [C] [=O] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [Branch1_1] [C] [C] [C] [C] [C] [C] [C] [C] [Branch1_1] [Ring1] [Branch1_3] [Ring1] [Branch2_2] [C] [Branch1_2] [Branch1_3] [Ring1] [Branch1_2] [C] [=O] [C] [C] [C] [C] [=O] [C] [=C] [Branch1_1] [C] [=C] [C] [C] [C] [C] [=C] [C] [Branch1_1] [Branch1_3] [Ring1] [Branch1_3] [C] [C] [C] [C] [C] [=C] [Branch1_1] [C] [C] [N] [Branch1_1] [C] [=C] [N] [C] [Branch1_2] [Branch1_2] [Branch1_2] [Branch1_3] [C] [Branch1_2] [C] [=N] [C] [=C] [Branch1_1] [Branch1_2] [Branch1_3] [Ring2] [=N] [C] [C] [=N] [C] [C] [=N] [=N] [=N] [C] [=C] [C] [Branch1_2] [Branch1_2] [C] [Branch1_2] [C] [Branch1_2] [Branch1_2] [C] [=O] [C] [=C] [C] [=C] [N] [N] [C] [Branch1_2] [Branch2_2] [Branch1_1] [C] [C] [=C] [Ring1] [Branch1_3] [C] [=C] [C] [Branch1_2] [Branch1_3] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring1] [Branch1_2] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring1] [Branch1_2] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring1] [Branch1_2] [C] [=C] [C] [=C] [Ring1] [Branch1_2] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [Ring1] [Branch1_2] [C] [Ring1] [Branch2_3] [Ring1] [C] [=C] [C] [=C] [C] [=C] [C] [Ring1] [Branch2_3] [Ring1] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring1] [Branch1_3] [C] [Branch1_2] [C] [=C] [C] [Branch1_2] [Branch1_1] [Branch1_2] [Ring2] [=C] [C] [=C] [C] [=C] [C] [Ring1] [Branch2_3] [Ring1] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring2] [Ring1] [Branch1_2] [C] [C] [Branch1_1] [C] [Branch1_1] [C] [Branch1_1] [C] [C] [Branch1_1] [C] [Ring2] [Ring1] [Branch1_2] [C] [C] [C] [C] [Ring2] [Ring1] [Branch1_2] [Branch1_2] [Branch1_2] [C] [=O] [C] [=O] [C] [C] [C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring1] [Branch1_2] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [C] [=C] [Ring1] [P] [=O] [C] [=C] [C] [Ring2] [Ring1] [Branch1_1] [C] [=C] [C] [C] [=C] [C]    
```
We will test these models by direcly generating smiles, and test other general LLMs (e.g. Llama-2) on the original text pair generation task. The results will be updated as soon as possible. 
+ Aids
  |         | Validity |  Proximity | Validity w/o Feas. | Proximity w/o Feas. | 
  |---------|------|---------|------|---------|
  | `GPT-4` | $0.0\pm0.0$ |  na |   $100.0\pm0.0$ | $925.64 \pm 838.75$ |
  | `Llama-2 7B` |   |   |   |   |
+ Clintox
  |         | Validity |  Proximity | Validity w/o Feas. | Proximity w/o Feas. | 
  |---------|------|---------|------|---------|
  | `GPT-4` | $2.50\pm3.33$  |  $18.05 \pm 34.98$ |  $100.0\pm0.0$ | $1516.92 ± 52.93$ |
  | `Llama-2 7B` |   |   |   |   |


