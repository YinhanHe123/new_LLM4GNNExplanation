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


## 5 Extended Explanation for Reviewer's Concerns and Questions

### 5.1 (R1) The reason for not finding the ground-truth counterfactual in brute forth.
Our evaluated graphs has the number of nodes ranging from 95 to 475. It is not realistic to find the groundtruth counterfactuals, since the combinatorial complexity, if we do not consider the valence theory, we assume that there are N nodes and M edges in the original graph. We also assume that there are $l$ types of nodes and $p$ types of edges available. Let us cosnider the complexity of one execution. One execution can be adding node/deleting node/adding edge/deleting edge/changing node type/changing edge type. For adding one node, we need to add one new node ($l$ choices) and at least one edge ($p$ choices) to connect the node to an existing node. Then the number of possible variaitons are $lpN$. For deleting one node, we have $N$ choices. For adding one edge, we need to select a node pair from the current graph (molecule) which is not connected by any edge. Typically, the molecule graphs are far from fully connected (the number of edges are $\leq10\%$ in all possible edges). Therefore, we have $0.9*(N-1)N/2*p$ choices. For deleting one edge, we have $0.1*(N-1)N/2$ choices. For chaning one node type, we have $N*(l-1)$ choices. For chaning one edge type, we have $0.1*(N-1)N/2*(p-1)$ choices. To sum up the number of choices are $lpN+0.9Ep+0.1E+N(l-1)+0.1E(p-1)$, where $E=(n-1)n/2$. We can plug in an example, say $N=200, p=3, l=10$, which gives $10*3*200+0.9*19900*3+200*(10-1)+0.1*19900*2=65510$ possible counterfactuals. And each conditions needs to be put in GT-GNN for evalaution, which is  not realistic. The computation above only account for one step of modification. However, it is possible that multiple steps of modificaitons are required for achieving counterfactuals. Based on this calculation, we find that doing two modificaions give us more than $4.5$ billion ($4,585,700,000$) possible counterfactuals, which makes it even more impossible to achieve finding the groundtruth counterfactuals. Even if the GPT-GNN can evaluate one molecule graph within $0.1$ seconds, it takes $14.5$ years to finish running it.

### 5.2 (R2) LLM can be hard to generalize to graphs other than molecules.
While the current implementation and experiments focus on molecule graphs, the overall framework of LLM-GCE is general and can be adapted to other types of graphs. 
To demonstrate how LLM-GCE can be generalized, let's consider an example of applying it to a social network graph, where nodes represent people and edges represent connections between them. The graph classification task could be predicting a binary attribute of people (e.g. high vs low income) based on their network. In this setting:
The prompts would need to be adjusted to query the LLM for meaningful counterfactuals in the social network context. For example: "The person represented by node X has connections to people with professions A, B and C, which may be most influential for their income level. What changes to their connections would make them more likely to have [high/low] income?" The LLM's knowledge of professions, demographics, socioeconomic factors etc. could be leveraged to generate plausible counterfactual social networks that provide insight into the GNN's predictions. Some domain-specific validity checks on the counterfactuals could be incorporated as needed. Here, we provide a set of prompts for LLM-GCE in this example:

<u>TP Query</u>:
Please describe the social network connections of the person represented by node {node_id}. Your generated response should be STRICTLY in the form of: "This person is connected to people with professions __, __, and __, among which the connection to __ may be most influential for their income level." Fill the blanks with the most relevant professions. If there are fewer than 3 significant professions, list only those that are relevant.

<u>CTP Query</u>:
For the person represented by node {node_id}, their connection to {profession} may be most influential for their {high/low} income level. What change to their connections would make them more likely to have {high/low} income? Please suggest the best alternative profession to replace the "__" in the following sentence (shown within quotes). Reply with ONLY the alternative profession.
"This person is connected to people with professions __, {other_profession1}, and {other_profession2}, among which the connection to __ may be most influential for their income level."

<u>Feedback</u>:
The generated counterfactual social network for node {node_id} shows the following changes: {counterfactual_connections}. The probability of this person having {high/low} income based on the counterfactual is {predicted_prob}. Please adjust ONE of the professions in the previous sentence (shown within quotes) to further {increase/decrease} the likelihood of {high/low} income. Reply ONLY in the format (old profession):(new profession).
"{original_sentence}"

<u>Validity Checking</u>:
Given the social network connections described as {connection_description}, please check if they represent a plausible real-world scenario. Consider factors like typical career paths, professional associations, and demographic correlations. If plausible, reply VALID. If not, reply INVALID. Reply with just a single word.

<u>Reconstruction</u>:
Please modify the following counterfactual social network connections to make them more realistic while preserving the overall structure. Aim to make minimal changes that improve plausibility.
Original connections: {original_connections}
Counterfactual connections: {counterfactual_connections}
Reply with ONLY the revised counterfactual connections.

Currently, openly available text-annotated graph datasets suited for counterfactual explanations are currently limited outside of the molecular domain. Curating such datasets for more diverse graph types is an important direction for future work to advance research on interpretable GNNs for various domains.

## 6 Complementary Experiment Results

### 6.1 (R1, R2, R5) Model efficiency and scalability evaluation.
#### 6.1.1 Efficiency
Below are runtime metrics for training for 250 epochs and generating counterfactual graphs for AIDS and ClinTox. The metrics are in seconds. All of the experiments were conducted on a single Nvidia RTX A6000 serially on a server equipped with 512GB RAM and dual AMD EPYC 7543 32-core CPUs. The reported result represents the average of five separate experiments.
|         | AIDS | ClinTox | 
|---------|------|---------|
| GNN_Explainer    | 1112.28  | 80.13  |
| CF_GNNExplainer |  2878.85 | 215.40  |   
| RegExplainer | 3457.97  | 201.26  |   
| CLEAR        | 1061.05  | 250.24  |
| LLM-GCE      | 7925.60 | 3441.08 |

Here, we see that GNN_Explainer is one of the faster baselines between the two datasets with RegExplainer being the slowest for AIDS. LLM-GCE is the slowest method compared to the baselines.

#### 6.1.2 Scalability
We further evaluate our method on the Peptides-func dataset from [LRGB](https://github.com/vijaydwivedi75/lrgb). The Peptides-func dataset has an average node count of 150.94, significantly higher than the average node numbers in the five datasets evaluated in our paper, which range from 15.69 to 27.74. To adapt to a binary classification task, we only consider the fifth label type (antiviral) from the original multi-labeled dataset. We will update the results as soon as possible.

### 6.2 (R1) More case studies should be done.

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


### 6.3 (R1) Parameter sensitivity of $\alpha$ and $\beta$.

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

### 6.4 (R2) Measure the performance of counterfactual explanation with solely GPT-4, i.e., SMILES in, SMILES out, calculate accuracy.

### 6.5 (R3, R5) Use different LLMs for prompt provider, for example GPT-MolBERTa/ChemGPT.
#### 6.5.1 Different auto-encoder
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
#### 6.5.2 Different LLMs for prompt provider
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


