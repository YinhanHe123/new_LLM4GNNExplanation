import pandas as pd
from rdkit import Chem
smiles = pd.read_csv("./original_data/SIDER_smiles.csv", header=None)[0].values.tolist()
labels = pd.read_csv("./original_data/SIDER_graph_labels.csv", header=None)[0].values.tolist()
output = pd.read_csv("./original_data/SIDER_output.csv", header=None).values.tolist()

mols = [(i, Chem.MolFromSmiles(x)) for i,x in enumerate(smiles) if Chem.MolFromSmiles(x) is not None]
filtered_mols = [x for x in mols if x[1].GetNumAtoms() <= 100]
idxs = [x[0] for x in filtered_mols]

smiles = [x for i,x in enumerate(smiles) if i in idxs]
labels = [x for i,x in enumerate(labels) if i in idxs]
output = [x for i,x in enumerate(output) if i in idxs]
smiles = pd.DataFrame(smiles)
smiles.to_csv("SIDER_smiles.csv", header=False, index=False)

labels = pd.DataFrame(labels)
labels.to_csv("SIDER_graph_labels.csv", header=False, index=False)

output = pd.DataFrame(output)
output.to_csv("SIDER_output.csv", header=False, index=False)