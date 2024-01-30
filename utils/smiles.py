from rdkit import Chem
from rdkit.Chem import AllChem
import torch

NODE_LABEL_MAP = {1.0: 'C', 2.0: 'O', 3.0: 'N', 4.0: 'Cl', 5.0: 'F', 6.0: 'S', 7.0: 'Se', 
                  8.0: 'P', 9.0: 'Na', 10.0: 'I', 11.0: 'Co', 12.0: 'Br', 13.0: 'Li', 
                  14.0: 'Si', 15.0: 'Mg', 16.0: 'Cu', 17.0: 'As', 18.0: 'B', 19.0: 'Pt', 
                  20.0: 'Ru', 21.0: 'K', 22.0: 'Pd', 23.0: 'Au', 24.0: 'Te', 25.0: 'W', 
                  26.0: 'Rh', 27.0: 'Zn', 28.0: 'Bi', 29.0: 'Pb', 30.0: 'Ge', 31.0: 'Sb', 
                  32.0: 'Sn', 33.0: 'Ga', 34.0: 'Hg', 35.0: 'Ho', 36.0: 'Tl', 37.0: 'Ni', 
                  38.0: 'Tb', 39.0: 'Nd', 40.0: 'H', 41.0: 'Ca'}

EDGE_LABEL_MAP = {1: Chem.rdchem.BondType.SINGLE,
                  2: Chem.rdchem.BondType.DOUBLE, 
                  3: Chem.rdchem.BondType.TRIPLE,}

def graph_to_smiles(x, adj_matrix, edge_attr_matrix, mask):
    mol = Chem.RWMol()
    node_idx_map = {}
    for idx, node in enumerate(x):
        if mask[idx]:
            a = Chem.Atom(NODE_LABEL_MAP[node[0]])
            mol_idx = mol.AddAtom(a)
            node_idx_map[idx] = mol_idx
    edge_idxs = torch.nonzero(adj_matrix)
    for edge_idx in edge_idxs:
        if edge_idx[0] < edge_idx[1]:
            i, j = edge_idx[0].item(), edge_idx[1].item()
            mol.AddBond(node_idx_map[i], node_idx_map[j], EDGE_LABEL_MAP[edge_attr_matrix[i,j].item()])
    return Chem.MolToSmiles(mol.GetMol()) 

def smiles_to_graph(smiles, max_nodes):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)

    inverted_node_map = {sym:num for num, sym in NODE_LABEL_MAP.items()}
    inverted_edge_map = {bond_type:num for num, bond_type in EDGE_LABEL_MAP.items()}

    atom_features = [[inverted_node_map[atom.GetSymbol()]] for atom in mol.GetAtoms()]
    atom_features.extend([[0] for _ in range(max_nodes - mol.GetNumAtoms())])
    atom_features = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
    
    adjacency_matrix = torch.zeros((max_nodes, max_nodes), dtype=torch.float)
    edge_attr_matrix = torch.zeros((max_nodes, max_nodes), dtype=torch.float)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
        edge_attr_matrix[i,j] = inverted_edge_map[bond.GetBondType()]
        edge_attr_matrix[j, i] = inverted_edge_map[bond.GetBondType()]
    
    mask = torch.BoolTensor([True] * mol.GetNumAtoms() +\
                                [False] * (max_nodes - mol.GetNumAtoms()))
    return atom_features, adjacency_matrix, edge_attr_matrix, mask