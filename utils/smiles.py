from rdkit import Chem
import torch

NODE_LABEL_MAP = {
                    "AIDS": { 1: 'C', 2: 'O', 3: 'N', 4: 'Cl', 5: 'F', 6: 'S', 7: 'Se', 8: 'P', 9: 'Na', 
                              10: 'I', 11: 'Co', 12: 'Br', 13: 'Li', 14: 'Si', 15: 'Mg', 16: 'Cu', 17: 'As', 
                              18: 'B', 19: 'Pt', 20: 'Ru', 21: 'K', 22: 'Pd', 23: 'Au', 24: 'Te', 25: 'W', 
                              26: 'Rh', 27: 'Zn', 28: 'Bi', 29: 'Pb', 30: 'Ge', 31: 'Sb', 32: 'Sn', 33: 'Ga', 
                              34: 'Hg', 35: 'Ho', 36: 'Tl', 37: 'Ni', 38: 'Tb'},
                    "BBBP": { 1.0: 'B', 2.0: 'Br', 3.0: 'C', 4.0: 'Ca', 5.0: 'Cl', 6.0: 'F', 7.0: 'H', 
                             8.0: 'I', 9.0: 'N', 10.0: 'Na', 11.0: 'O', 12.0: 'P', 13.0: 'S'},
                    "Mutagenicity": { 1: 'C', 2: 'O', 3: 'Cl', 4: 'H', 5: 'N', 6: 'F', 7: 'Br', 8: 'S', 9: 'P', 
                                      10: 'I', 11: 'Na', 12: 'K', 13: 'Li', 14: 'Ca'},
                    "SIDER": {1.0: 'Ag', 2.0: 'As', 3.0: 'Au', 4.0: 'B', 5.0: 'Ba', 6.0: 'Br', 7.0: 'C', 
                              8.0: 'Ca', 9.0: 'Cf', 10.0: 'Cl', 11.0: 'Co', 12.0: 'Cr', 13.0: 'Cu', 
                              14.0: 'F', 15.0: 'Fe', 16.0: 'Ga', 17.0: 'Gd', 18.0: 'Ge', 19.0: 'H', 
                              20.0: 'I', 21.0: 'In', 22.0: 'K', 23.0: 'La', 24.0: 'Li', 25.0: 'Mg', 
                              26.0: 'Mn', 27.0: 'N', 28.0: 'Na', 29.0: 'O', 30.0: 'P', 31.0: 'Pt', 
                              32.0: 'Ra', 33.0: 'S', 34.0: 'Se', 35.0: 'Sm', 36.0: 'Sr', 37.0: 'Tc', 
                              38.0: 'Tl', 39.0: 'Y', 40.0: 'Zn'}, 
                    "Tox21": { 1.0: 'Ag', 2.0: 'Al', 3.0: 'As', 4.0: 'Au', 5.0: 'B', 6.0: 'Ba', 7.0: 'Be', 
                              8.0: 'Bi', 9.0: 'Br', 10.0: 'C', 11.0: 'Ca', 12.0: 'Cd', 13.0: 'Cl', 
                              14.0: 'Co', 15.0: 'Cr', 16.0: 'Cu', 17.0: 'F', 18.0: 'Fe', 19.0: 'Ge', 
                              20.0: 'H', 21.0: 'Hg', 22.0: 'I', 23.0: 'In', 24.0: 'K', 25.0: 'Li', 
                              26.0: 'Mg', 27.0: 'Mn', 28.0: 'Mo', 29.0: 'N', 30.0: 'Na', 31.0: 'Nd', 
                              32.0: 'Ni', 33.0: 'O', 34.0: 'P', 35.0: 'Pb', 36.0: 'Pd', 37.0: 'Pt', 
                              38.0: 'S', 39.0: 'Sb', 40.0: 'Se', 41.0: 'Si', 42.0: 'Sn', 43.0: 'Sr', 
                              44.0: 'Ti', 45.0: 'Yb', 46.0: 'Zn', 47.0: 'Zr'},
                    "ClinTox": {1: 'Cu', 2: 'Tc', 3: 'Tl', 4: 'Al', 5: 'Cl', 6: 'Zn', 7: 'I', 8: 'Ti',
                                9: 'Ca', 10: 'S', 11: 'O', 12: 'N', 13: 'Pt', 14: 'As', 15: 'H', 16: 'Br',
                                17: 'Co', 18: 'Hg', 19: 'B', 20: 'Se', 21: 'Si', 22: 'C', 23: 'Mn', 24: 'Au', 
                                25: 'P', 26: 'Bi', 27: 'Fe', 28: 'F', 29: 'Cr'}
                   }

EDGE_LABEL_MAP = {1: Chem.rdchem.BondType.SINGLE,
                  2: Chem.rdchem.BondType.DOUBLE, 
                  3: Chem.rdchem.BondType.TRIPLE,
                  4: Chem.rdchem.BondType.AROMATIC,
                  5: Chem.rdchem.BondType.DATIVE}


def graph_to_smiles(x, adj_matrix, edge_attr_matrix, mask, dataset):
    mol = Chem.RWMol()
    node_idx_map = {}
    for idx, node in enumerate(x):
        if mask[idx] and int(node[0].item()) > 0:
            a = Chem.Atom(NODE_LABEL_MAP[dataset][int(node[0].item())])
            mol_idx = mol.AddAtom(a)
            node_idx_map[idx] = mol_idx
    edge_idxs = torch.nonzero(adj_matrix)
    for edge_idx in edge_idxs:
        i, j = edge_idx[0].item(), edge_idx[1].item()
        if i < j and i in node_idx_map and j in node_idx_map and int(edge_attr_matrix[i,j].item()) > 0:
            mol.AddBond(node_idx_map[i], node_idx_map[j], EDGE_LABEL_MAP[int(edge_attr_matrix[i,j].item())])
    return Chem.MolToSmiles(mol.GetMol()) 

def smiles_to_graph(smiles, dataset, max_nodes=None, add_hydrogen=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol == None:
        return None, None, None, None
    if add_hydrogen:
        mol = Chem.AddHs(mol)
    Chem.KekulizeIfPossible(mol, clearAromaticFlags=True)

    if max_nodes == None:
        max_nodes = mol.GetNumAtoms()

    inverted_node_map = {sym:num for num, sym in NODE_LABEL_MAP[dataset].items()}
    inverted_edge_map = {bond_type:num for num, bond_type in EDGE_LABEL_MAP.items()}
    
    # print([atom.GetSymbol() for atom in mol.GetAtoms()])
    atom_features = [[inverted_node_map[atom.GetSymbol()]] for atom in mol.GetAtoms() if atom.GetSymbol() != '*']
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
    
    mask = torch.BoolTensor([True] * mol.GetNumAtoms() + [False] * (max_nodes - mol.GetNumAtoms()))
    
    return atom_features, adjacency_matrix, edge_attr_matrix, mask