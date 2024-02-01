from rdkit import Chem
import torch

NODE_LABEL_MAP = {
                    "AIDS": {1.0: 'As', 2.0: 'Au', 3.0: 'B', 4.0: 'Bi', 5.0: 'Br', 6.0: 'C', 7.0: 'Cl', 
                             8.0: 'Co', 9.0: 'Cu', 10.0: 'F', 11.0: 'Ga', 12.0: 'Ge', 13.0: 'H', 
                             14.0: 'Hg', 15.0: 'Ho', 16.0: 'I', 17.0: 'K', 18.0: 'Li', 19.0: 'Mg', 
                             20.0: 'N', 21.0: 'Na', 22.0: 'Nd', 23.0: 'Ni', 24.0: 'O', 25.0: 'P', 
                             26.0: 'Pb', 27.0: 'Pd', 28.0: 'Pt', 29.0: 'Rh', 30.0: 'Ru', 31.0: 'S', 
                             32.0: 'Sb', 33.0: 'Se', 34.0: 'Si', 35.0: 'Sn', 36.0: 'Tb', 37.0: 'Te', 
                             38.0: 'Tl', 39.0: 'W', 40.0: 'Zn'},
                    "BBBP": { 1.0: 'B', 2.0: 'Br', 3.0: 'C', 4.0: 'Ca', 5.0: 'Cl', 6.0: 'F', 7.0: 'H', 
                             8.0: 'I', 9.0: 'N', 10.0: 'Na', 11.0: 'O', 12.0: 'P', 13.0: 'S'},
                    "Mutagenicity": { 1.0: 'Br', 2.0: 'C', 3.0: 'Ca', 4.0: 'Cl', 5.0: 'F', 6.0: 'H', 
                                      7.0: 'I', 8.0: 'K', 9.0: 'Li', 10.0: 'N', 11.0: 'Na', 12.0: 'O', 
                                      13.0: 'P', 14.0: 'S'},
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
                              44.0: 'Ti', 45.0: 'Yb', 46.0: 'Zn', 47.0: 'Zr'}
                   }

EDGE_LABEL_MAP = {1: Chem.rdchem.BondType.SINGLE,
                  2: Chem.rdchem.BondType.DOUBLE, 
                  3: Chem.rdchem.BondType.TRIPLE,}

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

def smiles_to_graph(smiles, dataset, max_nodes=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol == None:
        return None, None, None, None
    # mol = Chem.AddHs(mol)
    Chem.KekulizeIfPossible(mol, clearAromaticFlags=True)

    if max_nodes == None:
        max_nodes = mol.GetNumAtoms()

    inverted_node_map = {sym:num for num, sym in NODE_LABEL_MAP[dataset].items()}
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
    
    mask = torch.BoolTensor([True] * mol.GetNumAtoms() + [False] * (max_nodes - mol.GetNumAtoms()))
    
    return atom_features, adjacency_matrix, edge_attr_matrix, mask