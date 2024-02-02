# read the ClinTox_smiles and get all the appearred atom types in a dictionary

import csv
from rdkit import Chem 

csv_file = open('ClinTox_smiles.csv', 'r')
csv_reader = csv.reader(csv_file)
atom_types = set()
next(csv_reader) # skip the header
for row in csv_reader:
    smiles = row[0]
    try: 
        rdkit_mol = Chem.MolFromSmiles(smiles)
        for atom in rdkit_mol.GetAtoms():
            print(atom.GetSymbol())
            atom_types.add(atom.GetSymbol())
    except: 
        print(smiles)
        pass


#     smiles = row[0]
#     for atom in smiles:
#         atom_types.add(atom)

# # delete all the non-alphabet characters
# atom_types = list(atom_types)
# for atom in atom_types:
#     if not atom.isalpha():
#         atom_types.remove(atom)
#     else:
#         pass

# change the atom types into a dictionary, where the keys are indices
atom_types_dict = {}
for i, atom in enumerate(atom_types):
    # if atom is in alphabet (no matter upper or lower case), then add it to the dictionary
    if atom.isalpha():
        if atom.islower():
            atom_types_dict[i+1] = atom.upper()
        else:
            atom_types_dict[i+1] = atom
    else:
        pass
    
print(atom_types_dict)


    
