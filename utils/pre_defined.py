import random
import numpy as np
import torch

DATASET_QUERY_MAP = {
    "AIDS": ["AIDS treatment", "a potential AIDS drug"],
    "BBBP": ["blood-brain-barrier permeability (BBBP)", "blood-brain-barrier permeable"],
    "Mutagenicity": ['mutagenicity', "a potential mutagen"],
    "SIDER": ["causing Hepatobiliary disorders", "causing Hepatobiliary disorders"],
    "Tox21": ["toxicity testing on peroxisome proliferator-activated receptor gamma (NR-PPAR-gamma)", "toxic against NR-PPAR-gamma"],
    "ClinTox": ["failing clinical trials for toxicity reasons", "causing toxicity in clinical trials"]
}

query_format = 'Please describe this molecule: {molecule_data} Your generated response is ONLY a text description STRICTLY in the form of: "This molecule contains __, __, __, and __ functional groups, in which __ may be the most influential for {dataset_description}." NO OTHER sentence patterns allowed. Here, __ is the functional groups (best each less than 10 atoms) or significant subgraphs alphabetically. If you can not find 4 functional groups significant subgraphs, you can just put all you have found in the __ areas)'
# query_format = "Please describe the molecule represented by the following SMILES notation: {molecule_data}. Focus on identifying up to four key functional groups or significant subgraphs, particularly noting any that could be relevant for {dataset_description}. List the functional groups alphabetically and highlight which one might be the most influential for treatment purposes."

# query_format = 'This is a molecule in SMILES representation: {molecule_data}. Please generate a text description that includes up to four functional groups (consists of less than 10 atoms). Reply ONLY the functional groups and specify one that is most influential for {dataset_description}'
cf_query_format = 'In {smiles}, {key_component} may be the most influential for {dataset_description}; what can we change {key_component} to to {likely} the likelihood of it being {molecule_description}? Please find the best substitution functional group for {key_component} that can replace the "__" in the last sentence (shown below within " "). DO NOT reply with more than 3 words. Reply ONLY the substitution function group. "{caption_to_be_revised}"'
feedback_format = 'The generated counterfactual is {smiles}. The probability of it being {molecule_description} is {true_prob}. Please adjust ONE of the functional groups in the last sentence (shown below within " ") to {likely} the likelihood of the generated counterfactual being {molecule_description}. ONLY the functional group names in the sentence may be changed. Reply ONLY in the format (old functional group):(new functional group). "{original_caption}"'

check_valid_query = "Given this molecule in SMILES representation {molecule}, please check if it satisfies the Valance Bond Theory. If it does, then reply VALID. If not, then reply INVALID. DO NOT reply with more than 1 word"
get_valid_query = "Please find a valid molecule that is the most similar to the molecule {molecule}. The molecule MUST satisfy the Valance Bond Theory and be SIMILAR to the given molecule. ONLY reply in SMILES representation"

reconst_query = 'Please recover the valid SMILES representation from the corrupted one. The recovered SMILES should be similar to the reference SMILES but make as few changes as possible. Think of the Periodic Table and the changes of every element. Make sure that the recovered SMILES is \
                valid and follows the Valence Bond Theory. The desired SMILES is SIMILSR to BOTH the CORRUPTED SMILES and the REFERENCE SMILES but it is NOT the REFERENCE SMILES. The corrupted smiles is "{cf_smiles}" and the reference smiles is "{original_smiles}". Respond ONLY with the reconstructed SMILES. The reconstructed SMILES should have AT MOST {max_num_nodes} atoms'

openai_api_key = "sk-xxx"


def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False