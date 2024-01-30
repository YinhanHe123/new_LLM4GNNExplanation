DATASET_QUERY_MAP = {
    "AIDS": ["AIDS treatment", "a potential AIDS drug"],
    "BBBP": ["blood-brain-barrier permeability (BBBP)", "blood-brain-barrier permeable"],
    "Mutagenicity": ['mutagenicity', "a potential mutagen"],
    "SIDER": ["causing Hepatobiliary disorders", "causing Hepatobiliary disorders"],
    "Tox21": ["toxicity testing on peroxisome proliferator-activated receptor gamma (NR-PPAR-gamma)", "toxic against NR-PPAR-gamma"]

}

query_format = 'Please describe this molecule: {molecule_data} Your generated response is ONLY a text description STRICTLY in the form of: "This molecule contains __, __, __, and __ functional groups, in which __ may be the most influential for {dataset_description}." NO OTHER sentence patterns allowed. Here, __ is the functional groups (best each less than 10 atoms) or significant subgraphs alphabetically. If you can not find 4 functional groups significant subgraphs, you can just put all you have found in the __ areas)'    

cf_query_format = 'In {smiles}, {key_component} may be the most influential for {dataset_description}; what can we change {key_component} to to {likely} the likelihood of it being {molecule_description}? Please find the best substitution functional group for {key_component} that can replace the "__" in the last sentence (shown below within " ") and reply with ONLY the NAME of the substituted functional group. DO NOT reply with more than 3 words. "{caption_to_be_revised}"'
feedback_format = 'The generated counterfactual is {smiles}. The probability of it being {molecule_description} is {true_prob}. Please adjust ONE of the functional groups in the last sentence (shown below within " ") to {likely} the likelihood of the generated counterfactual being {molecule_description}. ONLY the functional group names in the sentence may be changed. Reply ONLY in the format (old functional group):(new functional group). "{original_caption}"'

check_valid_query = "Given this molecule in SMILES representation {molecule}, please check if it satisfies the Valance Bond Theory. If it does, then reply VALID. If not, then reply INVALID. DO NOT reply with more than 1 word"
get_valid_query = "Please find a valid molecule that is the most similar to the molecule {molecule}. The molecule MUST satisfy the Valance Bond Theory and be SIMILAR to the given molecule. ONLY reply in SMILES representation"

openai_api_key = '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'