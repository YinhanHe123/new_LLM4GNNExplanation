import numpy as np

file_path = "./ablation/nf_BBBP.csv"
f = open(file_path, "r")
exp_res = []
output = ""
for line in f.readlines():
    if "Start Experiment" in line or "End Experiment" in line:
        exp_res = []
    elif "Experiment" in line and "validity" not in line:
        res = [float(x) for x in line.split(",")[1:]]
        exp_res.append(res)
    elif "Summary" in line:
        line_split = line.split(",")
        line_split[-2] = f'{np.mean([exp_res[i][-2] for i in range(5)])} ± {np.std([exp_res[i][-2] for i in range(5)])}'
        line_split[-1] = f'{np.mean([exp_res[i][-1] for i in range(5)])} ± {np.std([exp_res[i][-1] for i in range(5)])}'
        line = ",".join(line_split)
        line += "\n"
    output += line
print(output)