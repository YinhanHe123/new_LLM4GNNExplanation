import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib

validity = [5.588235294117647,7.070707071, 2.352941176470588, 9.84848484848485]
validity_std = [2.6956327617387292,6.353406882, 2.6058889921572574, 3.2847702104058794]
proximity = [38.62223609974335, 38.27994283,  22.75172015428543, 37.55121032956359] 
proximity_std = [1.6935805948670837, 1.662707032, 19.44898201901953, 1.407979663613694]
labels = ['LLM-GCE-NP', 'LLM-GCE-NF', 'LLM-GCE-NT', 'LLM-GCE']
colors = ['#f27c4a', '#4d50e0', '#58a611', '#cc1926']
shadow_colors = ['#e9a486', '#9c9ee2', '#9de859', '#ea7b82']
markers = ['s', 'o', 'v', '*']  
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 18
fig, ax = plt.subplots()
ax.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5)
ax.set_facecolor('#f0f0f0')
n_std = 1
for i, (v, p) in enumerate(zip(validity, proximity)):
    plt.scatter(v, p, 
                marker=markers[i], 
                color=colors[i], 
                s=700 if i != 3 else 1800, 
                label=labels[i])
plt.ylim(20, 40)
plt.xlim(2, 11)
plt.xlabel('Validity (%)')
plt.ylabel('Proximity')
legend_handles = [mlines.Line2D(
    [], [], 
    color=color, marker=marker, 
    linestyle='None', markersize=20, label=label
    ) for color, marker, label in zip(colors, markers, labels)]
legend_handles = [legend_handles[-1]] + legend_handles[:-1]
# for v, v_std, p, p_std, shadow_color in zip(validity, validity_std, proximity, proximity_std, shadow_colors):
#     ellipse = Ellipse((v, p), width=v_std*2*n_std, height=p_std*2*n_std, 
#                       edgecolor='none', facecolor=shadow_color, alpha=0.5)
#     ax.add_patch(ellipse)
plt.legend(
    handles=legend_handles, 
    loc='lower right', 
    ncol=1, 
    borderaxespad=0.,
    frameon=True
)

# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()

# scale_ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]) * (plt.gcf().get_size_inches()[1] / plt.gcf().get_size_inches()[0])

# for i in range(3):
#     plt.annotate('',
#                 xy=(validity[i], proximity[i]),  
#                 xytext=(validity[3], proximity[3]),  
#                 arrowprops=dict(
#                     arrowstyle="->", lw=3, facecolor=colors[i], edgecolor=colors[i], alpha=1
#                 ),)
    # mid_x = (validity[i] + validity[3]) / 2
    # mid_y = (proximity[i] + proximity[3]) / 2
    # delta_x = validity[3] - validity[i]
    # delta_y = (proximity[3] - proximity[i]) / scale_ratio
    # angle = np.degrees(np.arctan2(delta_y, delta_x))
    # plt.text(mid_x, mid_y, 
    #          'Annotation{}'.format(i), fontsize=9, 
    #          ha='center', va='center', rotation=angle,)


plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()

plt.savefig('ablation.pdf', format='pdf', bbox_inches='tight')









# import matplotlib.pyplot as plt
# import numpy as np
# import networkx as nx
# import time
# import sys
# import math
# sys.path.append('../methods/')
# import scipy.sparse as sp
# import argparse
# def scale_plot():
#     # Data
#     labels = ["Eigenvec.Cen.", "Close.Cen", "Betw.Cen.", "ExPSCC", "Acquaintance", "CBF", "Random", "PageRank",
#               "Degree", "Hits"]
#     spectral_radius_drop = [1.4853, 1.6244, 1.7159, 1.8612, 0.0685, 1.4677, 0.0134, 1.4874, 1.6388, 0.0953]
#     running_time = [0.0855, 9.2186, 44.0978, 0.0052, 0.3718, 0.0003, 0.0034, 0.0016, 1.8743]
#     markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H']
#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'lime']

#     # Plotting Spectral Radius Drop vs Running Time with specific marker sizes
#     plt.figure(figsize=(8, 6))
#     for i, label in enumerate(labels):
#         if i < len(running_time):
#             plt.scatter(running_time[i], spectral_radius_drop[i], marker=markers[i], s=500, color=colors[i],
#                         label=label if i == 0 else "")  # Only label the first to control legend size

#     # Setting legend with custom handle for marker size
#     handles, labels = plt.gca().get_legend_handles_labels()
#     legend = plt.legend(handles[:1], labels[:1], loc="best", markerscale=0.45)

#     plt.title("Spectral Radius Drop vs Running Time")
#     plt.ylabel("Spectral Radius Drop")
#     plt.xlabel("Running Time (s)")
#     plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="grey")
#     plt.gca().set_facecolor('#eeeeee')
#     plt.tight_layout()

# # Combining the plots
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# plt.sca(axes[0])
# # plt.title("(a)")

# plt.sca(axes[1])
# scale_plot()
# # plt.title("(b)")

# plt.tight_layout()

# plt.savefig("combined_time_plot.pdf")
# plt.show()