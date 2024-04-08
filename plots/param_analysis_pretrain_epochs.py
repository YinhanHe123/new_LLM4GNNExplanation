import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import MaxNLocator

Epochs = (10, 50, 100, 200) 
validity = [6.969696969696971, 8.333333333333334, 9.84848484848485, 10.151515151515152]
validity_std = [1.6177391290956533, 3.921872456225692, 3.2847702104058794, 6.666666666666667]
proximity = [38.95636907346321, 38.142317842127206, 37.55121032956359, 39.15634443317532] # devise by 100
proximity_std = [1.7026591960806843,  2.8355254168025423, 1.407979663613694, 1.110101219640706]
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 55 

validity, validity_std = np.array(validity), np.array(validity_std)
proximity, proximity_std = np.array(proximity), np.array(proximity_std)

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 55


fig, ax1 = plt.subplots(figsize=(13, 13))  # Ensuring square aspect ratio
ax1.set_facecolor('#f0f0f0')


ax1.plot(Epochs, validity, 'g-', marker='^', markersize=45, label='Validity', color='#248000', lw=3)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_xlabel('Pretrain Epochs')
ax1.set_ylabel('Validity (%)')
ax1.set_ylim(bottom=-4, top=16)
ax1.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5)
# ax1.fill_between(
#     Epochs, 
#     validity - validity_std, 
#     validity + validity_std, 
#     color='#86e660', alpha=0.2)

ax2 = ax1.twinx()
ax2.plot(Epochs, proximity, 'b-', marker='*', markersize=45, label='Proximity', color='#9400D3', lw=3) 
ax2.set_ylabel('Proximity')
ax2.set_ylim(bottom=-3, top=43)
ax2.spines['top'].set_visible(False)
ax2.grid(False)
# ax2.fill_between(
#     Epochs, 
#     proximity - proximity_std, 
#     proximity + proximity_std, 
#     color='#d182f2', alpha=0.2)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles, labels = handles1 + handles2, labels1 + labels2

plt.legend(handles, labels, loc='lower left')
plt.show()

fig.savefig('epochs.pdf', format='pdf', bbox_inches='tight')