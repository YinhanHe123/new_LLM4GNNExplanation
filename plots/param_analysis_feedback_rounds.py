import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import MaxNLocator

feedback_times = (1, 2, 3, 4, 5) 
validity = [11.515151515151516, 12.424242424242422, 9.84848484848485, 10.0, 8.939393939393941] # devise by 100
validity_std = [ 2.6848553252529315, 2.424242424242425, 3.2847702104058794, 3.533910239300182, 2.553378719144351]
proximity = [38.03017732200169, 38.210168806893016, 37.55121032956359, 38.535303727558684, 38.93784172369804]
proximity_std = [2.2260197632679475, 0.7303319341671499, 1.407979663613694, 1.8367096496599309, 1.8893206216107754]
validity, validity_std = np.array(validity), np.array(validity_std)
proximity, proximity_std = np.array(proximity), np.array(proximity_std)

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 55


fig, ax1 = plt.subplots(figsize=(13, 13))  # Ensuring square aspect ratio
ax1.set_facecolor('#f0f0f0')


ax1.plot(feedback_times, validity, 'g-', marker='^', markersize=45, label='Validity', color='#248000', lw=3)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_xlabel('Feedback Iterations')
ax1.set_ylabel('Validity (%)')
ax1.set_ylim(bottom=-4, top=16)
ax1.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5)
# ax1.fill_between(
#     feedback_times, 
#     validity - validity_std, 
#     validity + validity_std, 
#     color='#86e660', alpha=0.2)

ax2 = ax1.twinx()
ax2.plot(feedback_times, proximity, 'b-', marker='*', markersize=45, label='Proximity', color='#9400D3', lw=3) 
ax2.set_ylabel('Proximity')
ax2.set_ylim(bottom=-3, top=43)
ax2.spines['top'].set_visible(False)
ax2.grid(False)
# ax2.fill_between(
#     feedback_times, 
#     proximity - proximity_std, 
#     proximity + proximity_std, 
#     color='#d182f2', alpha=0.2)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles, labels = handles1 + handles2, labels1 + labels2

plt.legend(handles, labels, loc='lower left')
plt.show()
fig.savefig('feedback.pdf', format='pdf', bbox_inches='tight')