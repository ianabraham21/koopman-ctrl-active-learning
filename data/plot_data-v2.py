import numpy as np
import matplotlib.pyplot as plt

import brewer2mpl

paths = [ 'normal_data2/', 'direct_attempts/','gp_data/','active_learning2/']
#paths = [ 'normal_data2/', 'direct_attempts/','gp_data/','active_learning2/', 'active_learning-init-variation-0.1/', 'active_learning-init-variation-0.01/','active_learning-init-variation-10.0/']
labels = [ 'Noisey Inputs', 'Direct Stabilization','GP', 'Our Method','init variation', '0.01 var', '10.0 var']
# paths = ['active_learning/']
colors = ['C1','C2', 'C3', 'C0','C4', 'C5', 'C6']

#bmap = brewer2mpl.get_map('Set2', 'qualitative', len(paths))
#colors = bmap.mpl_colors 

linestyles = ['-','-',':','--', '--', '--', '--']
params = {
   'axes.labelsize': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   'text.usetex': False,
   #'figure.figsize': [6.5, 1.5]
    'figure.autolayout': True
   }
plt.rcParams.update(params)

scale = 3
ratio = 1.461 # 1.161 
fig, ax = plt.subplots(1, 1, figsize=(scale*ratio,scale))

for path, color in zip(paths, colors):
    costs = np.load(path + 'costs.npy')/200
    max_cost = -np.inf
    if path == paths[0]:
        active_learning_costs = costs
        max_idx = None
        for i in range(costs.shape[0]):
            if costs[i,:].max() > max_cost:
                max_cost = costs[i,:].max()
                max_idx = i
active_learning_costs = list(active_learning_costs)
active_learning_costs.pop(max_idx)
active_learning_costs = np.array(active_learning_costs)

for path, color, label, ls in zip(paths, colors, labels, linestyles):
    costs = np.load(path + 'costs.npy')/200
    xdata = [i/200.0 for i in range(len(costs[0]))]
    if False:
    # if path == paths[0]:
        costs = active_learning_costs
    mean_cost = (np.mean(costs,axis=0))

    ax.plot(xdata, mean_cost, 
            color,
            linestyle=ls,
            label=label)

    # ax[0].plot(xdata, costs.T,color)
    ax.fill_between(xdata,
                    # np.amin(costs,axis=0),
                    mean_cost-np.std(costs,axis=0)/2.0,
                    mean_cost+np.std(costs,axis=0)/2.0,
                    color=color,
                    alpha=0.2)
    
ax.set_ylim(0,0.2)
ax.set_ylabel('Stabilization Error')
ax.set_xlabel('Time (s)')
legend = ax.legend()# bbox_to_anchor=(0.5, 0.6))
frame = legend.get_frame()
frame.set_facecolor('1.0')

plt.savefig('stab-err.pdf')

#fig.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.1)

fig, ax = plt.subplots(1, 1, figsize=(scale*ratio, scale))

for path, color, ls in zip(paths, colors, linestyles):
    inf_gains = np.load(path + 'inf_gains.npy')/200
    if False:
    # if path == paths[0]:
        inf_gains = list(inf_gains)
        inf_gains.pop(max_idx)
        inf_gains = np.array(inf_gains)
    xdata = [i/200.0 for i in range(len(costs[0]))]
    mean_inf_gain =(np.mean(inf_gains,axis=0))
    inf_std = np.std(inf_gains,axis=0)/2.0
    ax.plot(xdata, mean_inf_gain, 
            color,
            linestyle=ls,
            )
    # ax[1].plot(xdata, inf_gains.T, color)
    ax.fill_between(xdata,
                    #np.amin(inf_gains, axis=0),
                    mean_inf_gain-inf_std/2.0,
                    mean_inf_gain+inf_std/2.0,
                    color=color, 
                    alpha=0.2)

ax.set_ylim(0.4,2.0)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Information Gain')

plt.savefig('inf-gain.pdf')

plt.show()
