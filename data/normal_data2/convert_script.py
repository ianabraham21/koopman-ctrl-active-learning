import numpy as np

TRIAL_=4
paths = [ 'normal_data2/', 'direct_attempts/','active_learning2/','gp_data/']

for path in paths:

    data = np.load(path+'trajectories.npy')
    np.savetxt(path+'trajectories.csv', data[TRIAL_])
