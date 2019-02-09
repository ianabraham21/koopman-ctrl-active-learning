import numpy as np

TRIAL_=3
data = np.load('trajectories.npy')
np.savetxt('trajectories.csv', data[TRIAL_])
