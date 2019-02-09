import numpy as np

TRIAL_=5
data = np.load('trajectories.npy')
np.savetxt('trajectories.csv', data[TRIAL_])
