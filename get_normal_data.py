import numpy as np
from koopman_operator import KoopmanOperator
from quad import Quad
from task_objective import Task, Adjoint
import matplotlib.pyplot as plt
import scipy.linalg
from group_theory import VecTose3, TransToRp, RpToTrans
from finite_horizon_lqr import FiniteHorizonLQR


TRIALS = 20
# STARTING_SEED_ = 50 # initially 10
STARTING_SEED_ = 10

np.set_printoptions(precision=4, suppress=True)
np.random.seed(STARTING_SEED_)

class ContLQR(object):

    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        try:
            self.P = scipy.linalg.solve_continuous_are(A, B, Q, R)
            self.Klqr = np.linalg.inv(R).dot(B.T.dot(self.P))
        except:
            print('not working')
            self.Klqr = np.zeros(self.B.T.shape)
        self.target_state = None
    def set_target_state(self, target):
        self.target_state = target
    def __call__(self, x, xd=None):
        return np.clip(-self.Klqr.dot(x - self.target_state), -6,6)
    def get_linearization_from_trajectory(self, trajectory):
        return [-self.Klqr for state in trajectory]
def get_measurement(x):
    g = x[0:16].reshape((4,4))
    R,p = TransToRp(g)
    twist = x[16:]
    grot = np.dot(R, [0., 0., -9.81])
    return np.concatenate((grot, twist))

def get_position(x):
    g = x[0:16].reshape((4,4))
    R,p = TransToRp(g)
    return p

def main():


    quad = Quad()
    koopman_operator = KoopmanOperator(quad.time_step)
    adjoint = Adjoint(quad.time_step)
    task = Task()

    horizon = 20
    control_reg = np.diag([10.0]*4)
    inv_control_reg = np.linalg.inv(control_reg)
    default_action = lambda x: np.random.uniform(-0.1,0.1,size=(4,))
    action_schedule = [default_action(None) for _ in range(horizon-1)]
    ustar = [default_action(None) for _ in range(horizon-1)]
    djdlam = [0.0 for _ in range(horizon-1)]

    sat_val = 6.0

    trajectories = []
    actions = []
    costs = []
    inf_gains = []

    # path = 'data/normal_data2/'
    # trajectories = list(np.load(path+'trajectories.npy'))
    # actions = list(np.load(path+'actions.npy'))
    # costs = list(np.load(path+'costs.npy'))
    # inf_gains = list(np.load(path+'inf_gains.npy'))

    for trial in range(TRIALS): # ok I have to do that many trials

        R = np.diag([1.,1.,1.])
        p = np.array([0.,0.,0.])
        g = RpToTrans(R, p).ravel()
        twist = np.random.uniform(-1.,1.,size=(6,))*2.0
        state = np.r_[g, twist]

        trajectory_stack = [state.copy()]
        cost = []
        inf_gain = []
        actions_taken = []
        for time in range(1000): # ok so 1000 / 200 is 5 seconds total simulation

            # if True:
            if time >= 200:
                sat_val = 6.0
                task.inf_weight = 0.0
                t_state = koopman_operator.transform_state(get_measurement(state))
                Kx, Ku = koopman_operator.get_linearization()
                lqr_policy = FiniteHorizonLQR(Kx, Ku, task.Q, task.R, task.Qf,
                                horizon = horizon)
                lqr_policy.set_target_state(task.target_expanded_state)
                lqr_policy.sat_val = sat_val

                next_action = lqr_policy(t_state)
                if np.isnan(next_action).any():
                    print(' Something went wrong ')
                    next_action = default_action(None)

            elif time < 200:
                sat_val = 6.0
                task.inf_weight = 0.0
                t_state = koopman_operator.transform_state(get_measurement(state))
                next_action = default_action(None)*20.0

            next_state = quad.step(state, next_action)
            koopman_operator.compute_operator_from_data(get_measurement(state),
                                                        next_action,
                                                        get_measurement(next_state))


            state = next_state
            # fix up the containers now
            trajectory_stack.append(state.copy())
            cost.append(task.get_stab_cost(get_measurement(state)))
            actions_taken.append(next_action.copy())
            inf_gain.append(task.information_gain(t_state))
            # print('Time : {:1.2f}, State : {}, Action : {}'.format(
            #         time*quad.time_step, get_measurement(state), next_action
            #         ))
            # running_cost = task.l(t_state, action_schedule[0])
            # print('Time :  {:1.2f}, Cost : {:.2f}'.format(time*quad.time_step, running_cost))
            # del action_schedule[0]
            # action_schedule.append(default_action(None))
            print('Trial : {}, Iter : {}, Pose : {}, {}'.format(trial, time, get_measurement(state), next_action))
        trajectories.append(trajectory_stack)
        costs.append(cost)
        actions.append(actions_taken)
        inf_gains.append(inf_gain)


        np.random.seed(10+trial*100)
        # final_height.append(get_position(state)[-1])
        koopman_operator.clear_operator()
        # plt.clf()
        # plt.plot(final_height)
        # plt.pause(0.01)
    print('Saving the data now.. ')
    path = 'data/normal_data2/'
    np.save(path+'trajectories.npy', trajectories)
    np.save(path+'costs.npy', costs)
    np.save(path+'actions.npy', actions)
    np.save(path+'inf_gains.npy', inf_gains)
    print('Saved data!')











if __name__ == '__main__':
    main()
