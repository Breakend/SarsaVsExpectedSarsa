import numpy as np

def sarsa(mdp, alpha = 0.1, gamma = 0.9):
    # Initialization
    Q = [[0 for i in range(mdp.A)] for j in range(mdp.S)]
    pol = [1]*mdp.S
    old_Q = Q
    n_iter = 0
    epsilon = 0.1
#     r = 0 # TODO: Remove once found better termination criteria
    while n_iter < 10000000: # TODO: Pick a finish condition for episode
        s = 0 # Initialize s, starting state

        # With prob epsilon, pick a random action
        if np.random.random_sample() <= epsilon:
            a = np.random.random_integers(0, mdp.A-1)
        else:
            a = np.argmax(Q[s][:])
        r = 0
        while r!= 1: # TODO: Finish episode/trajectory on terminal state
            # Observe S and R
            s_new = np.argmax(mdp.T[s, a, :]) # TODO: Change to stochastic ?
            r = mdp.R[s_new]
            T_new = np.zeros((mdp.S, mdp.S))

            # Pick new action A' from S'
            if np.random.random_sample() <= epsilon:
                a_new = np.random.random_integers(0, mdp.A-1)
            else:
                a_new = np.argmax(Q[s_new][:])
            Q[s][a] = Q[s][a] + alpha*(r + gamma*Q[s_new][a_new] - Q[s][a])
            s = s_new
            a = a_new

            n_iter += 1
    return Q
