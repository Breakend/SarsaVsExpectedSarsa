import numpy as np

def double_sarsa(mdp, alpha = 0.1, gamma = 0.9):
    """
    A simple implementation of double sarsa
    Ganger, Michael, Ethan Duryea, and Wei Hu.
    "Double Sarsa and Double Expected Sarsa with Shallow and Deep Learning."
    Journal of Data Analysis and Information Processing 4.04 (2016): 159.
    """
    # Initialization Q_a, Q_b arbitrarily
    Q_a = [[0 for i in range(mdp.A)] for j in range(mdp.S)]
    Q_b = [[0 for i in range(mdp.A)] for j in range(mdp.S)]

    # generate an arbitrary policy
    pol = [1]*mdp.S

    old_Q = Q

    n_iter = 0
    epsilon = 0.1

    while n_iter < 10000000: # TODO: Pick a finish condition for episode
        s = 0 # Initialize s, starting state

        # With prob epsilon, pick a random action
        if np.random.random_sample() <= epsilon:
            a = np.random.random_integers(0, mdp.A-1)
        else:
            a = np.argmax(Q[s][:])

        r = 0

        while s is not mdp.is_terminal(s): # TODO: Finish episode/trajectory on terminal state
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
