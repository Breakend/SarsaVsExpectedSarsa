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

    n_iter = 0
    epsilon = 0.1

    while n_iter < 100000: # TODO: Pick a finish condition for episode
        s = 0 # Initialize s, starting state

        # With prob epsilon, pick a random action
        if np.random.random_sample() <= epsilon:
            a = np.random.random_integers(0, mdp.A-1)
        else:
            # import pdb; pdb.set_trace()
            a = np.argmax(np.mean([Q_a[s][:], Q_b[s][:]], axis=0))

        r = 0

        while not mdp.is_terminal(s): # TODO: Finish episode/trajectory on terminal state
            # Observe S and R
            s_new = np.argmax(mdp.T[s, a, :]) # TODO: Change to stochastic ?
            r = mdp.R[s_new]
            T_new = np.zeros((mdp.S, mdp.S))

            # Pick new action A' from S'
            if np.random.random_sample() <= epsilon:
                a_new = np.random.random_integers(0, mdp.A-1)
            else:
                a_new = np.argmax(np.mean([Q_a[s_new][:], Q_b[s_new][:]], axis=0))

            Q_a[s][a] = Q_a[s][a] + alpha*(r + gamma*Q_b[s_new][a_new] - Q_a[s][a])
            s = s_new
            a = a_new

            n_iter += 1

            # With some probability .5, swap Q_a and Q_b when performing updates.
            if np.random.random_sample() <= .5:
                tmp = Q_a
                Q_a = Q_b
                Q_b = tmp

    return np.mean([Q_a, Q_b], axis=0)
