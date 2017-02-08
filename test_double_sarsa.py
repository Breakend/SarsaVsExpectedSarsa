import numpy as np
from mdp_matrix import GridWorld
from double_sarsa import double_sarsa

test_rewards = [[i, j, -1] for i in range(5) for j in range(5)]
test_rewards[2] = [0, 2, 1]
test_rewards[23] = [4,3, 1]
# test_rewards = [[0, 3, 5],
#                 [0, 1, 10]]
print test_rewards
gw = GridWorld(5, test_rewards, terminal_states=[2, 23] )

Q = double_sarsa(gw)

import pdb; pdb.set_trace()

print np.reshape(np.argmax(Q, 1), (5,5))
