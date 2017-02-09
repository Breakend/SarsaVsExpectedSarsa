import numpy as np
from mdp_matrix import GridWorld
from double_sarsa import double_sarsa
from expected_sarsa import expected_sarsa
from double_expected_sarsa import double_expected_sarsa
import matplotlib.pyplot as plt
from sarsa import sarsa


# TODO: change these graphs to be over alpha like in the paper


test_rewards = [[i, j, -1] for i in range(10) for j in range(10)]
test_rewards[2] = [0, 2, 1]
test_rewards[23] = [4,3, 1]
# test_rewards = [[0, 3, 5],
#                 [0, 1, 10]]
print test_rewards
gw = GridWorld(10, test_rewards, terminal_states=[2, 23] )

average_reward_double_sarsa = []
all_rewards_per_episode_double_sarsa = []

average_reward_expected_sarsa = []
all_rewards_per_episode_expected_sarsa = []

average_reward_double_expected_sarsa = []
all_rewards_per_episode_double_expected_sarsa = []

average_reward_sarsa = []
all_rewards_per_episode_sarsa = []

epsilon_values = [.1]
n=20000
alphas = [x for x in np.arange(0.01, 1., .05)]
# import pdb; pdb.set_trace()

number_of_runs = 20

for r in range(number_of_runs):
    for epsilon in epsilon_values:
        for alpha in alphas:
            print(alpha)
            Q, average_reward, max_reward, all_rewards = double_sarsa(gw, n, epsilon=epsilon, alpha=alpha)
            average_reward_double_sarsa.append(average_reward)
            all_rewards_per_episode_double_sarsa.append(all_rewards)
            Q, average_reward, max_reward, all_rewards = expected_sarsa(gw, n, epsilon=epsilon, alpha=alpha)
            average_reward_expected_sarsa.append(average_reward)
            all_rewards_per_episode_expected_sarsa.append(all_rewards)
            Q, average_reward, max_reward, all_rewards = double_expected_sarsa(gw, n, epsilon=epsilon, alpha=alpha)
            average_reward_double_expected_sarsa.append(average_reward)
            all_rewards_per_episode_double_expected_sarsa.append(all_rewards)
            Q, average_reward, max_reward, all_rewards = sarsa(gw, n, epsilon=epsilon)
            average_reward_sarsa.append(average_reward)
            all_rewards_per_episode_sarsa.append(all_rewards)



# TODO: plot all sarsa, expected_sarsa, double_Sarsa
# import pdb; pdb.set_trace()
average_reward_double_sarsa = np.mean(np.split(np.array(average_reward_double_sarsa), number_of_runs), axis=0)
average_reward_expected_sarsa = np.mean(np.split(np.array(average_reward_expected_sarsa), number_of_runs), axis=0)
average_reward_double_expected_sarsa = np.mean(np.split(np.array(average_reward_double_expected_sarsa), number_of_runs), axis=0)
average_reward_sarsa = np.mean(np.split(np.array(average_reward_sarsa), number_of_runs), axis=0)
plt.plot(alphas, average_reward_double_sarsa, label="Double Sarsa")
plt.plot(alphas, average_reward_expected_sarsa, label="Expected Sarsa")
plt.plot(alphas, average_reward_double_expected_sarsa, label="Double Expected Sarsa")
plt.plot(alphas, average_reward_sarsa, label="Sarsa")

plt.ylabel('Average reward')
plt.xlabel('alpha')
ax = plt.gca()
# ax.set_xscale('symlog')
ax.legend(loc='lower center', shadow=True)
plt.show()

# import pdb; pdb.set_trace()

print("Max alpha SARSA: %f" % alphas[np.argmax(average_reward_sarsa)])
print("Max alpha Expected SARSA: %f" % alphas[np.argmax(average_reward_expected_sarsa)])
print("Max alpha Double SARSA: %f" % alphas[np.argmax(average_reward_double_sarsa)])
print("Max alpha Double Expected SARSA: %f" % alphas[np.argmax(average_reward_double_expected_sarsa)])

#
# for x, e in zip(all_rewards_per_episode_double_sarsa, alphas):
#     # import pdb; pdb.set_trace()
#     plt.plot(x, label="e=%s"%e)
#
#     # break
#
# plt.ylabel('Returns per episode')
# plt.xlabel('episode')
#
# ax = plt.gca()
# # ax.set_xscale('symlog')
# ax.legend(loc='lower right', shadow=True)
# plt.show()
#
#
#
# for x, e in zip(all_rewards_per_episode_expected_sarsa, alphas):
#     # import pdb; pdb.set_trace()
#     plt.plot(x, label="e=%s"%e)
#
#     # break
#
# plt.ylabel('Returns per episode')
# plt.xlabel('episode')
#
# ax = plt.gca()
# # ax.set_xscale('symlog')
# ax.legend(loc='lower right', shadow=True)
# plt.show()
#
# for x, e in zip(all_rewards_per_episode_double_expected_sarsa, alphas):
#     # import pdb; pdb.set_trace()
#     plt.plot(x, label="e=%s"%e)
#
#     # break
#
# plt.ylabel('Returns per episode')
# plt.xlabel('episode')
#
# ax = plt.gca()
# # ax.set_xscale('symlog')
# ax.legend(loc='lower right', shadow=True)
# plt.show()
