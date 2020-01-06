import numpy as np
from matplotlib import pyplot as plt

grid_size = 8
Qs = np.zeros(shape=(grid_size, grid_size, 4))
rewards = np.zeros(shape=(grid_size, grid_size))

initial_state = (7, 0)  # given
goal_state = (1, 5)  # given

rewards[goal_state] = 100

discount_factor = 0.9
lr = 0.5

actions = [0, 1, 2, 3]
action_indices = [(0, -1), (0, 1), (-1, 0), (1, 0)]
NUM_EPISODE = 10000


def init():
    global Qs
    Qs = np.zeros(shape=(grid_size, grid_size, 4))


def epsilon_greedy(s, eps=0.9):
    val = np.random.uniform()
    if val < eps:
        return np.random.randint(low=0, high=len(actions))
    return np.argmax(Qs[s])


def is_valid_action(s, a):
    return (a == 0 and s[1] > 0) or \
           (a == 1 and s[1] < grid_size - 1) or \
           (a == 2 and s[0] > 0) or \
           (a == 3 and s[0] < grid_size - 1)


def take_action(s, a, det=True):
    if det is False:
        val = np.random.uniform()
        if val >= 0.5:
            if a == 0 or a == 1:
                if val <= 0.75:
                    a = 2
                else:
                    a = 3
            elif a == 2 or a == 3:
                if val <= 0.75:
                    a = 0
                else:
                    a = 1

    if is_valid_action(s, a):
        a_i = action_indices[a]
        s = (s[0] + a_i[0], s[1] + a_i[1])

    return s, rewards[s]


def run(det=True):
    for _ in range(NUM_EPISODE):
        s = initial_state
        a = epsilon_greedy(s)
        while s != goal_state:
            s_prime, r = take_action(s, a, det=det)
            if det:
                if s != s_prime:
                    Qs[s][a] = Qs[s][a] + lr * (r + discount_factor * np.max(Qs[s_prime]) - Qs[s][a])
                a = epsilon_greedy(s_prime)
                s = s_prime
            else:
                a_prime = epsilon_greedy(s_prime)
                if s != s_prime:
                    Qs[s][a] = Qs[s][a] + lr * (r + discount_factor * Qs[s_prime][a_prime] - Qs[s][a])
                s = s_prime
                a = a_prime


def visualize():
    max_vals = np.max(Qs, axis=-1)
    max_actions = np.argmax(Qs, axis=-1)

    plt.figure(figsize=(6, 6))
    plt.imshow(max_vals, cmap='Oranges', interpolation='nearest', vmin=0, vmax=100)

    ax = plt.gca()
    ax.set_xticks(np.arange(9) - .5)
    ax.set_yticks(np.arange(9) - .5)
    ax.set_xticklabels(range(9))
    ax.set_yticklabels(range(9, -1, -1))

    action_dict = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}

    for y in range(grid_size-1, -1, -1):
        for x in range(grid_size-1, -1, -1):
            best_action = max_actions[y, x]
            best_value = max_vals[y, x]

            plt.text(x, y, str(int(best_value)), color='black', size=16, verticalalignment='center', horizontalalignment='center',
                     fontweight='bold')

            u, v = action_dict[best_action]
            plt.arrow(x, y, u * .3, -v * .3, color='cyan', head_width=0.12, head_length=0.12)
    plt.show()


run()
visualize()

init()

run(det=False)
visualize()
