"""
Basic Idea : The Tower of Hanoi problem will be solved. For this three pins (1,2 and 3) and two disks (A and B) are used.
Disk A is larger than B. The goal is to move the two disks to pin 3 such that the larger disk A is at the bottom and the
smaller disk B is at the top.

Markov Decision Process will be used to solve the problem. For this we define different rewards (first draft):
    - Reaching the goal : 100
    - Placing a larger disk on top of a smaller disk : - 10 (Penalty)
    - Every other step : -1 (force the agent to solve the problem with minimal steps)

The rules of the game are:
    - Only on disk may be moved at a time
    - Only the most upper disk from one of the rods can be moved
   [- It can only be put on another rod, if the rod is empty or a larger disk is on it]

Addition:
    The agent can make mistakes. When moving a disk from i to j, the agent may actually put the disk on pin k where
    k != i and k != j. The probability of this mistake is 10 %.

MDP:
    The general idea of a MDP is, that you model a problem using states and actions. Actions can move the agent from
    one state to another. With every action a probability is associated that the action actually yields the desired result.
    Furthermore a reward is defined for every transition from one state to another. To work with possible future state a
    discount factor is implemented, that states how important "future rewards" are compared to current rewards.

    The discount factor of future rewards is 0.9
    The problem will be solved using both Value Iteration and Policy Iteration.
"""
import logging
from copy import deepcopy

logging.basicConfig(level=logging.DEBUG)


def gen_actions(state, disk=None):
    moves = []
    for idx_out, pin in enumerate(state):
        for idx, move_to in enumerate(state):
            if pin != move_to and len(pin) != 0 and (disk is None or pin[-1] == disk):
                new_move = deepcopy(state)
                del new_move[idx_out][-1]
                new_move[idx].append(pin[-1])
                moves.append(new_move)
    return moves


def get_disk(state, action):
    for i in range(len(state)):
        if state[i] != action[i]:
            d = set(state[i]).symmetric_difference(set(action[i]))
            logging.debug("Moving disk {}".format(d))
            return d.pop()


def r(state, action):
    for pin in action:
        if not all(pin[i] >= pin[i + 1] for i in range(len(pin) - 1)):
            return -10
    else:
        if action[-1] == [2, 1]:  # bad hardcoded win condition, fix later (adjust field size)
            return 100
        else:
            return -1


def t(state, action):
    disk = get_disk(state, action)
    transitions = []
    actions = gen_actions(state, disk)
    for g_a in actions:
        p = 0.9 if g_a == action else 0.1/(len(actions)-1)
        transitions.append((g_a, p))
    return transitions


# generate all states - by hand, because life sucks and then you die
states = [
    [[2, 1], [], []],
    [[], [2, 1], []],
    [[], [], [2, 1]],
    [[1, 2], [], []],
    [[], [1, 2], []],
    [[], [], [1, 2]],
    [[1], [2], []],
    [[1], [], [2]],
    [[2], [1], []],
    [[2], [], [1]],
]

for s in states:
    possible_actions = gen_actions(s)
    for a in possible_actions:
        possible_transitions = t(s, a)
        reward = sum([t[1] * r(s, t[0]) for t in possible_transitions])
        utility = reward + 0

        logging.info("Investigating state {} and action {}".format(s, a))
        [logging.debug("p: {}\t r: {}\t s': {}\t".format(t[1], r(s, t[0]), t[0])) for t in possible_transitions]
        logging.info("Utility for action : {}".format(reward))
        logging.info("Reward for action : {}".format(reward))

