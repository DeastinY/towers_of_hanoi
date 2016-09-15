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

from copy import deepcopy


class MDP:
    def __init__(self, states, actions, transaction_model, reward_function, initial_state):
        self.states = states
        self.actions = actions
        self.transaction_model = transaction_model
        self.reward_function = reward_function
        self.initial_state = initial_state


def gen_moves(s):
    moves = []
    for idx_out, pin in enumerate(hanoi):
        for idx, move_to in enumerate(hanoi):
            if pin != move_to and len(pin) != 0:
                new_move = deepcopy(hanoi)
                del new_move[idx_out][-1]
                new_move[idx].append(pin[-1])
                moves.append(new_move)
    return moves


def reward(state):
    for pin in state:
        if not all(state[i] >= state[i+1] for i in range(len(pin)-1)):
            return -10
    else:
        if state[-1] == [2, 1]:  # bad hardcoded win condition, fix later (adjust field size)
            return 100
        else:
            return -1


hanoi = [[2, 1], [], []]  # List represents the pins. For every pin we keep a list of disk-sizes (last is top)
print(hanoi)
[print("{} : {}".format(p, reward(p))) for p in gen_moves(hanoi)]
states = [
    [[2, 1],]
]

mdp = MDP(states, actions, transaction_model, reward, hanoi)