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
import itertools

logging.basicConfig(level=logging.INFO)


def gen_actions(state, disk=None):
    if state[-1] == [2, 1]:
        return []
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
    if state[-1] == [2, 1]:
        return []
    disk = get_disk(state, action)
    transitions = []
    actions = gen_actions(state, disk)
    for g_a in actions:
        p = 0.9 if g_a == action else 0.1/(len(actions)-1)
        transitions.append((g_a, p))
    return transitions


def u(state, updated_utility=None):
    global best_utility
    for us in best_utility:
        if state in us:
            us[1] = updated_utility if updated_utility is not None else us[1]
            return us[1]
    else:
        initial_utility = 0
        best_utility.append([state, initial_utility])
        return initial_utility


best_utility = []

def unique(iterable):
    # http://stackoverflow.com/questions/6534430/why-does-pythons-itertools-permutations-contain-duplicates-when-the-original
    seen = set()
    for i in iterable:
        if str(i) in seen:
            continue
        seen.add(str(i))
        yield i

# generate all states - by hand, because life sucks and then you die
states = [i for i in unique(itertools.permutations([[2,1],[],[]]))] + \
         [i for i in unique(itertools.permutations([[2],[1],[]]))] + \
         [i for i in unique(itertools.permutations([[1,2],[],[]]))]

def value_iteration(epsilon):
    iterations = 0
    while True:
        iterations += 1
        states_delta = []
        states_best = []
        for s in states:
            possible_actions = gen_actions(s)
            if len(possible_actions) == 0:
                continue  # skip terminal states
            state_utilities = []
            state_moves = []
            for a in possible_actions:
                possible_transitions = t(s, a)
                reward = sum([tr[1] * r(s, tr[0]) for tr in possible_transitions])
                utility = reward + 0.9 * sum([tr[1] * u(tr[0]) for tr in possible_transitions])
                state_utilities.append(utility)
                state_moves.append(a)
                logging.info("Investigating state {} and action {}".format(s, a))
                [logging.debug("p: {}\t r: {}\t s': {}\t".format(
                    tr[1], r(s, tr[0]), tr[0])) for tr in possible_transitions]
                logging.debug("Reward for action : {}".format(reward))
                logging.debug("Utility for action : {}".format(utility))

            idx = state_utilities.index(max(state_utilities))
            new = state_utilities[idx]
            move = state_moves[idx]
            u(s, new)
            logging.info("Updated utility {} to {} for {}".format(u(s), new, s))
            logging.info("Current best move for this state is {}".format(move))
            states_delta.append(abs(u(s)-new))
            states_best.append((s, move, new))
        if all([d < epsilon for d in states_delta]):
            logging.info("Finished after {} iterations!".format(iterations))
            for b in states_best:
                logging.info("For state {} action {} is best with utility {}".format(*b))
            return

value_iteration(0.001)

