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

Implementation:
    Take care, there are a few hickups:
        1. Actions described as an initial and final state
"""

import logging
import numpy as np
from copy import deepcopy
import itertools
import random
from collections import namedtuple

logging.basicConfig(level=logging.DEBUG)
Action = namedtuple('Action', 'InitialState FinalState')
Transition = namedtuple('Transition', 'Action Probability')


def gen_actions(state, disk=None):
    """Generates all possible actions (=resulting states) from a base state.
    If disk is not None only action that move disk are returned."""
    actions = [Action(state, state)]
    if state[-1] == [2, 1]:
        return actions
    for idx_out, pin in enumerate(state):
        for idx, move_to in enumerate(state):
            if pin != move_to and len(pin) != 0 and (disk is None or pin[-1] == disk):
                new_state = deepcopy(state)
                del new_state[idx_out][-1]
                new_state[idx].append(pin[-1])
                actions.append(Action(state, new_state))
    return actions


def get_disk(action):
    """return the disk that is moved from in the transition from state to action."""
    for i in range(len(action.InitialState)):
        if action.InitialState[i] != action.FinalState[i]:
            d = set(action.InitialState[i]).symmetric_difference(set(action.FinalState[i]))
            return d.pop()


def t(action):
    """Returns the possible transitions trying to execute an action on a state.
    The desired state s' has a 90% success rate. Every different state has an equal probability."""
    if action.InitialState == action.FinalState:
        return([Transition(action, 1.0)])
    disk = get_disk(action)
    transitions = []
    actions = gen_actions(action.InitialState, disk)
    for g_a in actions:
        if g_a.FinalState == action.FinalState: p = 0.9
        elif g_a.FinalState == action.InitialState: p = 0.0
        else: p = 0.1/(len(actions)-2)
        transitions.append(Transition(g_a, p))
    return transitions


def r(action, state=None):
    if not state is None:
        transitions = t(action)
        return sum([tr.Probability * r(tr.Action) for tr in transitions])
    else:
        for pin in action.FinalState:
            if not all(pin[i] >= pin[i + 1] for i in range(len(pin) - 1)):
                return -10
        else:
            if action.FinalState[-1] == [2, 1]:  # bad hardcoded win condition, fix later (adjust field size)
                return 100 if not action.InitialState == action.FinalState else 0
            else:
                return -1


def u(state, swap_utility, updated_utility=None):
    """Calculates or updates the current utility for a state. If no utility is present it is initialized to 0."""
    global best_utility
    for idx, us in enumerate(best_utility):
        if state in us:
            if swap_utility:
                best_utility_swap[idx][1] = updated_utility if updated_utility is not None else best_utility_swap[idx][1]
            else:
                us[1] = updated_utility if updated_utility is not None else us[1]
            return us[1]
    else:
        initial_utility = 0
        best_utility.append([state, initial_utility])
        best_utility_swap.append([state, initial_utility])
        return initial_utility


best_utility = []
best_utility_swap = []

def unique(iterable):
    """Used to adjust the itertool permutations method to our needs.
    http://stackoverflow.com/questions/6534430/why-does-pythons-itertools-permutations-contain-duplicates-when-the-original"""
    seen = set()
    for i in iterable:
        if str(i) in seen:
            continue
        seen.add(str(i))
        yield i


def get_state_id(states, x):
    for i, s in enumerate(states):
        if s == x:
            return i
    else:
        raise Exception("State not in States.")


def value_iteration(epsilon, states, swap_utility = False):
    iterations = 0
    while True:
        iterations += 1
        states_delta, states_best = [], []
        for s in states:
            logging.warning("State : {}".format(s))
            possible_actions = gen_actions(s)
            if len(possible_actions) == 0:
                continue  # skip terminal states
            u_a_mapping = {}
            for a in possible_actions:
                logging.debug("Action : {}".format(a))
                reward = r(a, s)
                [logging.debug("To {} Reward : {} \t Probability : {}".format(tr.Action.FinalState, r(tr.Action), tr.Probability)) for tr in t(a)]
                utility = reward + 0.9 * sum([tr.Probability * u(tr.Action.FinalState, swap_utility) for tr in t(a)])
                logging.debug("Reward : {} \t Utility : {}".format(reward, utility))
                u_a_mapping[utility] = a

            best_u = max(u_a_mapping, key=float)
            best_a = u_a_mapping[best_u]
            u(s, swap_utility, best_u)
            states_delta.append(abs(u(s, swap_utility)-best_u))
            states_best.append((best_a, best_u))
            logging.debug("Best Utility : {} Action : {} ".format(best_u, best_a))

        global best_utility
        global best_utility_swap
        if swap_utility:
            best_utility = []
            for i in range(len(best_utility_swap)):
                best_utility.append(deepcopy(best_utility_swap[i]))

        if all([d < epsilon for d in states_delta]):
            logging.info("Finished after {} iterations!".format(iterations))
            for b in states_best:
                logging.info("For State {} moving to State {} is best, with utility {}".format(b[0].InitialState, b[0].FinalState, b[1]))
            return


class Policy:
    def __init__(self, states):
        self.policy = [[s, gen_actions(s)[-1]] for s in states ]

    def update(self, state, action):
        self.get_policy(state)[1] = action

    def get_action(self, state):
        return self.get_policy(state)[1]

    def get_policy(self, state):
        return [p for p in self.policy if p[0] == state][0]

iterations = 0
def policy_iteration(epsilon, states, policy=None):
    global iterations
    iterations+=1
    policy = Policy(states) if policy is None else policy
    logging.debug("Initial Policy: {}".format(policy))
    leq_a, leq_b = [], []
    for s in states:
        a = policy.get_action(s)
        logging.info("In {} do {}".format(s, a.FinalState))
        reward = r(a, s)
        leq = "LEQ : {} + 0.9 * ( {} )".format(reward, ["{} * u({})".format(tr.Probability, tr.Action.FinalState) for tr in t(a)])
        ids = {}
        for tr in t(a):
            ids[get_state_id(states, tr.Action.FinalState)] = tr.Probability
        part_leq_a = []
        for i in range(len(states)):
            if i in ids:
                if i == len(leq_b):
                    part_leq_a.append(1.0)
                elif i == get_state_id(states, a.FinalState):
                    part_leq_a.append((-0.9 * ids[i]))
                else:
                    part_leq_a.append(-0.9*ids[i])
            else:
                part_leq_a.append(0)
        leq_a.append(part_leq_a)
        leq_b.append(reward)
        logging.debug(leq)
        logging.debug("\n"+str(np.array(leq_a)))
        logging.debug("\n"+str(np.array(leq_b)))
    utility = np.linalg.solve(np.array(leq_a), np.array(leq_b))
    logging.info("Current Utilities:\n"+str(utility))
    for s in states:
        state_id = get_state_id(states, s)
        policy_action = policy.get_action(s)
        other_actions = [a for a in gen_actions(s) if a != policy_action]
        for a in other_actions:
            other_utility = r(a, s) + 0.9 * sum([tr.Probability*utility[get_state_id(states, tr.Action.FinalState)] for tr in t(a)])
            if (other_utility > utility[state_id]):
                utility[state_id] = other_utility
                policy.update(s, a)
                policy_iteration(epsilon, states, policy)
                return
    else:
        logging.info("Finished after {} iterations !".format(iterations))


if __name__ == "__main__":
    states = [i for i in unique(itertools.permutations([[2,1],[],[]]))] + \
             [i for i in unique(itertools.permutations([[2],[1],[]]))] + \
             [i for i in unique(itertools.permutations([[1,2],[],[]]))]

    swap_utility = True  # Used to debug vs hand-calculated results. Only updates utility after one iteration
    #value_iteration(0.000001, states, swap_utility)
    policy_iteration(0.000001, states)


