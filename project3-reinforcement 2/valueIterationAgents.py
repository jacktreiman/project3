# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import copy

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            values = self.values.copy()
            for j in self.mdp.getStates():
                if self.mdp.isTerminal(j):
                    values[j] = 0
                else:
                    q = -10000000
                    for action in self.mdp.getPossibleActions(j):
                        if self.getQValue(j, action) > q:
                            q = self.getQValue(j, action)
                    values[j] = q
            self.values = values
        

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0
        for nextState, tranProb in self.mdp.getTransitionStatesAndProbs(state,action):
            reward = self.mdp.getReward(state, action, nextState)
            q += tranProb * (reward + self.discount*self.getValue(nextState))
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        topAction = None
        topValue = -1000000
        if len(self.mdp.getPossibleActions(state)) == 0:
            return None
        else:
            for action in self.mdp.getPossibleActions(state):
                if self.getQValue(state, action) > topValue:
                    topValue = self.getQValue(state, action)
                    topAction = action
        return topAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        for i in range(self.iterations):
            values = self.values.copy()
            j = self.mdp.getStates()
            curr = j[i%len(j)]
            if self.mdp.isTerminal(curr):
                values[curr] = 0
            else:
                q = -10000000
                for action in self.mdp.getPossibleActions(curr):
                    if self.getQValue(curr, action) > q:
                        q = self.getQValue(curr, action)
                values[curr] = q
            self.values = values
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # predecessors = {}
        # priQue = util.PriorityQueue()
        # for i in self.mdp.getStates():
        #     if not self.mdp.isTerminal(i):
        #         for j in self.mdp.getStates():
        #             if self.mdp.isTerminal(j):
        #                 print('hi')
        #             else:
        #                 for k in self.mdp.getPossibleActions(j):
        #                     for state, action in self.mdp.getTransitionStatesAndProbs(j, k):
        #                         if state is i:
        #                             predecessors[i] += state
        # print(predecessors) 
        predecessors = {}
        priQue = util.PriorityQueue()
        # for i in self.mdp.getStates():
        #     for j in self.mdp.getStates():
        #         if self.mdp.isTerminal(j):
        #             print('hi')
        #         else:
        #             for k in self.mdp.getPossibleActions(j):
        #                 for state, action in self.mdp.getTransitionStatesAndProbs(j, k):
        #                     if state is i:
        #                         predecessors[i].append(state)

        for i in self.mdp.getStates():
            if not self.mdp.isTerminal(i):
                for k in self.mdp.getPossibleActions(i):
                    for state, prob in self.mdp.getTransitionStatesAndProbs(i,k):
                        if state in predecessors:
                            predecessors[state].add(i)
                        else:
                            predecessors[state] = {i}
        for i in self.mdp.getStates():
            if not self.mdp.isTerminal(i):
                maxAct = self.getPolicy(i)
                maxQ = self.getQValue(i, maxAct)
                diff = abs(self.values[i] - maxQ)
                priQue.push(i, -diff)
        
        for _ in range(self.iterations):
            if priQue.isEmpty():
                return
            else:
                s = priQue.pop()
                values = self.values.copy()
                q = values[s]
                for action in self.mdp.getPossibleActions(s):
                    if self.getQValue(s, action) > q:
                        q = self.getQValue(s, action)
                self.values[s] = q
                for pred in predecessors[s]:
                    maxAct = self.getPolicy(pred)
                    maxQ = self.getQValue(pred, maxAct)
                    diff = abs(self.values[pred] - maxQ)
                    if diff > self.theta:
                        priQue.update(pred, -diff)

        
        print(predecessors)
