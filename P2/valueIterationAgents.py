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

from learningAgents import ValueEstimationAgent

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
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # get states
        listofstates = self.mdp.getStates()
        # iterate through iterations
        for it in range(self.iterations):
            v = util.Counter()
            #iterate through states
            for state in listofstates:
                # start max at -inf
                maxValue = None
                possibleActions = self.mdp.getPossibleActions(state)
                # if its the terminal state, value[state] becomes 0
                if not (self.mdp.isTerminal(state)) :
                    for act in possibleActions:
                        q = self.getQValue(state,act)
                        # if qVal is greater than max, the max becomes qval
                        if (q >= maxValue):
                            maxValue = q
                        # value[state] = max
                        v[state] = maxValue
                # else iterate through actions
                else:
                    v[state] = 0
            # at end of state iteration, self.values = value
            self.values = v

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
        # set q, calculation, and reward to 0
        q = 0
        calculation = 0
        reward = 0
        # get Transition States And Probs for state and action
        tsp = self.mdp.getTransitionStatesAndProbs(state,action)
        # if the length of tsp is not 0
        if (len(tsp) == 0) :
            return 0
        else:
            # iterate through transitions in states
            for transition in tsp:
                calculation = self.values[transition[0]] * self.discount
                reward = self.mdp.getReward(state,action,transition[0])
                q += (reward+calculation) * transition[1]
        return q
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # init values
        nextMove = None
        maxValue = None
        # if length of possible actions is 0 or terminal state 
        if (len(self.mdp.getPossibleActions(state)) == 0) :
            return None
        else:
            # iterate through possible actions
            for act in self.mdp.getPossibleActions(state):
                # q becomes computeQValuesFromVales(state,possible action)
                qVal = self.computeQValueFromValues(state,act)
                # if q is greater than or equal to max, max becomes q and nextMove becomes action
                if (qVal >= maxValue) :
                    nextMove = act
                    maxValue = qVal
        return nextMove
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
