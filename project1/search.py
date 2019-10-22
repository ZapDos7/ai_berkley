# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    fringe = util.Stack()
    explored = []
    fringe.push([(problem.getStartState(), "Stop" , 0)])

    while fringe.isEmpty() == False:
        monopati = fringe.pop()
        elem = monopati[len(monopati)-1]
        elem = elem[0]
        if problem.isGoalState(elem):
            return [x[1] for x in monopati][1:]

        if elem not in explored:
            explored.append(elem)
            for successor in problem.getSuccessors(elem):
                if successor[0] not in explored:
                    succPath = monopati[:]
                    succPath.append(successor)
                    fringe.push(succPath)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    
    fringe = util.Queue()
    explored = []
    fringe.push([(problem.getStartState(), "Stop" , 0)])

    while fringe.isEmpty() == False:
        monopati = fringe.pop()
        elem = monopati[len(monopati)-1]
        elem = elem[0]
        if problem.isGoalState(elem):
            return [x[1] for x in monopati][1:]

        if elem not in explored:
            explored.append(elem)
            for successor in problem.getSuccessors(elem):
                if successor[0] not in explored:
                    succPath = monopati[:]
                    succPath.append(successor)
                    fringe.push(succPath)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    fringe = util.PriorityQueue()
    visited = dict()
    state = problem.getStartState()
    node = {}
    node["parent"] = None
    node["action"] = None
    node["state"] = state
    node["cost"] = 0
    fringe.push(node, node["cost"])

    while not fringe.isEmpty():
        node = fringe.pop()
        state = node["state"]
        cost = node["cost"]

        if visited.has_key(state):
            continue

        visited[state] = True
        if problem.isGoalState(state) == True:
            break

        for child in problem.getSuccessors(state):
            if not visited.has_key(child[0]):
                sub_node = {}
                sub_node["parent"] = node
                sub_node["state"] = child[0]
                sub_node["action"] = child[1]
                sub_node["cost"] = child[2] + cost
                fringe.push(sub_node, sub_node["cost"])

    actions = []
    while node["action"] != None:
        actions.insert(0, node["action"])
        node = node["parent"]

    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    fringe = util.PriorityQueue()
    visited = dict()

    state = problem.getStartState()
    node = {}
    node["parent"] = None
    node["action"] = None
    node["state"] = state
    node["cost"] = 0
    node["eval"] = heuristic(state, problem)
    # A* use f(n) = g(n) + h(n)
    fringe.push(node, node["cost"] + node["eval"])

    while not fringe.isEmpty():
        node = fringe.pop()
        state = node["state"]
        cost = node["cost"]
        v = node["eval"]
        #print state

        if visited.has_key(state):
            continue

        visited[state] = True
        if problem.isGoalState(state) == True:
            break

        for child in problem.getSuccessors(state):
            if not visited.has_key(child[0]):
                sub_node = {}
                sub_node["parent"] = node
                sub_node["state"] = child[0]
                sub_node["action"] = child[1]
                sub_node["cost"] = child[2] + cost
                sub_node["eval"] = heuristic(sub_node["state"], problem)
                fringe.push(sub_node, sub_node["cost"] + node["eval"])

    actions = []
    while node["action"] != None:
        actions.insert(0, node["action"])
        node = node["parent"]
    
    return actions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
