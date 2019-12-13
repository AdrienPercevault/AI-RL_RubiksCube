"""
A user-facing interface for using a trained model to solve a cube.
"""
from cube_model import CubeModel
from batch_cube import BatchCube
from typing import List, Optional
from mcts_nn_cube import MCTSAgent, State
from puzzle_cube import PuzzleCube, valid_moves

import numpy as np

### -------------------------- ###
# Puzzle Cube Solver environment #
### -------------------------- ###
def cube_to_initial_mcts_state(cube: PuzzleCube, history: int) -> State:
    blank_history = tuple(None for _ in range(history-1))
    internal_state = (cube._inner_cube,) + blank_history
    print("BONJOUR !!!!!!!!!! {}".format(State(_internal_state=internal_state)))
    return State(_internal_state=internal_state)


class CubeSolver:
    """
    A convinient wrapper around the MCTS solver used to solve the cube.
    """
    def __init__(self, cube: PuzzleCube, model: CubeModel):
        """
        :param cube: The starting puzzle cube.
        :param model: The trained model to use.
        """

        # assert (model._model is not None), "model must be loaded"
        # history_length = model._model.history
        history_length = 8
        blank_history = tuple(None for _ in range(history_length - 1))
        internal_state = (cube._inner_cube,) + blank_history
        initial_state = State(_internal_state=internal_state)

        self._mcts_agent = MCTSAgent(model._function(), initial_state, max_depth=100)

    def solve(self, steps: int, stop_early: bool = True) -> None:
        """
        Run the solver for a certain number of steps.
        :param steps: Number of steps to run the MCTS solver.
        :param stop_early: Whether to stop the search once a solution is found.
        (More steps may find shorter solutions.)  Default is True.
        """
        self._mcts_agent.search(steps, stop_early)

    def solution(self) -> Optional[List[str]]:
        if self._mcts_agent.initial_node.terminal:
            return []
        elif np.any(self._mcts_agent.initial_node.connected_to_terminal):
            node = self._mcts_agent.initial_node
            moves = []
            while not node.terminal:
                best_action = np.argmax(node.action_visit_counts() * node.connected_to_terminal)
                moves.append(valid_moves[best_action])
                node = node.children[best_action]
            return moves
        else:
            return None


### ------------------------###
# OpenAI Gym Cube environment #
### ------------------------###
def gymcube_state_to_puzzle_cube(state: State):

    print("state !!!!!!!!!!! {}".format(state))
    one_face_list = []
    every_face_list = []
    for elem in state:
        one_face_list.append(elem)
        if len(one_face_list) == 9:
            every_face_list.append(one_face_list)
            one_face_list = []

    tmp_list = []
    tmp_list.append(every_face_list[3])
    tmp_list.append(every_face_list[4])
    tmp_list.append(every_face_list[0])
    tmp_list.append(every_face_list[5])
    tmp_list.append(every_face_list[2])
    tmp_list.append(every_face_list[1])

    face_state_list = []
    for face in tmp_list:
        for color in face:
            face_state_list.append(color)

    gym_state_list = face_state_list
    for index, item in enumerate(face_state_list):
        if item == 0:
            gym_state_list[index] = 0
        elif item == 1:
            gym_state_list[index] = 4
        elif item == 2:
            gym_state_list[index] = 1
        elif item == 3:
            gym_state_list[index] = 2
        elif item == 4:
            gym_state_list[index] = 5
        elif item == 5:
            gym_state_list[index] = 3

    print("Modified state !!!!!!!!!!! {}".format([gym_state_list]))

    batch_array = np.repeat([gym_state_list], repeats=1, axis=0)
    # batch_array = np.repeat([state], repeats=1, axis=0)
    bc = BatchCube(cube_array=batch_array)
    pc = PuzzleCube(bc)
    return pc

class OpenAICubeSolver:
    """
    A convinient wrapper around the MCTS solver used to solve the cube.
    """
    valid_moves = ["L", "L'", "R", "R'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]

    def __init__(self, puzzle_state, model: CubeModel):
        """
        :param state: The starting puzzle State.
        :param model: The trained model to use.
        """

        # assert (model._model is not None), "model must be loaded"
        # history_length = model._model.history
        history_length = 8
        blank_history = tuple(None for _ in range(history_length - 1))
        internal_state = (puzzle_state._inner_cube,) + blank_history
        initial_state = State(_internal_state=internal_state)

        self._mcts_agent = MCTSAgent(model._function(), initial_state, max_depth=100)

    def solve(self, steps: int, stop_early: bool = True) -> None:
        """
        Run the solver for a certain number of steps.
        :param steps: Number of steps to run the MCTS solver.
        :param stop_early: Whether to stop the search once a solution is found.
        (More steps may find shorter solutions.)  Default is True.
        """
        print("------ solve -------")
        self._mcts_agent.search(steps, stop_early)
        print(self._mcts_agent.search(steps, stop_early))

    def solution(self) -> Optional[List[str]]:
        print("----- solution -----")
        print(np.any(self._mcts_agent.initial_node.connected_to_terminal))
        if self._mcts_agent.initial_node.terminal:
            print("1")
            return []
        elif np.any(self._mcts_agent.initial_node.connected_to_terminal):
            print("2")
            node = self._mcts_agent.initial_node
            moves = []
            while not node.terminal:
                print("3")
                best_action = np.argmax(node.action_visit_counts() * node.connected_to_terminal)
                moves.append(self.valid_moves[best_action])
                node = node.children[best_action]
            return moves
        else:
            return None
