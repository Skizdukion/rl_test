import numpy as np
from abc import ABC, abstractmethod


class Game(ABC):
    def __init__(self, row_count: int, column_count: int):
        self.row_count = row_count
        self.column_count = column_count
        self.action_size = self.row_count * self.column_count

    @abstractmethod
    def get_initial_state(self):
        """Returns the initial state of the game"""
        pass

    @abstractmethod
    def get_next_state(self, state: np.ndarray, action: np.uint8 | None, player: int):
        """Returns the next state after an action"""
        pass

    def get_state_size(self):
        return self.row_count * self.column_count

    @abstractmethod
    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        """Returns a binary array representing valid moves"""
        pass

    @abstractmethod
    def check_win(self, state: np.ndarray, action: np.uint8) -> bool:
        """Checks if a move has won the game"""
        pass

    @abstractmethod
    def get_value_and_terminated(self, state: np.ndarray, action: np.uint8):
        """Returns the value and if the game is terminated"""
        pass

    @abstractmethod
    def reserve_state(self, state: np.ndarray) -> np.ndarray:
        """Reserves or flips the state (if needed for the game logic)"""
        pass

    # @abstractmethod
    # def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
    #     """Encodes the state into a format suitable for a neural network"""
    #     pass


class Caro(Game):
    def __init__(self):
        super().__init__(7, 7)

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(
        self, state: np.ndarray, action: np.uint8 | None, player: int = 1
    ):
        if action is None:
            return np.copy(state)
        _state = np.copy(state)
        row = action // self.column_count
        column = action % self.column_count
        _state[row, column] = player
        return _state

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False

        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        win_condition = 5

        def count_consecutive(row_offset, col_offset):
            """Count consecutive stones of the current player in a given direction."""
            count = 0
            r, c = row, column
            # Move in the positive direction (offset)
            while (
                0 <= r < self.row_count
                and 0 <= c < self.column_count
                and state[r, c] == player
            ):
                count += 1
                r += row_offset
                c += col_offset

            r, c = row - row_offset, column - col_offset
            # Move in the negative direction (opposite offset)
            while (
                0 <= r < self.row_count
                and 0 <= c < self.column_count
                and state[r, c] == player
            ):
                count += 1
                r -= row_offset
                c -= col_offset

            return count

        if (
            count_consecutive(0, 1) >= win_condition  # Horizontal
            or count_consecutive(1, 0) >= win_condition  # Vertical
            or count_consecutive(1, 1) >= win_condition  # Major diagonal (\)
            or count_consecutive(1, -1) >= win_condition
        ):  # Minor diagonal (/)
            return True

        return False

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def reserve_state(self, state):
        return np.copy(state * -1)

    def get_encoded_state(self, state):
        encoded_state = np.stack((state == -1, state == 0, state == 1)).astype(
            np.float32
        )

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state


class TicTacToe(Game):
    def __init__(self):
        super().__init__(3, 3)

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(
        self, state: np.ndarray, action: np.uint8 | None, player: int = 1
    ):
        if action is None:
            return np.copy(state)
        _state = np.copy(state)
        row = action // self.column_count
        column = action % self.column_count
        _state[row, column] = player
        return _state

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False

        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def reserve_state(self, state):
        return np.copy(state * -1)

    # def get_encoded_state(self, state):
    #     encoded_state = np.stack((state == -1, state == 0, state == 1)).astype(
    #         np.float32
    #     )

    #     if len(state.shape) == 3:
    #         encoded_state = np.swapaxes(encoded_state, 0, 1)

    #     return encoded_state

    def get_encoded_single_state(self, state):
        return state.reshape(-1)

    def get_encoded_states(self, states):
        shape = states.shape
        return states.reshape(shape[0], -1)
