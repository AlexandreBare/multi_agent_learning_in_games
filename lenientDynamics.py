import numpy as np
from open_spiel.python.egt.dynamics import SinglePopulationDynamics
from open_spiel.python.egt.dynamics import MultiPopulationDynamics


def lenient_utility(tries, state, payoff_matrix):
    utility = np.zeros(payoff_matrix.shape[0])
    for i in range(payoff_matrix.shape[0]):
        for j in range(payoff_matrix.shape[1]):
            term1 = 0
            for k in range(payoff_matrix.shape[1]):
                if payoff_matrix[i, k] <= payoff_matrix[i, j]:
                    term1 += state[k]
            term1 **= tries

            term2 = 0
            for k in range(payoff_matrix.shape[1]):
                if payoff_matrix[i, k] < payoff_matrix[i, j]:
                    term2 += state[k]
            term2 **= tries

            term3 = 0
            for k in range(payoff_matrix.shape[1]):
                if payoff_matrix[i, k] == payoff_matrix[i, j]:
                    term3 += state[k]

            utility[i] += payoff_matrix[i, j] * state[j] * (term1 - term2) / term3
    # print("Utility:\n", utility)
    return utility


class SinglePopulationLenientDynamics(SinglePopulationDynamics):
    """Continuous-time single population lenient dynamics.
      Attributes:
        tries: The number of consecutive tries for each action from which the
          rewards are collected before an update
        payoff_matrix: The payoff matrix as an `numpy.ndarray` of shape `[2, k_1,
          k_2]`, where `k_1` is the number of strategies of the first player and
          `k_2` for the second player. The game is assumed to be symmetric.
        dynamics: A callback function that returns the time-derivative of the
          population state.

    """

    def __init__(self, tries, payoff_matrix, dynamics):
        """Initializes the single-population dynamics."""
        assert tries >= 1
        super().__init__(payoff_matrix, dynamics)
        self.tries = tries

    def __call__(self, state=None, time=None):
        """Time derivative of the population state.
        Args:
          state: Probability distribution as list or
            `numpy.ndarray(shape=num_strategies)`.
          time: Time is ignored (time-invariant dynamics). Including the argument in
            the function signature supports numerical integration via e.g.
            `scipy.integrate.odeint` which requires that the callback function has
            at least two arguments (state and time).
        Returns:
          Time derivative of the population state as
          `numpy.ndarray(shape=num_strategies)`.
        """
        state = np.array(state)
        assert state.ndim == 1
        assert state.shape[0] == self.payoff_matrix.shape[0]
        # (Ax')' = xA'

        utility = lenient_utility(self.tries, state, self.payoff_matrix)
        # print(utility)
        return self.dynamics(state, utility)


class MultiPopulationLenientDynamics(MultiPopulationDynamics):
    """Continuous-time multi-population lenient dynamics.
    Attributes:
    tries: The number of consecutive tries for each action from which the
      rewards are collected before an update
    payoff_tensor: The payoff tensor as an numpy.ndarray of size `[n, k0, k1,
      k2, ...]`, where n is the number of players and `k0` is the number of
      strategies of the first player, `k1` of the second player and so forth.
    dynamics: List of callback functions for the time-derivative of the
      population states, where `dynamics[i]` computes the time-derivative of the
      i-th player's population state. If at construction, only a single callback
      function is provided, the same function is used for all populations.
    """

    def __init__(self, tries, payoff_tensor, dynamics):
        """Initializes the multi-population dynamics."""
        assert tries >= 1
        super().__init__(payoff_tensor, dynamics)
        self.tries = tries

    def __call__(self, state, time=None):
        """Time derivative of the population states.
        Args:
          state: Combined population state for all populations as a list or flat
            `numpy.ndarray` (ndim=1). Probability distributions are concatenated in
            order of the players.
          time: Time is ignored (time-invariant dynamics). Including the argument in
            the function signature supports numerical integration via e.g.
            `scipy.integrate.odeint` which requires that the callback function has
            at least two arguments (state and time).
        Returns:
          Time derivative of the combined population state as `numpy.ndarray`.
        """

        state = np.array(state)
        n = self.payoff_tensor.shape[0]  # number of players
        ks = self.payoff_tensor.shape[1:]  # number of strategies for each player
        assert state.shape[0] == sum(ks)

        states = np.split(state, np.cumsum(ks)[:-1])
        dstates = [None] * n
        for i in range(n):
            payoff_matrix = np.moveaxis(self.payoff_tensor[i], i, 0)
            utility = lenient_utility(self.tries, states[i], payoff_matrix)

            dstates[i] = self.dynamics[i](states[i], utility)

        return np.concatenate(dstates)