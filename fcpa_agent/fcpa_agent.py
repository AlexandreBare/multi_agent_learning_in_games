#!/usr/bin/env python3
# encoding: utf-8
"""
fcpa_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2021 KU Leuven. All rights reserved.
"""

import sys
import argparse
import logging
import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots
import os

import tensorflow.compat.v1 as tf

# from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import policy_gradient
from open_spiel.python.algorithms import dqn
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel


logger = logging.getLogger('be.kuleuven.cs.dtai.fcpa')

def get_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(package_directory, 'saved_agents')
    return model_file

def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id)
    return my_player


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play FCPA poker.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        pyspiel.Bot.__init__(self)
        self.player_id = player_id
        self.log = False
        sess = tf.Session()

        info_state_size = 4498
        num_actions = 4

        loss_str = "a2c" # "a2c"
        self.agent = policy_gradient.PolicyGradient(
                        sess,
                        player_id,
                        info_state_size,
                        num_actions,
                        loss_str=loss_str,
                        max_global_gradient_norm=5.0,
                        # critic_learning_rate=0.1,
                        # pi_learning_rate=0.01,
                        # optimizer_str="adam",
                        batch_size=16,
                        hidden_layers_sizes=(128, 64))
        
        sess.run(tf.global_variables_initializer())

        print("\n> exist check point: {}\n".format(self.agent.has_checkpoint(get_path())))

        if self.agent.has_checkpoint(get_path()):
            self.agent.restore(get_path())
        
        # self.agent.restore(get_path())

    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        if self.log:
            print('>restart_at:{}\n'.format(state))

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        if self.log:
            print('>inform_action:\nstate:{}\nplayer_id:{}\naction:{}\n'.format(state, player_id, action))

    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        if len(state.legal_actions(self.player_id)) == 0:
            return pyspiel.INVALID_ACTION

        action, probs = self.agent._act(state.information_state_tensor(), state.legal_actions(self.player_id))
        if self.log:
            print(">step:\n")
            print('available action:{}\n'.format(state.legal_actions(self.player_id)))
            print('chosen action:{}\n'.format(action))
            print('actions probs:{}\n'.format(probs))
        return action


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
    game = pyspiel.load_game(fcpa_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0,1]]
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    print(get_path())
    test_api_calls()

if __name__ == "__main__":
    sys.exit(main())

