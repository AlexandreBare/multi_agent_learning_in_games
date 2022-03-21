# Command-line arguments
import argparse

# Arrays
import numpy as np

# Visualization & Graphs
from matplotlib import pyplot as plt

# Openspiel
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
# from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.egt import visualization

import lenientDynamics

OUTPUT_DIR = "outputs/"


def load_game(game_name):
    payoff_matrix_player_0 = None
    payoff_matrix_player_1 = None

    if game_name == 'Biased Rock Paper Scissors':
        payoff_matrix_player_0 = [[0, -0.25, 0.5],
                                  [0.25, 0, -0.05],
                                  [-0.5, 0.05, 0]]
        payoff_matrix_player_1 = [[0, 0.25, -0.5],
                                  [-0.25, 0, 0.05],
                                  [0.5, -0.05, 0]]
        # payoff_matrix_player_1 = [[0, -0.25, 0.5],
        #                           [0.25, 0, -0.05],
        #                           [-0.5, 0.05, 0]]
        actions = ['R', 'P', 'S']
    elif game_name == 'Dispersion Game':
        payoff_matrix_player_0 = [[-1, 1],
                                  [1, -1]]
        payoff_matrix_player_1 = [[-1, 1],
                                  [1, -1]]
        actions = ['A', 'B']
    elif game_name == 'Battle of the Sexes':
        payoff_matrix_player_0 = [[3, 0],
                                  [0, 2]]
        payoff_matrix_player_1 = [[2, 0],
                                  [0, 3]]
        actions = ['O', 'M']
    elif game_name == 'Subsidy Game':
        payoff_matrix_player_0 = [[10, 0],
                                  [11, 12]]
        payoff_matrix_player_1 = [[10, 11],
                                  [0, 12]]
        # payoff_matrix_player_1 = [[10, 0],
        #                           [11, 12]]
        actions = ['S1', 'S2']

    game = pyspiel.create_matrix_game(game_name,
                                      game_name,
                                      actions,
                                      actions,
                                      payoff_matrix_player_0,
                                      payoff_matrix_player_1)
    game = pyspiel.convert_to_turn_based(game)
    return game


def train_agents(agents, env, training_episodes):
    # Train the agents
    for cur_episode in range(training_episodes):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])
        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)

    return agents


def plot_egt_dynamics(game,
                      dynamics_func=dynamics.replicator,
                      temperature=1.,
                      tries=1,
                      lenient=False,
                      save_output=False):
    payoff_tensor = game_payoffs_array(game)
    game_name = game.get_parameters()['game']['name']
    game_name_small = game_name.replace(' ', '_').lower()
    dynamics_func_ = dynamics_func
    if lenient:
        dynamics_func_ = (lambda state_, utility_: dynamics_func(state_, utility_, temperature))

    if game.num_distinct_actions() == 2:
        if lenient:
            dyn = lenientDynamics.MultiPopulationLenientDynamics(tries, payoff_tensor, dynamics_func_)
        else:
            dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics_func_)
        projection = "2x2"
        args_quiver = {"num_points": 12}
        args_streamplot = {"num_points": 50, "linewidth": "velocity", "color": "velocity"}

    elif game.num_distinct_actions() == 3:
        if lenient:
            dyn = lenientDynamics.SinglePopulationLenientDynamics(tries, payoff_tensor, dynamics_func_)
        else:
            dyn = dynamics.SinglePopulationDynamics(payoff_tensor, dynamics_func_)
        projection = "3x3"
        args_quiver = {}
        args_streamplot = {"dt": 0.1, "density": 1.0, "min_length": 0.4, "linewidth": "velocity", "color": "velocity"}

    state = game.new_initial_state()


    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.quiver(dyn, **args_quiver)
    if projection == "2x2":
        plt.xlabel(f"Player 1 playing {state.action_to_string(0)}")
        plt.ylabel(f"Player 2 playing {state.action_to_string(0)}")
    else:
        ax.set_labels([state.action_to_string(i) for i in range(game.num_distinct_actions())])

    if lenient:
        plt.title("Directional Field Plot for " + game_name + "\n(" + f"tries={tries}" + ", " + f"temperature={temperature}" + ")")
    else:
        plt.title("Directional Field Plot for " + game_name)
    plt.show()

    dir = OUTPUT_DIR + game_name_small + "/"
    if save_output:
        if lenient:
            fig.savefig(dir + "directional_field_" + game_name_small + f"_tries={tries}" + "_" + f"temperature={temperature}" + ".png")
        else:
            fig.savefig(dir + "directional_field_" + game_name_small + ".png")


    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.streamplot(dyn, **args_streamplot)
    if projection == "2x2":
        plt.xlabel(f"Player 1 playing {state.action_to_string(0)}")
        plt.ylabel(f"Player 2 playing {state.action_to_string(0)}")
    else:
        ax.set_labels([state.action_to_string(i) for i in range(game.num_distinct_actions())])
    if lenient:
        plt.title("Trajectory for " + game_name + "\n(" + f"tries={tries}" + ", " + f"temperature={temperature}" + ")")
    else:
        plt.title("Trajectory for " + game_name)

    plt.show()

    if save_output:
        if lenient:
            fig.savefig(dir + "trajectory_" + game_name_small + f"_tries={tries}" + "_" + f"temperature={temperature}" + ".png")
        else:
            fig.savefig(dir + "trajectory_" + game_name_small + ".png")

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.quiver(dyn, **args_quiver)
    ax.streamplot(dyn, **args_streamplot)
    if projection == "2x2":
        plt.xlabel(f"Player 1 playing {state.action_to_string(0)}")
        plt.ylabel(f"Player 2 playing {state.action_to_string(0)}")
    else:
        ax.set_labels([state.action_to_string(i) for i in range(game.num_distinct_actions())])
    if lenient:
        plt.title("Dynamics for " + game_name + "\n(" + f"tries={tries}" + ", " + f"temperature={temperature}" + ")")
    else:
        plt.title("Dynamics for " + game_name)

    plt.show()

    if save_output:
        if lenient:
            fig.savefig(
                dir + "dynamics_" + game_name_small + f"_tries={tries}" + "_" + f"temperature={temperature}" + ".png")
        else:
            fig.savefig(dir + "dynamics_" + game_name_small + ".png")


def main(args):
    np.random.seed(42)
    num_players = 2
    game = load_game(args.game_name)
    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]

    # Initial state
    policy = policy_lib.TabularPolicy(game)
    print(policy.states_per_player)
    print("Initial Action Probability Array:\n", policy.action_probability_array)
    print("Nash Conv: ", exploitability.nash_conv(game, policy))
    agents = [
        tabular_qlearner.QLearner(player_id=player_id, num_actions=num_actions)
        for player_id in range(num_players)
    ]

    # Training
    training_episodes = 1000
    agents = train_agents(agents, env, training_episodes)

    # Play Time !
    time_step = env.reset()
    agent_output = agents[0].step(time_step, is_evaluation=True)
    print("Agent 0: ", agent_output)
    time_step = env.step([agent_output.action])
    agent_output = agents[1].step(time_step, is_evaluation=True)
    time_step = env.step([agent_output.action])
    print("Agent 1: ", agent_output)

    # Replicator Dynamics
    plot_egt_dynamics(game, dynamics_func=dynamics.replicator, save_output=args.save_output)

    # Lenient Boltzmann Q Dynamics
    plot_egt_dynamics(game,
                      dynamics_func=dynamics.boltzmannq,
                      temperature=1.,
                      tries=5,
                      lenient=True,
                      save_output=args.save_output)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    # 'Biased Rock Paper Scissors', 'Dispersion Game', 'Battle of the Sexes', 'Subsidy Game'
    args_parser.add_argument('--game_name', help='Name of the game to play',
                             default='Subsidy Game', type=str)
    args_parser.add_argument('--save_output', help='Whether we save the output figures',
                             default=True, type=bool)
    args = args_parser.parse_args()
    main(args)
