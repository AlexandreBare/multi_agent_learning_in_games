"""Policy gradient agents trained and evaluated on Kuhn Poker."""


from absl import app
from absl import flags
from absl import logging

import pandas as pd
import numpy as np
from statistics import mean, stdev

from matplotlib import pyplot as plt

import tensorflow.compat.v1 as tf

# from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import policy_gradient
from open_spiel.python.algorithms import dqn
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel

SEED = 1234


def load_agent(agent_type, game, player_id, rng):
  """Return a bot based on the agent type."""
  if agent_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  elif agent_type == "human":
    return human.HumanBot()
  elif agent_type == "check_call":
    policy = pyspiel.PreferredActionPolicy([1, 0])
    return pyspiel.make_policy_bot(game, player_id, SEED, policy)
  elif agent_type == "fold":
    policy = pyspiel.PreferredActionPolicy([0, 1])
    return pyspiel.make_policy_bot(game, player_id, SEED, policy)
  elif agent_type == "50/50_call_fold":
    policy = pyspiel.PreferredActionPolicy([0.5, 0.5])
    return pyspiel.make_policy_bot(game, player_id, SEED, policy)
  elif agent_type == "pot_sized_bet":
    policy = pyspiel.PreferredActionPolicy([0, 0, 1])
    return pyspiel.make_policy_bot(game, player_id, SEED, policy)
  elif agent_type == "all_in":
    policy = pyspiel.PreferredActionPolicy([0, 0, 0, 1])
    return pyspiel.make_policy_bot(game, player_id, SEED, policy)
  else:
    raise RuntimeError("Unrecognized agent type: {}".format(agent_type))

def get_all_bot_types():
  return ['random', 
             'check_call', 
             'fold', 
            #  '50/50_call_fold',
             'pot_sized_bet', 
             'all_in']

def eval_agent0(env, rl_agent, eval_eps, bot_type='random'):
    agent_reward = []
    wins = 0
    for bot_type1 in get_all_bot_types():
      bot_agent = load_agent(bot_type1, env.game, 1, np.random.RandomState(SEED))
      for i in range(eval_eps):
        eval_agents = [rl_agent, bot_agent]
        time_step = env.reset()
        while not time_step.last():
          player_id = time_step.observations["current_player"]
          if player_id == 0:
            agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
            time_step = env.step([agent_output.action])
          else:
            action = eval_agents[player_id].step(env.get_state)
            time_step = env.step([action])

        if time_step.rewards[0] > time_step.rewards[1]:
          wins += 1
        agent_reward.append(time_step.rewards[0])

    return mean(agent_reward), 1.0*wins/(eval_eps * len(get_all_bot_types()))

def eval_agent1(env, rl_agent, eval_eps, bot_type='random'):
    agent_reward = []
    wins = 0
    
    for bot_type1 in get_all_bot_types():
      bot_agent = load_agent(bot_type1, env.game, 0, np.random.RandomState(SEED))
      for i in range(eval_eps):
        eval_agents = [bot_agent, rl_agent]
        time_step = env.reset()
        while not time_step.last():
          player_id = time_step.observations["current_player"]
          if player_id == 1:
            agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
            time_step = env.step([agent_output.action])
          else:
            action = eval_agents[player_id].step(env.get_state)
            time_step = env.step([action])

        if time_step.rewards[1] > time_step.rewards[0]:
          wins += 1
        agent_reward.append(time_step.rewards[1])

    return mean(agent_reward), 1.0*wins/(eval_eps * len(get_all_bot_types()))

def eval_with_bot_agent0(env, rl_agent, eval_eps, bot_type='random'):
    agent_reward = []
    wins = 0
    bot_agent = load_agent(bot_type, env.game, 1, np.random.RandomState(SEED))
    for i in range(eval_eps):
      eval_agents = [rl_agent, bot_agent]
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if player_id == 0:
          agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
          time_step = env.step([agent_output.action])
        else:
          action = eval_agents[player_id].step(env.get_state)
          time_step = env.step([action])

      if time_step.rewards[0] > time_step.rewards[1]:
        wins += 1
      agent_reward.append(time_step.rewards[0])

    return mean(agent_reward), 1.0*wins/(eval_eps)

def eval_with_bot_agent1(env, rl_agent, eval_eps, bot_type='random'):
    agent_reward = []
    wins = 0
  
    bot_agent = load_agent(bot_type, env.game, 0, np.random.RandomState(SEED))
    for i in range(eval_eps):
      eval_agents = [bot_agent, rl_agent]
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if player_id == 1:
          agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
          time_step = env.step([agent_output.action])
        else:
          action = eval_agents[player_id].step(env.get_state)
          time_step = env.step([action])

      if time_step.rewards[1] > time_step.rewards[0]:
        wins += 1
      agent_reward.append(time_step.rewards[1])

    return mean(agent_reward), 1.0*wins/(eval_eps)


num_episodes = int(10e5)
save_every = int(1e3)

num_players = 2
# Create the game
fcpa_game_string = (
    "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
    "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
    "stack=20000 20000,bettingAbstraction=fcpa)")

game = pyspiel.load_game(fcpa_game_string)

rl_algo = "PolicyGradient" #"PolicyGradient" #"DQN"

checkpoint_dir= "/home/capu/vscode_projects/content/task4_agents2"

env = rl_environment.Environment(game)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]
print(info_state_size)
print(num_actions)

sess = tf.Session()
# with tf.Session() as sess:
  # pylint: disable=g-complex-comprehension
loss_str = ""
if rl_algo == "PolicyGradient":
    loss_str = "a2c" # "a2c"
    agents = [
        policy_gradient.PolicyGradient(
            sess,
            idx,
            info_state_size,
            num_actions,
            loss_str=loss_str,
            max_global_gradient_norm=5.0,
            # critic_learning_rate=0.1,
            # pi_learning_rate=0.01,
            # optimizer_str="adam",
            batch_size=16,
            hidden_layers_sizes=(128, 64)) for idx in range(num_players)
    ]
    
elif rl_algo == "DQN":
    loss_str = "mse" # "huber"
    agents = [
        dqn.DQN(
            sess,
            player_id=idx,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            loss_str=loss_str, 
            hidden_layers_sizes=[128, 32],
            optimizer_str="adam",
            learning_rate=0.01,
            epsilon_decay_duration=num_episodes, #should set to total_eps or total_eps/(2/3/4/5)
            replay_buffer_capacity=10000) for idx in range(num_players)
    ]


sess.run(tf.global_variables_initializer())

losses_list = []
rewards_list = []
p_win_list = []
best_losses = []
eval_episodes = 200

max_p_win_0 = 0.0
max_reward_0 = 0.0

max_p_win_1 = 0.0
max_reward_1 = 0.0

for ep in range(num_episodes):

  if ((ep + 1) % int(save_every) == 0):
    losses = [agent.loss for agent in agents]
    losses_list.append(losses)

    avg_reward0, p_win0 = eval_agent0(env, agents[0], eval_episodes)
    avg_reward1, p_win1 = eval_agent1(env, agents[1], eval_episodes)
    rewards_list.append((avg_reward0, avg_reward1))
    p_win_list.append((p_win0, p_win1))

    # if not best_losses:
    #   # initialize best_losses
    #   best_losses = [loss + 1 if (not isinstance(loss, (list, tuple))) else [l + 1 for l in loss] for loss in losses]

    # if (np.array(losses) < np.array(best_losses)).sum() == len(losses): 
    #   # if both losses have improved,
    #   # we are probably getting closer to a nash equilibrium where both 
    #   # agents' strategy have to be optimal at the same time
    #   best_losses = losses
    #   # Let's save the current agents
    #   for agent in agents:
    #     agent.save(checkpoint_dir)

    if max_p_win_0 < p_win0:
      max_p_win_0 = p_win0
      max_reward_0 = avg_reward0
      agents[0].save(checkpoint_dir)

    if max_p_win_0 == p_win0 and max_reward_0 < avg_reward0:
      max_p_win_0 = p_win0
      max_reward_0 = avg_reward0
      agents[0].save(checkpoint_dir)

    if max_p_win_1 < p_win1:
      max_p_win_1 = p_win1
      max_reward_1 = avg_reward1
      agents[1].save(checkpoint_dir)

    if max_p_win_1 == p_win1 and max_reward_1 < avg_reward1:
      max_p_win_1 = p_win1
      max_reward_1 = avg_reward1
      agents[1].save(checkpoint_dir)

    msg = "-" * 80 + "\n"
    msg += "ep:{}:\n losses={}\n avg_reward0={}\n avg_reward1={}\n p-win0={}\n p-win1={}\n ".format(ep + 1, losses, avg_reward0, avg_reward1, p_win0, p_win1)
    print(msg)

  time_step = env.reset()
  while not time_step.last():
    player_id = time_step.observations["current_player"]
    agent_output = agents[player_id].step(time_step)
    action_list = [agent_output.action]
    time_step = env.step(action_list)

  # Episode is over, step all agents with final info state.
  for agent in agents:
    agent.step(time_step)






