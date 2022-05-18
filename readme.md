# Multi-Agent Learning in Games
In this file we will walk-through the task and explain what kind of structure and files that are needed to get the result in our report. 

**For the notebooks in our submission, since we developped the notebooks on Google Colab (Pro version for GPU support and 25GB RAM) it is recommended to upload the notebooks to colab if re-run is required for the best compatibility**

### Task2 - Learning & Dynamics:
To get the graphs in the report, simply run the following command to visualize them. Set save_output to True if graph files are required.

`python task2.py --game_name 'Dispersion Game' --save_output False`

`--game_name`: 
- `'Biased Rock Paper Scissors'`, 
- `'Dispersion Game'`, 
- `'Battle of the Sexes'`, 
- `'Subsidy Game'`

### Task 3 - Kuhn poker
In this task, we separate our code into 2 notebooks:

- task_3_deep_cfr2.ipynb: contains the deep CFR implementation (graphs included). The output are not cleared so if new results are required then re-run the whole notebook. 
- task3_DQN_PG.ipynb: contains the Deep Q network and Policy Gradient implementation. To adjust between algorithms, replace the variable **rl_algo** in section **Train 2 agents to play Kuhn poker** then re-run the notebook to train and visualize the agents' performance.

### Task 4: FCPA poker
Similar to the previous task, we separate our code into 2 notebooks:

- task_4_deep_cfr2.ipynb: contains the deep CFR implementation (graphs included). The output are not cleared so if new results are required then re-run the whole notebook. The policy network checkpoints are stored in **/tmp** folder
- task4_DQN_PG.ipynb: contains the Deep Q network and Policy Gradient implementation. To adjust between algorithms, replace the variable **rl_algo** in section **Train 2 agents to play FCPA poker** then re-run the notebook to train and visualize the agents' performance.


Futhermore, this task require submission so we also have other files that are dedicated for tournament:
- Folder cfr_chkpt: contains the policy network checkpoints for our FCPA bot
- fcpa.py: contains the fcpa agent that loads the checkpoints from the above folder for tournament. 
- task4_PG_gen_agent_tf1.py: the old version of our submission that using Policy Gradient and tf1 (**discarded**). But if required, run this file to train and store the best win-rate agent.
