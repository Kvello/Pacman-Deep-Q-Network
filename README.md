# Pacman DQN with reduced observability
This project was developed as part of the **ESP3201 Machine Learning in Robotics and Engineering** at NUS, fall 2023.

This project was forked from https://github.com/applied-ai-collective/Pacman-Deep-Q-Network

## :nut_and_bolt:Requires:
	* python 3.5
	* pytorch 0.4
## :exclamation: Important files:
	* pacmanDQN_Agents.py 
	* architectures/*
	* curriculumLearning.py
	* environment.py


## :wrench: Setup
### :hourglass: To train the DQN agen, launch:
```bash
	python3 pacman.py -p <name of pacman class> -n <num games> -x <num training> -e <environment configuration> -a <agent config>
```
### :chart_with_upwards_trend: To test the a DQN agent, launch:
```bash
	python3 pacman.py -p <Name of agent class> -a <config file for agent> -e <config file for env> -n num_runs
```
## :hammer: The config file
The main way of configuring the project is through configuration files

The environment is configured as either an evolving environment(curriculum learning)
or as a static environment.
Thigs that can be configured include:
* Layout
* Random starting positions
* Number of ghosts(only smaller or equal to a max number configured by the layout)
* Curriculum learning
	* Stepping criteria
	* Nuber of evaluation games
	* Evaluation frequency
	* Metric (win rate or score)
Examples for both curriculum learning and regular (usefull for testing) are provided

The agents can be configured (mainly) as either feedforward networks or LSTM
Support for Conv-net is also included but not the focus of THIS code.

"Name of agent class" can be one of
* PacmanDQN (full obs space, needs convolutional network etc. - not relevant)
* PacmanPOMDPDQN (used for feed forward networks)
* PacmanLSTMPOMDP (used for LSTM DQN)

-x (Number of training rounds) and -n (Number of games) is not relevant for Curriculum learning

## :memo: Saving results
The config file for the model has an option to save the model. If set during curriculum learning it will save the model every time it gets a new best (in terms of the metric defined) on a particular environment. Setting this option also saves a .npy file of the mean Q values during training.

The config file for the environment includes an option for saving performance. This saves a pickle file dump of the score performance during training. This gets saved uppon steping environments in curriculum learning, or on exit for normal mode.

A script plotter.py for plotting the results is provided for your convinience
It is located in LearningResults.
To run it use
```bash
python3 plotter.py --file <path_to_file>
```
Other options can specify plot title, whetherto save the file, and whether to show it.
For more info do
```bash
python3 plotter.py --help
``` 
### Cheatsheat:
```bash
python3 pacman.py -p PacmanPOMDPDQN -e env.config -a feedForwardDQN.config -c --timeout 20 -n 5
python3 pacman.py -p PacmanLSTMPOMDP -e env.config -a LSTMDQN.config -c --timeout 20 -n 5
python3 pacman.py -p PacmanLSTMPOMDP -e gradualEvolvingEnv.config -a LSTMDQN.config -c --timeout 20
```

I used the Pacman game engine provided by the UC Berkley Intro to AI project:
http://ai.berkeley.edu/reinforcement.html
