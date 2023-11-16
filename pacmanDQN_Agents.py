# import pacman game
from pacmanUtils import *
from parsers import get_optimizer_from_dict, get_loss_from_dict, get_scheduler_from_dict
# import torch library
import torch
import torch.nn as nn
import torch.optim as optim
from architectures.Conv import Conv
from architectures.ResLSTM import ResLSTM, LSTM
from architectures.feedForward import FeedForward
from architectures.utils import ReplayMemory, Transition, EarlyStopper

# import other libraries
import numpy as np
import random
import copy

from typing import Dict, Any, Tuple
from pacmanUtils import GameState

# model parameters


class randomActionProb:
    def __init__(self, type="fixed", initial=1, min:int = 0, args: Dict[str, Any] = {}):
        self.type = type
        self.initial = initial
        self.epsilon = initial
        self.N = 0
        self.min = min
        self.args = args
        self.evaluation = False

    def getProbability(self):
        if self.evaluation:
            return self.min
        if self.type == "fixed":
            return self.initial
        elif self.type == "linear":
            return max(self.min, (self.initial-float(self.N)*float(self.args["eps_step"])))
        elif self.type == "delta":
            if self.N < self.args["before"]:
                return self.initial
            else:
                return self.args["valAfter"]

    def stepN(self):
        self.N += 1
    def setEvaluation(self, evaluation):
        self.evaluation = evaluation
    def reset(self, initial=None):
        self.N = 0
        if initial is not None:
            self.initial = initial
        self.epsilon = self.initial


class PacmanDQN(PacmanUtils):
    def __init__(self,
                 model: Dict[str, Any],
                 obs_size: Tuple[int, int],
                 rand_prob: Dict[str, Any] = {
                     "type": "fixed", "initial": 0.05, "args": {}},
                 num_objecttypes: int = 6,
                 num_actions=4,
                 discount=0.8,
                 training = True,
                 optimizer={
                     "type": "RMSprop",
                     "args": {
                         "lr": 1e-3,
                         "eps": 1e-6,
                         "alpha": 0.95
                     }
                 },
                 loss={"type":"Huber"},
                 replay_buffer_size=100000,
                 batch_size=32,
                 load=False,
                 tau=0.005,
                 path='models/pacmanDQNAgent.pt',
                 early_stopper=None,
                 quiet=False,
                 batch_norm=None,
                 stat_freq=100,
                 start_training=300):
        print("Started PacmanDQN")
        # pytorch parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # init model
        if (load == True):
            self.policy_net = torch.load(
                path, map_location=torch.device(self.device)).to(self.device)
            self.policy_net.device = self.device
            self.target_net = torch.load(
                path, map_location=torch.device(self.device)).to(self.device)
            self.target_net.device = self.device
        else:
            if model["arch"].lower() == "conv":
                obs_size = (num_objecttypes, obs_size[0], obs_size[1])
                self.policy_net = Conv(obs_size=obs_size,
                                          num_actions=num_actions,
                                          **model["args"]).to(self.device)
                self.target_net = Conv(obs_size=obs_size,
                                          num_actions=num_actions,
                                          **model["args"]).to(self.device)
            elif model["arch"].lower() == "resnetlstm":
                obs_size = (num_objecttypes, obs_size[0], obs_size[1])
                self.policy_net = ResLSTM(obs_size=obs_size,
                                             num_actions=num_actions,
                                             **model["args"]).to(self.device)
                self.target_net = ResLSTM(obs_size=obs_size,
                                             in_chns=num_objecttypes,
                                             num_actions=num_actions,
                                             **model["args"]).to(self.device)
            elif model["arch"].lower() == "feedforward":
                obs_size = np.prod(obs_size)*num_objecttypes
                self.policy_net = FeedForward(obs_size=obs_size,
                                                 num_actions=num_actions,
                                                 **model["args"]).to(self.device)
                self.target_net = FeedForward(obs_size=obs_size,
                                                 num_actions=num_actions,
                                                 **model["args"]).to(self.device)
            elif model["arch"].lower() == "lstm":
                obs_size = np.prod(obs_size)*num_objecttypes
                self.policy_net = LSTM(obs_size=obs_size,
                                                num_actions=num_actions,
                                                device=self.device,
                                                **model["args"]).to(self.device)
                self.target_net = LSTM(obs_size=obs_size,
                                                num_actions=num_actions,
                                                device=self.device,
                                                **model["args"]).to(self.device)
            else:
                raise ValueError(
                    "Model {} not recognized".format(model["arch"]))
        print("Model:", self.policy_net)
        self.model = model
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.double()
        self.target_net.double()

        self.obs_size = obs_size
        self.start_training = start_training
        self.RandomProb = randomActionProb(**rand_prob)
        self.gamma = discount
        self.training = training
        self.loss = loss
        self.ReplayMemory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.path = path
        self.cum_rewards = 0
        self.cum_wins = 0
        self.cum_loss = 0
        self.episode_number = 0
        self.win_counter = 0
        self.tau = tau
        self.Q_values = np.array([])
        self.evaluation_batch = []
        self.batch_norm = batch_norm
        self.episode_reward = 0
        self.terminal = False
        self.Q_mean_vals = np.array([])

        if early_stopper is not None:
            self.early_stopper = EarlyStopper(**early_stopper)
        else:
            self.early_stopper = None
        self.quiet = quiet
        self.stat_freq = stat_freq

        self.optimizer = get_optimizer_from_dict(self.policy_net,
                                                 optimizer)

        self.criterion = get_loss_from_dict(self.loss)
        if "scheduler" in optimizer:
            self.scheduler = get_scheduler_from_dict(
                self.optimizer, optimizer["scheduler"])
        else:
            self.scheduler = None

    def getMove(self):  # epsilon greedy
        random_value = np.random.rand()
        if random_value > self.RandomProb.getProbability():  # exploit
            # get current state
            temp_current_state = torch.from_numpy(np.stack(self.current_state))
            temp_current_state = temp_current_state.unsqueeze(0)
            temp_current_state = temp_current_state.to(self.device)

            # get best_action - value between 0 and 3
            self.policy_net.eval()
            with torch.no_grad():
                best_action = torch.argmax(
                    self.policy_net(temp_current_state)).item()
            move = self.get_direction(best_action)

        else:  # explore
            # random value between 0 and 3
            random_value = np.random.randint(0, 4)
            move = self.get_direction(random_value)

        # save last_action
        self.last_action = self.get_value(move)
        return move

    def observation_step(self, state):
        if self.last_action is not None:
            # get state
            self.last_state = np.copy(self.current_state)
            self.last_game_state = copy.deepcopy(self.current_game_state)
            self.current_game_state = state
            self.current_state = self.getObservation(state)

            # get reward
            reward = self.getReward(state)
            if (state.isWin()):
                self.win_counter += 1
                self.cum_wins += 1
            self.episode_reward += reward

            # store transition

            transition = Transition(
                self.last_state, reward, self.last_action, self.current_state, self.terminal)
            self.ReplayMemory.push(transition)

            if self.episode_number < self.start_training:
                self.evaluation_batch.append(self.last_state)
            # train model
            self.train()
    def evaluate_Qestimates(self):
        self.policy_net.eval()
        states = torch.from_numpy(np.stack(self.evaluation_batch))
        Q_values = self.policy_net(states.to(self.device))
        Q_mean, Q_var = torch.mean(Q_values.detach()), torch.var(Q_values.detach())
        self.Q_mean_vals = np.append(self.Q_mean_vals,Q_mean.cpu().numpy())
        return Q_mean, Q_var
    def getReward(self, state):
        return self.updateScore(state)
    def stepEnv(self):
        self.RandomProb.reset(initial=max(0.5,self.RandomProb.getProbability()))
        # self.ReplayMemory.clear()
        self.evaluation_batch = []
        self.episode_number = 0
    def final(self, state):
        # do observation
        if self.episode_number > self.start_training and self.training:
            self.RandomProb.stepN()
        self.terminal = True
        self.observation_step(state)
        self.cum_rewards += self.episode_reward

        if (self.episode_number % self.stat_freq == 0):
            if self.scheduler is not None:
                self.scheduler.step()
            if not self.quiet:
                print("---------------------------------------")
                print("Episode no = " + str(self.episode_number) +
                      "; won: " + str(self.cum_wins))
                print("Avg reward = " + str(self.cum_rewards / self.stat_freq))
                print("Epsillon = " + str(self.RandomProb.getProbability()))
                print("Loss: {}".format(self.cum_loss/self.stat_freq))
                if len(self.evaluation_batch) > 0:
                    Q_mean, Q_std = self.evaluate_Qestimates()
                    print("Q_mean: {}, Q_std: {}".format(Q_mean, Q_std))
            self.cum_wins = 0
            self.Q_values = np.array([])
            self.cum_rewards = 0
            self.cum_loss = 0
    def saveModel(self):
        if not self.quiet:
            print("Saving model")
        torch.save(self.policy_net, self.path)
        np.save("models/Q_mean_vals.npy",self.Q_mean_vals)
    def setTraining(self, training):
        self.RandomProb.setEvaluation(not training)
        self.training = training
    def getObservation(self, state):
        return self.getStateMatrix(state)

    def train(self):
        if (self.episode_number <= self.start_training
                or not self.training):
            return
        self.policy_net.train()
        batch = self.ReplayMemory.sample(self.batch_size)
        batch_s, batch_r, batch_a, batch_n, batch_t = Transition(*zip(*batch))

        # convert from numpy to pytorch
        batch_s = torch.from_numpy(np.stack(batch_s)).to(self.device)
        batch_r = torch.DoubleTensor(batch_r).unsqueeze(1).to(self.device)
        batch_a = torch.LongTensor(batch_a).unsqueeze(1).to(self.device)
        batch_n = torch.from_numpy(np.stack(batch_n)).to(self.device)
        batch_t = torch.BoolTensor(batch_t).unsqueeze(1).to(self.device)
        if self.batch_norm == "Z-score":
            batch_r = (batch_r - batch_r.mean())/batch_r.std()
        elif self.batch_norm == "MinMax":
            batch_r = (batch_r - batch_r.min())/(batch_r.max()-batch_r.min())
        # get Q(s, a)
        state_action_values = self.policy_net(batch_s)
        state_action_values = state_action_values.gather(1, batch_a)

        # get V(s')
        next_state_values = self.target_net(batch_n).to(self.device)

        # Compute the expected Q values
        next_state_values = next_state_values.detach().max(1)[0]
        next_state_values = next_state_values.unsqueeze(1)
        next_state_values[batch_t] = 0.0

        expected_state_action_values = (
            next_state_values * self.gamma) + batch_r

        # calculate loss
        # Compute Loss
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.cum_loss += loss.item()
        # optimize model - update weights
        self.optimizer.zero_grad()
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        loss.backward()
        self.optimizer.step()
        for key in self.policy_net.state_dict():
            self.target_net.state_dict()[key] = self.policy_net.state_dict(
            )[key]*self.tau + self.target_net.state_dict()[key]*(1-self.tau)
        self.target_net.load_state_dict(self.policy_net.state_dict())


class PacmanPOMDPDQN(PacmanDQN):
    def __init__(self, **kwargs):
        if "num_obs_directions" in kwargs:
            self.num_obs_directions = kwargs["num_obs_directions"]
            del kwargs["num_obs_directions"]
        else:
            self.num_obs_directions = 4
        self.obs_size = (self.num_obs_directions,)
        kwargs["obs_size"] = self.obs_size
        if "path" not in kwargs:
            kwargs["path"] = "models/pacmanPOMDPDQN.pt"
        super().__init__(**kwargs)
        print("Started PacmanPOMDPDQN")

    def getObservation(self, state: GameState):
        state_matrix = self.getPartialObservation(state,self.num_obs_directions)
        return state_matrix.flatten()

    def getReward(self, state: GameState):
        return self.getComplexReward(state,num_obs_dirs=self.num_obs_directions,alpha = 3)



class PacmanLSTMPOMDP(PacmanPOMDPDQN):
    def __init__(self, **kwargs):
        if "path" not in kwargs:
            kwargs["path"] = "models/pacmanLSTMPOMDP.pt"
        self.sequence_length = kwargs["sequence_length"]
        del kwargs["sequence_length"]
        super().__init__(**kwargs)
        self.episode_history = []
        self.evaluation_batch = []
        print("Started PacmanLSTMPOMDP")

    def observation_step(self, state):
        if self.last_action is None:
            return
        # Reset LSTM hidden state
        if self.terminal:
            self.policy_net.reset()
        # get state
        self.last_state = np.copy(self.current_state)
        self.current_state = self.getObservation(state)

        self.last_game_state = copy.deepcopy(self.current_game_state)
        self.current_game_state = state
        # get reward
        reward = self.getReward(state)
        if (state.isWin()):
            self.win_counter += 1
            self.cum_wins += 1
        self.episode_reward += reward

        # store sequence if the episode ended, or if the sequence is full
        # if the episode is terminal we need to pad the sequence with zeros
        transition = Transition(
            self.last_state, self.last_action, self.current_state, reward, self.terminal)
        self.episode_history.append(transition)
        if self.terminal:
            # Pad if episode to short
            current_lenght = len(self.episode_history)
            for _ in range(self.sequence_length-current_lenght):
                self.episode_history.append(Transition(
                    self.last_state,self.last_action, self.current_state, 0, self.terminal))

        if len(self.episode_history) >= self.sequence_length:
            if self.episode_number < self.start_training:
                self.evaluation_batch.append(copy.deepcopy(
                    list(map(lambda x: (x.state,x.terminal),self.episode_history))))
            self.ReplayMemory.push(copy.deepcopy(self.episode_history))
            self.episode_history.clear()
            # train model
            self.train()
    def evaluate_Qestimates(self):
        self.policy_net.eval()
        batch_s, batch_len = [], []
        for seq in self.evaluation_batch:
            seq_s, seq_t = zip(*seq)
            seq_len = len(seq_s) - max(0,sum(seq_t)-1)
            batch_s.append(seq_s)
            batch_len.append(seq_len)
        # convert from numpy to pytorch
        batch_s = torch.from_numpy(np.stack(batch_s)).to(self.device)
        batch_len = torch.LongTensor(batch_len).to("cpu")
        Q_values = self.policy_net(batch_s, batch_len)
        Q_mean, Q_std = torch.mean(Q_values.detach()), torch.std(Q_values.detach())
        self.Q_mean_vals = np.append(self.Q_mean_vals,Q_mean.cpu().numpy())
        return Q_mean, Q_std
    def train(self):
        if (self.episode_number <= self.start_training
                or not self.training):
            return
        self.policy_net.train()
        batch_seq = random.sample(self.ReplayMemory.memory, self.batch_size)
        batch_s, batch_r, batch_a, batch_n, batch_t, batch_len = [], [], [], [], [], []
        for seq in batch_seq:
            seq_s, seq_a, seq_n, seq_r, seq_t = Transition(*zip(*seq))
            seq_len = len(seq_s) - max(0,sum(seq_t)-1)
            batch_s.append(seq_s)
            batch_r.append(seq_r)
            batch_a.append(seq_a)
            batch_n.append(seq_n)
            batch_t.append(seq_t)
            batch_len.append(seq_len)
        # convert from numpy to pytorch
        batch_s = torch.from_numpy(np.stack(batch_s)).to(self.device)
        batch_r = torch.DoubleTensor(batch_r).unsqueeze(2).to(self.device)
        batch_a = torch.LongTensor(batch_a).unsqueeze(2).to(self.device)
        batch_n = torch.from_numpy(np.stack(batch_n)).to(self.device)
        batch_t = torch.BoolTensor(batch_t).unsqueeze(2).to(self.device)
        batch_len = torch.LongTensor(batch_len).to("cpu")
        # Sequence length has to be on the cpu
        if self.batch_norm == "Z-score":
            batch_r = (batch_r - batch_r.mean())/batch_r.std()
        elif self.batch_norm == "MinMax":
            batch_r = (batch_r - batch_r.min())/(batch_r.max()-batch_r.min())
        # get Q(s, a)
        state_action_values = self.policy_net(batch_s, batch_len)
        self.Q_values = np.append(
            self.Q_values, state_action_values.detach().max(2)[0].cpu().numpy())
        state_action_values = state_action_values.gather(2, batch_a)
        # get V(s')
        next_state_values = self.target_net(batch_n,batch_len).to(self.device)

        # Compute the expected Q values
        next_state_values = next_state_values.detach().max(2)[0]
        next_state_values = next_state_values.unsqueeze(2)
        next_state_values[batch_t] = 0.0
        expected_state_action_values = (
            next_state_values * self.gamma) + batch_r

        # calculate loss
        # Compute Loss
        criterion = get_loss_from_dict(self.loss)
        loss = criterion(state_action_values, expected_state_action_values)
        self.cum_loss += loss.item()
        # optimize model - update weights
        self.optimizer.zero_grad()
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        loss.backward()
        self.optimizer.step()
        for key in self.policy_net.state_dict():
            self.target_net.state_dict()[key] = self.policy_net.state_dict(
            )[key]*self.tau + self.target_net.state_dict()[key]*(1-self.tau)
        self.target_net.load_state_dict(self.policy_net.state_dict())
