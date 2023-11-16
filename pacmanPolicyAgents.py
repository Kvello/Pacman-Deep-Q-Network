from typing import Dict, Any, Tuple
import copy
import numpy as np
import torch
from torch.distributions import Categorical
from parsers import get_optimizer_from_dict, get_loss_from_dict
from pacmanUtils import PacmanUtils
from policyNets import PolicyNet
from architectures.Conv import Conv
from architectures.feedForward import FeedForward
from architectures.ResLSTM import ResLSTM, LSTM
from pacman import GameState
"""
This code is yet another example of the problems with inheritance.
A proper clean up of this code would be to use more composition instead of
inheritance. However, this would require a complete rewrite of the code
and is therefore not feasible at this point.
"""
default_optimizer = {"type": "RMSprop",
                     "args": {
                         "lr": 1e-3,
                         "eps": 1e-6,
                         "alpha": 0.95
                        }
                    }
def format_tensors(tensor_list, max_size):
    new_tensor_list = []
    tensor_lengths = []
    for tensor in tensor_list:
        if tensor.size(0) > max_size:
            split_tensors = torch.split(tensor, max_size)
            new_tensor_list.extend(split_tensors)
        else:
            new_tensor_list.append(tensor)
    tensor_lengths = [tensor.size(0) for tensor in new_tensor_list]
    new_tensor = torch.nn.utils.rnn.pad_sequence(new_tensor_list, batch_first=True)
    return new_tensor, tensor_lengths


def reward_to_go(rews,gamma=1):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1]*gamma if i+1 < n else 0)
    return rtgs

class PacmanPolicyAgent(PacmanUtils):
    def __init__(self, 
                 model:Dict[str,Any],
                 num_actions:int,
                 discount_factor:float,
                 obs_size:Tuple[int,int],
                 num_object_types:int,
                 training:bool=True,
                 optimizer:Dict[str,Any]={},
                 path:str="models/pacmanPolicyAgent.pth",
                 quiet:bool=False,
                 batch_norm:str="None",
                 batch_size:int=32,
                 load_model:bool=False,
                 stat_freq:int=100
                 ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quiet = quiet
        self.training = training
        self.discount_factor = discount_factor
        self.path = path
        self.log_probabilities = []
        self.cum_rewards = 0
        self.episode_number = 0
        self.batch_norm = batch_norm
        self.terminal = False
        self.obs_size = (num_object_types, ) + obs_size
        self.num_actions = num_actions
        self.batch_weights = torch.Tensor([]).to(self.device)
        self.batch_mv = []
        self.batch_obs = []
        self.ep_rews = []
        self.stat_freq = stat_freq
        self.cum_wins = 0
        self.cum_entropy = 0
        self.num_steps = 0
        self.gradients = torch.tensor([]).to(self.device)
        self.batch_size = batch_size
        self.current_state = None
        self.last_action = None
        self.MC_estimate = 0
        self.MC_variance = 0
        self.grads = {}


        # Although these models where made for Q-learning,
        # they can be used for policy learning as well
        if (load_model == True):
            self.model = torch.load(
                path, map_location=torch.device(self.device)).to(self.device)
        else:
            self.model = PolicyNet(num_object_types,
                                   num_actions,
                                   obs_size,
                                   **model).double().to(self.device)
        self.optimizer = get_optimizer_from_dict(self.model,
                                                 optimizer)
        print("Started PacmanPolicyAgent")
        print("Model: ", self.model)
    def getPolicy(self,obs):
        logits = self.model(obs)
        return Categorical(logits=logits)
    def getLoss(self, obs, action, weights):
        logp = self.getPolicy(obs).log_prob(action)
        samples = logp * weights
        print("len samples: {}".format(len(samples)))
        return -samples.mean(), 1/len(samples)*torch.var(samples)
    def getMove(self):
        """samples action from the policy"""
        current_state_tensor = torch.from_numpy(np.stack(self.current_state))
        current_state_tensor = current_state_tensor.unsqueeze(0).to(self.device).double()
        policy = self.getPolicy(current_state_tensor)
        move = policy.sample()
        self.batch_mv.append(move)
        self.batch_obs.append(current_state_tensor.clone())
        move = self.get_direction(move)
        self.cum_entropy += policy.entropy().item()
        self.last_action = move
        self.num_steps += 1
        return move
    def getObservation(self,state:GameState):
        """returns the observation from the state"""
        return self.getStateMatrix(state)
    def getReward(self,state:GameState):
        """returns the reward from the state"""
        return self.getComplexReward(state)
    def updateStats(self,state:GameState,reward:float):
        if state.isWin():
            self.cum_wins += 1
        self.cum_rewards += reward

    def observation_step(self,state:GameState):
        if self.last_action is None:
            return
        self.current_state = self.getObservation(state)
        reward = self.getReward(state)
        self.updateStats(state,reward)
        self.ep_rews.append(reward)
        if self.terminal:
            discounted_rtg = torch.from_numpy(reward_to_go(self.ep_rews,self.discount_factor)).to(self.device)
            if len(discounted_rtg)>1:
                if self.batch_norm == "zscore":
                    discounted_rtg =\
                        (discounted_rtg - discounted_rtg.mean()) / discounted_rtg.std()
                elif self.batch_norm == "minmax":
                    discounted_rtg = \
                        (discounted_rtg - discounted_rtg.min()) / (discounted_rtg.max() - discounted_rtg.min())
            self.batch_weights = torch.cat((self.batch_weights,discounted_rtg))
            self.ep_rews = []

        if len(self.batch_weights) > self.batch_size and self.training:
            self.update()
    def update(self):
        self.MC_estimate, self.MC_variance = self.getLoss(
            torch.cat(self.batch_obs).double().to(self.device),
            torch.as_tensor(self.batch_mv).double().to(self.device),
            torch.as_tensor(self.batch_weights).double().to(self.device))
        self.optimizer.zero_grad()
        self.MC_estimate.backward()
        newGrads = {}
        sign_changes = 0
        num_params = 0
        for name, parametr in self.model.named_parameters():
            num_params += parametr.numel()
            if name in self.grads:
                newGrads[name] = parametr.grad
                sign_changes += torch.sum(torch.sign(self.grads[name]) != torch.sign(newGrads[name])).item()
                self.grads[name] = newGrads[name]
            else :
                self.grads[name] = parametr.grad
        print("percentage of sign changes: {}".format(sign_changes/num_params))
        self.optimizer.step()
        self.batch_weights= torch.Tensor([]).to(self.device)
        self.batch_mv = []
        self.batch_obs = []
    def printStats(self):
        print("---------------------------------------")
        print("Episode no = " + str(self.episode_number) +
                "; won: " + str(self.cum_wins))
        print("Avg reward = " + str(self.cum_rewards / self.stat_freq))
        print("Entropy: {}".format(self.cum_entropy/self.num_steps))
        if self.MC_estimate != 0:
            print("MC Estimate: {}".format(self.MC_estimate))
            print("MC Variance: {}".format(self.MC_variance))
    def resetStats(self):
        self.cum_wins = 0
        self.num_steps = 0
        self.cum_entropy = 0
        self.cum_rewards = 0
    def final(self, state):
        # do observation
        self.terminal = True
        self.observation_step(state)
        if not self.quiet and self.episode_number % self.stat_freq == 0:
            self.printStats()
            self.resetStats()
        if isinstance(self.model, LSTM):
            self.model.reset()
    def setTraining(self, training:bool):
        self.training = training
    def saveModel(self):
        torch.save(self.model, self.path)
class PacmanPOMDPPolicyAgent(PacmanPolicyAgent):
    def __init__(self, **kwargs):
        if "num_obs_directions" in kwargs:
            self.num_obs_directions = kwargs["num_obs_directions"]
            del kwargs["num_obs_directions"]
        else:
            self.num_obs_directions = 4
        self.obs_size = (self.num_obs_directions,)
        kwargs["obs_size"] = self.obs_size
        if "path" not in kwargs:
            kwargs["path"] = "models/pacmanPOMDPolicyAgent.pt"
        super().__init__(**kwargs)
        print("Started PacmanPOMD PolicyAgent")
    def getObservation(self, state: GameState):
        state_matrix = self.getPartialObservation(state,self.num_obs_directions)
        return state_matrix.flatten()

    def getReward(self, state: GameState):
        return self.getComplexReward(state,num_obs_dirs=self.num_obs_directions)

class PacmanCritic(PacmanUtils):
    def __init__(self,
                 model:Dict[str,Any],
                 optimizer:Dict[str,Any],
                 obs_size:Tuple[int,int],
                 num_object_types:int,
                 training:bool=True,
                 path:str="models/pacmanCritic.pt",
                 quiet:bool=False,
                 load_model:bool=False,
                 num_epochs:int=1,
                 loss:str={"type":"MSE"}):
        print("Started PacmanCritic")
        self.training = training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path = path
        self.quiet = quiet
        self.num_epochs = num_epochs
        self.loss = loss
        self.num_updates = 0
        self.cum_loss = 0
        self.avg_value = 0
        if (load_model == True):
            self.model = torch.load(
                path, map_location=torch.device(self.device)).to(self.device)
        else:
            arch = model["arch"]
            args = model["args"]
            args["obs_size"] = (num_object_types,)+obs_size
            args["num_actions"] = 1
            if arch.lower() == "feedforward":
                self.model = FeedForward(**args).double().to(self.device)
            elif arch.lower() == "conv":
                self.model = Conv(**args).double().to(self.device)
            elif arch.lower() == "lstm":
                self.seq_len = model["sequence_length"]
                self.model = LSTM(**args).double().to(self.device)
            elif arch.lower() == "reslstm":
                self.model = ResLSTM(**args).double().to(self.device)
            else:
                raise ValueError("invalid architecture {}".format(model["arch"]))
        print("Model: ", self.model)
            
        self.optimizer = get_optimizer_from_dict(self.model,
                                                    optimizer)
    def getValue(self,observation:torch.Tensor):
        self.model.eval()
        return self.model(observation)
    def saveModel(self):
        torch.save(self.model, self.path)
    def setTraining(self, training:bool):
        self.training = training
    def printStats(self):
        if self.num_updates >0:
            print("Average value: {}".format(self.avg_value/self.num_updates))
            print("Critic loss: {}".format(self.cum_loss/self.num_updates))
    def resetStats(self):
        self.avg_value = 0
        self.cum_loss = 0
        self.num_updates = 0
    def update(self, input, target):
        loss = get_loss_from_dict(self.loss)
        # Format data baased on model type
        self.model.train()
        if isinstance(self.model, LSTM):
            input, seq_len = format_tensors(input, self.seq_len)
            target, _ = format_tensors(target, self.seq_len)
            target = target.unsqueeze(2)
        else:
            input = torch.cat(input)
            target = torch.cat(target)
            target = target.unsqueeze(1)
        for _ in range(self.num_epochs):
            if isinstance(self.model, LSTM):
                pred = self.model(input, seq_len)
            else:
                pred = self.model(input)
            loss_val = loss(pred, target)
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
        self.num_updates += 1
        self.avg_value += pred.squeeze(1).mean()
        self.cum_loss += loss_val.item()
        return loss_val.item()
    def saveModel(self):
        torch.save(self.model, self.path)

class PacmanActorCritic(PacmanUtils):
    def __init__(self, **kwargs):
        critic_conf = kwargs["critic"]
        actor_conf = kwargs["actor"]
        num_object_types = kwargs["num_object_types"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quiet = kwargs['quiet']
        self.training = kwargs['training']
        self.path = critic_conf["path"]
        self.cum_rewards = 0
        self.episode_number = 0
        self.discount_factor = kwargs['discount_factor']
        self.terminal = False
        self.stat_freq = kwargs['stat_freq']
        self.cum_wins = 0
        self.batch_size = kwargs['batch_size']
        obs_size = kwargs['obs_size']
        self.sample_set = []
        self.episode_history = []
        self.discounted_rtgs = []
        self.samples = 0

        print("Started PacmanActorCritic")
        self.critic = PacmanCritic( obs_size=obs_size,
                                      num_object_types=num_object_types,
                                      training=kwargs["training"],
                                      quiet=kwargs["quiet"],
                                      load_model=kwargs["load_model"],
                                      **critic_conf)
        self.actor = PacmanPolicyAgent(model=actor_conf["model"],
                                       num_actions=actor_conf["num_actions"],
                                       discount_factor=kwargs["discount_factor"],
                                       obs_size=kwargs["obs_size"],
                                       num_object_types=kwargs["num_object_types"],
                                       training=False,
                                       optimizer=actor_conf["optimizer"],
                                       path=actor_conf["path"],
                                       quiet=kwargs["quiet"],
                                       batch_norm=actor_conf["batch_norm"],
                                       batch_size=kwargs["batch_size"],
                                       load_model=kwargs["load_model"],
                                       stat_freq=kwargs["stat_freq"])

    def observation_step(self,state:GameState):
        if self.actor.last_action is None:
            return
        self.last_state = np.copy(self.current_state)
        self.current_state = self.getObservation(state)
        self.actor.observation_step(state)
        last_state_tensor = np.copy(self.last_state)
        last_state_tensor = torch.from_numpy(last_state_tensor)
        last_state_tensor = last_state_tensor.to(self.device).double()
        self.episode_history.append(last_state_tensor)
        if self.terminal:
            rewards = self.actor.ep_rews
            states = torch.stack(self.episode_history)
            states.to(self.device)
            self.samples += len(self.episode_history)
            discounted_rtg = torch.from_numpy(reward_to_go(rewards,self.discount_factor)).to(self.device)
            self.sample_set.append(states.clone())
            self.discounted_rtgs.append(discounted_rtg)
            self.episode_history = []
            self.actor.ep_rews = []
        if self.samples > self.batch_size and self.training:
            self.update()
    def update(self):
        self.critic.update(self.sample_set, self.discounted_rtgs)
        with torch.no_grad():
            if isinstance(self.critic.model, LSTM):
                state_seqs, seq_lens = format_tensors(self.sample_set, self.critic.seq_len)
                value_estimates = self.critic.model(state_seqs, seq_lens)
                value_estimates = value_estimates.flatten()
            else:
                value_estimates = self.critic.getValue(torch.cat(self.sample_set)).squeeze(1)
        self.discounted_rtgs = torch.cat(self.discounted_rtgs)
        advantages = self.discounted_rtgs - value_estimates
        if self.actor.batch_norm == "zscore":
            advantages =\
                (advantages - advantages.mean()) / advantages.std()
        elif self.actor.batch_norm == "minmax":
            advantages = \
                (advantages - advantages.min()) / (advantages.max() - advantages.min())
        self.actor.batch_weights = advantages
        self.actor.update()
        self.sample_set = []
        self.samples = 0
        self.discounted_rtgs = []
    def final(self, state):
        # do observation
        self.terminal = True
        self.observation_step(state)
        self.actor.episode_number = self.episode_number
        if (self.episode_number % self.stat_freq == 0):
            if not self.quiet:
                self.actor.printStats()
                self.critic.printStats()
            self.actor.resetStats()
            self.critic.resetStats()
        if isinstance(self.actor.model, LSTM):
            self.actor.model.reset()
        if isinstance(self.critic.model, LSTM):
            self.critic.model.reset()
    def setTraining(self, training:bool):
        self.training = training
    def saveModel(self):
        print("Saving model")
        self.actor.saveModel()
        self.critic.saveModel()
    def getMove(self):
        return self.actor.getMove()
    def getObservation(self,state:GameState):
        return self.actor.getObservation(state)
    def registerInitialState(self, state: GameState):
        super().registerInitialState(state)
        self.actor.current_state = self.current_state
        self.actor.last_state = self.last_state
        self.actor.last_action = self.last_action
        self.actor.episode_number = self.episode_number
        self.actor.last_score = self.last_score
        self.actor.terminal = self.terminal
        self.actor.last_reward = self.last_reward
        self.actor.won = self.won

class PacmanActorCriticPOMDP(PacmanActorCritic):
    def __init__(self, **kwargs):
        if "num_obs_directions" in kwargs:
            self.num_obs_directions = kwargs["num_obs_directions"]
            del kwargs["num_obs_directions"]
        else:
            self.num_obs_directions = 4
        self.obs_size = (self.num_obs_directions,)
        kwargs["obs_size"] = self.obs_size
        if "path" not in kwargs:
            kwargs["path"] = "models/pacmanActorCriticPOMDP.pt"
        super().__init__(**kwargs)
        print("Started PacmanActorCriticPOMDP")
        self.actor = PacmanPOMDPPolicyAgent(model=kwargs["actor"]["model"],
                                        num_actions=kwargs["actor"]["num_actions"],
                                        discount_factor=kwargs["discount_factor"],
                                        obs_size=self.obs_size,
                                        num_object_types=kwargs["num_object_types"],
                                        training=kwargs["training"],
                                        optimizer=kwargs["actor"]["optimizer"],
                                        path=kwargs["actor"]["path"],
                                        quiet=kwargs["quiet"],
                                        batch_norm=kwargs["actor"]["batch_norm"],
                                        batch_size=kwargs["batch_size"],
                                        load_model=kwargs["load_model"],
                                        stat_freq=kwargs["stat_freq"],
                                        num_obs_directions=self.num_obs_directions)
    def getObservation(self, state: GameState):
        state_matrix = self.getPartialObservation(state,self.num_obs_directions)
        return state_matrix.flatten()
    def getReward(self, state: GameState):
        return self.getComplexReward(state,num_obs_dirs=self.num_obs_directions)



