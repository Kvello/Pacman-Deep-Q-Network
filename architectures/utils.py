from collections import namedtuple, deque
import random
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','terminal'))
class EarlyStopper:
    def __init__(self, patience=1, threshold=0, threshold_mode='rel',mode = "max"):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.mode = mode
        if mode not in ["min","max"]:
            raise ValueError("Mode {} not supported".format(mode))
        if mode == "min":
            self.extreme_validation_loss = float('inf')
        else:
            self.extreme_validation_loss = -float('inf')
        if threshold_mode not in ['rel', 'abs']:
            raise ValueError("Threshold mode {} not supported".format(threshold_mode))
        self.threshold_mode = threshold_mode

    def early_stop(self, loss ,quiet=False):
        if self.threshold_mode == 'rel':
            if self.mode == "min":
                dynamic_threshold = self.extreme_validation_loss * (1 + self.threshold)
            else:
                dynamic_threshold = self.extreme_validation_loss * (1 - self.threshold)
        else:
            if self.mode == "min":
                dynamic_threshold = self.extreme_validation_loss + self.threshold
            else:
                dynamic_threshold = self.extreme_validation_loss - self.threshold
        if self.mode == "max":
            loss = -loss
            dynamic_threshold = -dynamic_threshold
        if loss < dynamic_threshold:
            self.extreme_validation_loss = loss
            self.counter = 0
        elif loss> dynamic_threshold:
            self.counter += 1
            if not quiet:
                print("Validation loss increased, counter: {}".format(self.counter))
            if self.counter >= self.patience:
                if not quiet:
                    print("Count: {}, Patience: {} - quitting".format(self.counter,self.patience))
                return True
        return False                   

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(*args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def clear(self):
        self.memory.clear()
    def __len__(self):
        return len(self.memory)