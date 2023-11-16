from typing import List, Any, Union, Dict
from environment import Environment
class EnvStepRule(object):
    """
    Class for creating rules for stepping to the next environment
    Args:
        performance_threshold: Threshold for performance
        patience: Number of times the performance threshold must be reached
        thres_direction: Direction of the threshold
    """
    def __init__(self,
                performance_threshold:float,
                patience:int=1,
                thresh_direction:str='above'
                ):
        self.performance_threshold = performance_threshold
        self.progress_counter = 0
        self.patience = patience
        self.thresh_direction = thresh_direction
    def checkRule(self, performance:float):
        if self.thresh_direction == 'above':
            if performance >= self.performance_threshold:
                self.progress_counter += 1
            else:
                self.progress_counter = 0
        elif self.thresh_direction == 'below':
            if performance <= self.performance_threshold:
                self.progress_counter += 1
            else:
                self.progress_counter = 0
        if self.progress_counter >= self.patience:
            return True
        else:
            return False
    
class EnvScheduler(object):
    """
    Class for scheduling the environment used for curriculum learning
    Args:
        env_list: List of environments
        env_start: Starting environment
        env_step_rule: Dictionary specifying the rules for stepping to the next environment
        env_back_rule: Dictionary specifying the rules for stepping back to the previous environment
    """
    def __init__(self,
                 env_list:List[Environment],
                 env_step_rule:Union[Dict[str, Any],EnvStepRule],
                 env_back_rule:Union[Dict[str, int], EnvStepRule,None] = None,
                 env_start:int = 0,
                 quiet:bool = True):
        self.env_num = env_start
        self.num_envs = len(env_list)
        self.performace = 0
        self.quiet = quiet
        self.env_list = sorted(env_list, key=lambda x: x.difficulty)
        if isinstance(env_step_rule, EnvStepRule):
            self.env_step_rule = env_step_rule
        else:
            self.env_step_rule = EnvStepRule(**env_step_rule)
        if env_back_rule is not None:
            if isinstance(env_back_rule, EnvStepRule):
                self.env_back_rule = env_back_rule
            else:
                self.env_back_rule = EnvStepRule(**env_back_rule)
        else:
            self.env_back_rule = None

    def registerPerformance(self, performance):
        if self.env_step_rule.checkRule(performance):
            if not self.quiet:
                print("Stepping to next environment")
            self.env_num += 1
        elif self.env_back_rule is not None:
            if self.env_back_rule.checkRule(performance):
                if not self.quiet:
                    print("Stepping back to previous environment")
                self.env_num -= 1
    def getEnv(self):
        if self.env_num < self.num_envs:
            return self.env_list[self.env_num]
        return None

        