import torch
from typing import Dict, Any
def get_optimizer_from_dict(model:torch.nn.Module,
                            optimizer:Dict[str,Any]) -> torch.optim.Optimizer:
    """Returns an optimizer from a dictionary of the form:
    {"type": "RMSprop",
     "args": {
         "lr": 1e-3,
         "eps": 1e-6,
         "alpha": 0.95
        }
    }
    """
    if optimizer["type"] == "RMSprop":
        return torch.optim.RMSprop(model.parameters(),
                                   **optimizer["args"])
    elif optimizer["type"] == "Adam":
        return torch.optim.Adam(model.parameters(),
                                **optimizer["args"])
    elif optimizer["type"] == "SGD":
        return torch.optim.SGD(model.parameters(),
                               **optimizer["args"])
    elif optimizer["type"] == "AdamW":
        return torch.optim.AdamW(model.parameters(),
                                 **optimizer["args"])
    else:
        raise ValueError("Optimizer type {} not recognized".format(optimizer["type"]))
    
def get_loss_from_dict(loss:Dict[str,Any]) -> torch.nn.Module:
    """Returns a loss function from a dictionary of the form:
    {"type": "Huber",
     "args": {
         "reduction": "mean",
         "delta": 1.0
        }
    }
    """
    if "args" in loss.keys():
        args = loss["args"]
    else:
        args = {}
    if loss["type"] == "Huber":
        return torch.nn.SmoothL1Loss(**args)
    elif loss["type"] == "MSE":
        return torch.nn.MSELoss(**args)
    elif loss["type"] == "L1":
        return torch.nn.L1Loss(**args)
    else:
        raise ValueError("Loss type {} not recognized".format(loss["type"]))

def get_scheduler_from_dict(optimizer:torch.optim.Optimizer,
                            scheduler:Dict[str,Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """Returns a scheduler from a dictionary of the form:
    {"type": "StepLR",
     "args": {
         "step_size": 30,
         "gamma": 0.1
        }
    }
    """
    if scheduler["type"] == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                               **scheduler["args"])
    elif scheduler["type"] == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    **scheduler["args"])
    elif scheduler["type"] == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                      **scheduler["args"])
    elif scheduler["type"] == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          **scheduler["args"])
    else:
        raise ValueError("Scheduler type {} not recognized".format(scheduler["type"]))
