import layout
import os
from typing import List

class Environment():
    """
    Base class for environments
    Args:
        difficulty: Difficulty of the environment
        id: Unique identifier for the environment
    """
    def __init__(self,difficulty:int,id:str):
        self.difficulty = difficulty
        self.id = id


class PacmanEnvironment(Environment):
    """
    Pacman environment
    Args:
        difficulty: Difficulty of the environment
        id: Unique identifier for the environment
        numGhosts: Number of ghosts in the environment
        randomStart: Whether to start the agents at random locations
        ghostType: Type of ghosts to use
    """
    def __init__(self,difficulty:int,
                 id:str,
                 random_start:bool,
                 lay:layout,
                 ghosts:List[object]):
        super().__init__(difficulty,id)
        self.random_start = random_start
        self.ghosts = ghosts
        self.lay = lay

def loadAgent(pacman, nographics):
    # Looks through all pythonPath Directories for the right module,
    pythonPathStr = os.path.expandvars("$PYTHONPATH")
    if pythonPathStr.find(';') == -1:
        pythonPathDirs = pythonPathStr.split(':')
    else:
        pythonPathDirs = pythonPathStr.split(';')
    pythonPathDirs.append('.')

    for moduleDir in pythonPathDirs:
        if not os.path.isdir(moduleDir):
            continue
        moduleNames = [f for f in os.listdir(
            moduleDir) if f.endswith('gents.py')]
        for modulename in moduleNames:
            try:
                module = __import__(modulename[:-3])
            except ImportError:
                continue
            if pacman in dir(module):
                if nographics and modulename == 'keyboardAgents.py':
                    raise Exception(
                        'Using the keyboard requires graphics (not text display)')
                return getattr(module, pacman)
    raise Exception('The agent ' + pacman +
                    ' is not specified in any *Agents.py.')
def parsePacmanEnv(noKeyboard,**envArg):

    ghostType = loadAgent(envArg["ghost"],noKeyboard)
    ghosts = [ghostType(i + 1) for i in range(envArg["numGhosts"])]
    lay = layout.getLayout(envArg["layout"])
    if lay == None:
        raise Exception("The layout " + envArg['layout'] + " cannot be found")
    return PacmanEnvironment(difficulty=envArg["difficulty"],
                            id=envArg["layout"],
                            random_start=envArg["randomStartPos"],
                            lay=lay,
                            ghosts=ghosts)