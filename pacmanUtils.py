# import pacman game 
from pacman import Directions, GameState
import game
import numpy as np

class PacmanUtils(game.Agent):

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        if direction == Directions.EAST:
            return 1.
        if direction == Directions.SOUTH:
            return 2.
        if direction == Directions.WEST:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        if value == 1.:
            return Directions.EAST
        if value == 2.:
            return Directions.SOUTH
        if value == 3.:
            return Directions.WEST
			
    def observationFunction(self, state:GameState):
        # do observation
        self.terminal = False
        self.observation_step(state)

        return state
		
    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((batch_size, 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrix(self, state):
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height 
        width, height = self.obs_size[1], self.obs_size[2]
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)
        return observation

    def registerInitialState(self, state:GameState): # inspects the starting state
        # Reset reward
        self.last_score = 0
        self.last_reward = 0.
        self.episode_reward = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getObservation(state)

        self.last_game_state = None
        self.current_game_state = state

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.delay = 0

        # Next
        self.frame = 0
        self.episode_number += 1
    def getAction(self, state):
        move = self.getMove()

        # check for illegal moves
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP
        return move
    def getPartialObservation(self,state:GameState,num_directions = 4)->np.ndarray:

        """ Return an partial environment observation.
        The observation is a 5xnum_directions matrix, where each row corresponds to a different
        object type (wall, ghost, scared ghost, food, capsule) and each column corresponds to a
        direction (north, east, south, west, northeast, southeast, southwest, northwest).
        The values in the matrix are the distance to the nearest object of that type in that
        direction. If there is no object in that direction, the distance is 0.
        Pacman can see through food, and capsules, but not through walls or ghosts.
        He only sees the nearest food, and capsule, but can see ghosts and walls behind 

        Args:
            state (GameState): The current game state
            num_directions (int, optional): The number of directions to consider, either 4 or 8.
        Returns:
            np.ndarray: The partial observation
        """
        pacmanPos = np.array(state.getPacmanPosition())
        ghosts = state.getGhostStates()
        scared_ghosts = [g for g in ghosts if g.scaredTimer > 0]
        ghosts = list(set(ghosts) - set(scared_ghosts))
        capsules = state.getCapsules()
        obs = np.zeros((5, num_directions))
        if num_directions == 8:
            directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        elif num_directions == 4:
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        else:
            raise ValueError("num_directions must be either 4 or 8")
        for i, direction in enumerate(directions):
            # Get the distance to the nearest object in each direction
            # and the type of object
            step = 1
            hit = False
            hit_food = False
            hit_capsule = False
            while not hit:
                pos = pacmanPos + step * np.array(direction)
                pos = [int(pos[0]), int(pos[1])]
                if state.hasWall(int(pos[0]), int(pos[1])):
                    hit = True
                    obs[0, i] = step
                elif tuple(pos) in [tuple(g.getPosition()) for g in ghosts]:
                    hit = True
                    obs[1, i] = step
                elif tuple(pos) in [tuple(g.getPosition()) for g in scared_ghosts]:
                    hit = True
                    obs[2, i] = step
                elif state.hasFood(int(pos[0]), int(pos[1])):
                    if not hit_food:
                        obs[3, i] = step
                        hit_food = True
                elif tuple(pos) in capsules:
                    if not hit_capsule:
                        obs[4, i] = step
                        hit_capsule = True
                step += 1
        return obs
    
    def getComplexReward(self, state:GameState,num_obs_dirs:int=4,alpha:float=2.5)->float:
        """
        Returns a reward based on the current state of the game
        The equation is as follows:
        R = Rwin + Rfood + Rcapsule + Rscared - Rghost + Rstationary
        where:
        Rwin = 500 if the game is won, 0 otherwise
        Rfood = 5*exp(-alpha*foodDist)/exp(alpha) if food is in sight, 0 otherwise
        Rcapsule = 50*exp(-alpha*capsuleDist)/exp(alpha) if capsule is in sight, 0 otherwise
        Rscared = 100*exp(-alpha*scaredDist)/exp(alpha) if scared ghost is in sight, 0 otherwise
        Rghost = 500*exp(-alpha*ghostDist)/exp(alpha) if ghost is in sight, 0 otherwise
        Rstationary = -2 always
        """
        prev_state = self.last_game_state
        stateMatrix = self.getPartialObservation(state,num_obs_dirs)
        foodReward = 0
        capsuleReward = 0
        scaredReward = 0
        ghostPunish = 0
        # Reward being in sight of food
        pacmanPos = np.array(state.getPacmanPosition())
        ghosts = np.array(state.getGhostStates())
        scaredGhostPos = np.array([g.getPosition() for g in ghosts if g.scaredTimer > 0])
        ghostPos = np.array([g.getPosition() for g in ghosts if g.scaredTimer <= 0])
        if num_obs_dirs == 4:
            straight_distances = stateMatrix
        elif num_obs_dirs == 8:
            straight_distances = stateMatrix[:,0::2]
        indx = np.where(straight_distances[3,:]>0)[0]
        foodDist = straight_distances[3,indx]
        if len(foodDist) > 0:
            foodDist = min(foodDist)
        else:
            foodDist = -1 if prev_state.hasFood(int(pacmanPos[0]),int(pacmanPos[1])) else 0
        indx = np.where(straight_distances[4,:]>0)[0]
        capsuleDist = straight_distances[4,indx]
        if len(capsuleDist) > 0:
            capsuleDist = min(capsuleDist)
        else:
            capsuleDist = -1 if tuple(pacmanPos) in state.getCapsules() else 0
        scaredDist = np.where(straight_distances[2,:]>0)[0]
        if len(scaredDist) > 0:
            scaredDist = min(scaredDist)
        else:
            scaredDist = -1 if tuple(pacmanPos) in [tuple(g) for g in scaredGhostPos] else 0
        indx = np.where(straight_distances[1,:]>0)[0]
        ghostDist = straight_distances[1,indx]
        if len(ghostDist) > 0:
            ghostDist = min(ghostDist)
        else:
            ghostDist = -1 if tuple(pacmanPos) in [tuple(g) for g in ghostPos] else 0
        if num_obs_dirs == 8:
            diagonal_distances = stateMatrix[:,1::2]
            indx = np.where(diagonal_distances[3,:]>0)[0]
            foodDistDiag = diagonal_distances[3,indx]
            if len(foodDistDiag) > 0:
                foodDist = foodDist + np.sqrt(2)*min(foodDistDiag)
            indx = np.where(diagonal_distances[4,:]>0)[0]
            capsuleDistDiag = diagonal_distances[4,indx]
            if len(capsuleDistDiag) > 0:
                capsuleDist = capsuleDist + np.sqrt(2)*min(capsuleDistDiag)
            indx = np.where(diagonal_distances[2,:]>0)[0]
            scaredDistDiag = diagonal_distances[2,indx]
            if len(scaredDistDiag) > 0:
                scaredDist = scaredDist + np.sqrt(2)*min(scaredDistDiag)
            indx = np.where(diagonal_distances[1,:]>0)[0]
            ghostDistDiag = diagonal_distances[1,indx]
            if len(ghostDistDiag) > 0:
                ghostDist = ghostDist + np.sqrt(2)*min(ghostDistDiag)
        if foodDist != 0:
            foodReward = 0.25*np.exp(-alpha*foodDist)/np.exp(alpha)
        # Reward being in sight of a capsule
        if capsuleDist != 0:
            capsuleReward = 1*np.exp(-alpha*capsuleDist)/np.exp(alpha)
        # Reward being in sight of a scared ghost
        if scaredDist != 0:
            if scaredDist < 0:
                scaredDist = -1
            scaredReward = 2*np.exp(-alpha*scaredDist)/np.exp(alpha)
        # Punish being in sight of a ghost
        if ghostDist != 0:
            if ghostDist < 0:
                ghostDist = -1
            ghostPunish = 5*np.exp(-alpha*ghostDist)/np.exp(alpha)

        # Reward winning the game
        winReward = 5 if state.isWin() else 0
        stationary_cost = -0.02
        reward = winReward + foodReward + capsuleReward + scaredReward - ghostPunish + stationary_cost
        return reward
    def updateScore(self, state:GameState)->float:
        """
        Updates the score of the agent, and gives the score change
        """
        current_score = state.getScore()
        scoreChange = current_score - self.last_score
        self.last_score = current_score
        return scoreChange