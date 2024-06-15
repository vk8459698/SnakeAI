from re import M
import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from snake_game import BLOCK_SIZE, SnakeGameAI, Direction, Point
from helper import plot

# how many previous moves will be stored in memory_deque
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

INPUT_SIZE = 11  # has to be the length of Agent.get_state
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3  # has to be the number of possible actions, Agent.get_action


class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate, has to be less than 1, usually 0.8-0.99
        # deque auto removes items if it gets larger than maxlen, popleft()
        self.memory_deque = deque(maxlen=MAX_MEMORY)

        self.model: Linear_QNet = Linear_QNet(
            INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer: QTrainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI):
        """From the game, get some parameters and returns a list
        0: If there is any danger Straight,
        1: If there is any danger Right,
        2: If there is any danger Left
        3: If current direction is left
        4: If current direction is right
        5: If current direction is up
        6: If current direction is down
        7: Is food on the left
        8: Is food on the right
        9: Is food up
        10: Is food down

        Args:
            game (SnakeGameAI): the game

        Returns:
            list: booleans for each game condition
        """
        head = game.snake[0]

        # get points around the head
        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)

        is_direction_right = game.direction == Direction.RIGHT
        is_direction_left = game.direction == Direction.LEFT
        is_direction_up = game.direction == Direction.UP
        is_direction_down = game.direction == Direction.DOWN

        state = [
            # Danger straight (same direction)
            (is_direction_right and game.is_collision(point_right)) or
            (is_direction_left and game.is_collision(point_left)) or
            (is_direction_up and game.is_collision(point_up)) or
            (is_direction_down and game.is_collision(point_down)),

            # Danger right (danger is at the right of current direction)
            (is_direction_right and game.is_collision(point_down)) or
            (is_direction_left and game.is_collision(point_up)) or
            (is_direction_up and game.is_collision(point_right)) or
            (is_direction_down and game.is_collision(point_left)),

            # Danger left (danger is at the left of current direction)
            (is_direction_right and game.is_collision(point_up)) or
            (is_direction_left and game.is_collision(point_down)) or
            (is_direction_up and game.is_collision(point_left)) or
            (is_direction_down and game.is_collision(point_right)),

            # current direction
            is_direction_right,
            is_direction_left,
            is_direction_up,
            is_direction_down,

            # food location
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        # memory_deque calls popleft automatically if size greater than MAX_MEMORY
        # store as a tuple containing all variables
        self.memory_deque.append(
            (state, action, reward, next_state, game_over))

    def train_long_memory(self):
        # trains in all the previous moves
        # increasing agent performance
        if len(self.memory_deque) > BATCH_SIZE:
            # list of tuples (state, action, reward, next_state, game_over)
            batch_sample = random.sample(self.memory_deque, BATCH_SIZE)
        else:
            batch_sample = self.memory_deque

        states, actions, rewards, next_states, game_overs = zip(*batch_sample)
        self.trainer.train_step(states, actions, rewards,
                                next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff between exploration vs exploitation
        self.epsilon = 80 - self.number_of_games
        action = [0, 0, 0]
        # in the beginning this is true for some time, later self.number_of_games is larger
        # than 80 and this will never get called
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            # prediction is a list of floats
            prediction = self.model.forward(state0)
            # get the larger number index
            move = torch.argmax(prediction).item()
            # set the index to 1
            action[move] = 1

        return action

    def get_play(self, state):
        self.model.eval()
        action = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model.forward(state0)
        move = torch.argmax(prediction).item()
        action[move] = 1

        return action


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        action = agent.get_action(state_old)

        # perform action, play game, and get new state
        reward, game_over, score = game.play_step(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(
            state_old, action, reward, state_new, game_over)

        # remember
        agent.remember(state_old, action, reward, state_new, game_over)

        if game_over:
            game.reset()
            agent.number_of_games += 1
            if score > best_score:
                best_score = score
                agent.model.save()

            print(
                f'Game {agent.number_of_games} Score {score} Record {best_score}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_of_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # train long memory (also called replay memory, or experience replay)
            agent.train_long_memory()


def main():
    agent = Agent()
    agent.model.load()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        action = agent.get_play(state_old)
        reward, game_over, score = game.play_step(action)
        if game_over:
            game.reset()


if __name__ == "__main__":
    # train()
    main()
