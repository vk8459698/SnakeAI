import streamlit as st
import torch
import numpy as np
from snake_game import SnakeGameAI, Direction, Point
from agent import Agent

# Initialize the game and agent
game = SnakeGameAI()
agent = Agent()

# Load the trained model
agent.model.load()

# Define a Streamlit layout
st.title("Snake Game AI")
st.write("This is a reinforcement learning agent playing Snake.")
game_placeholder = st.empty()
score_placeholder = st.empty()

# Function to render the game state
def render_game(game):
    game._update_ui()
    img = pygame.surfarray.array3d(game.display)
    img = img.swapaxes(0, 1)
    return img

# Main loop to display the game step by step
while True:
    state_old = agent.get_state(game)
    action = agent.get_play(state_old)
    reward, game_over, score = game.play_step(action)

    # Render the game
    game_image = render_game(game)
    game_placeholder.image(game_image)

    # Update score
    score_placeholder.write(f"Score: {score}")

    if game_over:
        game.reset()
        st.write("Game Over. Restarting...")

        if st.button("Reset Game"):
            game.reset()
            agent.number_of_games += 1
            score_placeholder.write(f"Score: {score}")
        else:
            break
