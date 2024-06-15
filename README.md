# Snake AI

Summary
Exploration: The agent takes random actions with decreasing probability as more games are played.
Exploitation: The agent takes the best-known action based on its current knowledge (Q-values predicted by the neural network).
This balance helps the agent learn effectively by initially exploring the environment and later exploiting its knowledge to maximize rewards.






### Python setup
```in shell terminal execute these commands
pip install pygame
pip install torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib ipython
```

Run with `agent.py`.  
To train, uncomment `train()` from `agent.py`.
