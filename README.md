***

# Minesweeper Energy-Based Model (EBM) Agent

This repository contains a PyTorch implementation of an Energy-Based Model (EBM) designed to play Minesweeper. By treating the game as a constrained probabilistic inference problem, the model uses "System 2" thinking—iterative, gradient-based sampling at inference time—to predict the locations of hidden mines.

This codebase is inspired by and based on the concepts presented in [arXiv:2507.02092v1](https://arxiv.org/html/2507.02092v1).

## 🚀 Performance & System 2 Scaling

The agent's performance scales with the amount of inference-time compute (System 2 thinking). By increasing the number of Langevin dynamics steps during sampling, the agent achieves better board coverage and higher win rates.

* **Advanced Density (9x9 board, 16 mines):** ~50% Win Rate
* **Beginner Density (9x9 board, 10 mines):** ~94% Win Rate 

---

## 🧠 How It Works

1.  **State Encoding:** The board is encoded into an 11-channel tensor representing revealed numbers (one-hot encoded), hidden cells, and the global mine density.
2.  **Energy Model:** A Convolutional Neural Network (`MinesweeperEnergyModel`) evaluates a joint configuration of the `board_state` and `candidate_mines`, outputting a scalar energy value. Lower energy represents a more likely configuration.
3.  **Inference (System 2):** The `EnergySampler` uses gradient descent to optimize candidate mine probabilities, iteratively refining the predictions to minimize the energy function output.
4.  **Agent:** The `EBMAgent` uses the sampler to predict mine probabilities and clicks the hidden cell with the lowest probability of containing a mine.

---

## 🛠️ Installation

You will need Python 3.8+ and the following dependencies:

```bash
pip install torch numpy matplotlib ipython
```

---

## 💻 Usage

The codebase is organized into modules for data generation, training, gameplay, and evaluation. Below is a standard workflow for using the code.

### 1. Generating a Training Dataset
Before training, you need to generate a dataset of random Minesweeper board states.

```python
from minesweeper_ebm import create_and_save_dataset

# Generates 100,000 board states and saves them to 'minesweeper_train.pt'
create_and_save_dataset(
    filepath="minesweeper_train.pt",
    num_samples=100_000,
    height=9,
    width=9,
    num_mines=16
)
```

### 2. Training the EBM
The model is trained using Contrastive Divergence, minimizing the energy of true mine placements while maximizing the energy of sampled, incorrect placements.

```python
from minesweeper_ebm import train

# Train the model; checkpoints will be saved in the 'checkpoints/' directory
model = train(
    data_path="minesweeper_train.pt",
    checkpoint_dir="checkpoints",
    epochs=50,
    batch_size=128,
    lr=3e-4,
    device_str="auto"
)
```

### 3. Evaluating and Benchmarking (System 2 Scaling)
You can evaluate how well your trained model performs as you allocate more compute steps to the sampling phase. 

```python
import torch
from minesweeper_ebm import load_agent_from_checkpoint, benchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_agent_from_checkpoint("checkpoints/ebm_epoch050.pt", device).model

# Run the benchmark to see how win rate scales with inference steps
results = benchmark(
    model=model,
    alpha_base=1.0,
    device=device,
    num_games=300,
    steps_list=[1, 5, 15, 30, 50]
)
```
*This will output a `system2_scaling.png` graph demonstrating the relationship between inference compute and agent success.*

### 4. Visualizing a Game
To watch the agent play a game step-by-step, use the built-in visualization tools. This requires an environment that supports `IPython.display` (like Jupyter Notebooks).

```python
from minesweeper_ebm import load_agent_from_checkpoint, visualise_game

agent = load_agent_from_checkpoint("checkpoints/ebm_epoch050.pt", device, inference_steps=30)

# Plays a single game and visualizes the board and probability heatmap
won = visualise_game(agent, height=9, width=9, num_mines=16, save_path="game_run")
print(f"Game won: {won}")
```

---

## 📂 Code Structure

* **`MinesweeperEnergyModel`:** The core convolutional architecture predicting the energy $E$ of a given board and mine arrangement.
* **`EnergySampler`:** Handles the iterative sampling of mine locations using continuous relaxation and gradients.
* **`MinesweeperGame`:** A fully featured python-based Minesweeper engine.
* **`EBMAgent`:** Wraps the model and sampler to interact with the game engine.
* **`ebm_train_step` & `train`:** Implementation of the training loop with contrastive divergence and dynamic noise scaling.

***

Would you like me to write a quick `requirements.txt` file or an `example.py` script that you can bundle alongside this README so users can test it instantly?
