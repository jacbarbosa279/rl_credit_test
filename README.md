# Credit Limit Management with Deep Reinforcement Learning

A Deep Q-Network (DQN) agent that learns credit-limit management strategies in
a simulated customer environment. The simulation tracks credit utilization,
delinquency, and credit score over episodes of up to 12 steps.

## Architecture

The GitHub-rendered diagram below shows the main runtime flow. The agent
interacts with the simulated environment, learns from rewards, and sends
episode metrics to the plotting utility.

```mermaid
flowchart TD
    A["src/main.py<br/>Application entry point"] --> B["src/services/training.py<br/>DQN training loop"]
    B --> C["src/models/credit.py<br/>CreditLimitEnv"]
    B --> D["src/models/credit.py<br/>QNetwork"]
    C -->|"State, reward, done"| B
    B -->|"Select action"| C
    D -->|"Q-values for actions"| B
    B -->|"Rewards and default history"| E["src/utils/plotting.py<br/>Metrics and charts"]
    E --> F["output/rl_training_results.png"]
    G["tests/"] -.->|"Validates environment and network"| C
    G -.-> D
```

### Project structure

```text
.
├── output/                 # Generated training charts
├── src/
│   ├── main.py             # Application entry point
│   ├── models/
│   │   └── credit.py       # Simulated environment and Q-network
│   ├── services/
│   │   └── training.py     # DQN training loop
│   └── utils/
│       └── plotting.py     # Training metric visualization
├── tests/                  # Automated tests
├── Dockerfile              # Container image definition
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

Python 3.9 or later. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Run the training

From the repository root, run:

```bash
python -m src.main
```

The training run displays a chart and saves it to
`output/rl_training_results.png`. The `output/` directory is created
automatically when needed.

## Run with Docker

Build the image from the repository root:

```bash
docker build -t rl-credit-agent .
```

Run the training and save its chart to the local `output/` directory:

```powershell
docker run --rm -v "${PWD}\output:/app/output" rl-credit-agent
```

## Run the tests

```bash
python -m pytest
```
