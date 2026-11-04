from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_results(rewards, defaults):
    window_size = 20

    moving_avg_rewards = np.convolve(
        rewards, np.ones(window_size) / window_size, mode="valid"
    )
    moving_avg_defaults = (
        np.convolve(defaults, np.ones(window_size) / window_size, mode="valid")
        * 100
    )

    style_name = (
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.style.use(style_name)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(rewards, alpha=0.3, color="dodgerblue", label="Episode Reward")
    ax1.plot(
        range(window_size - 1, len(rewards)),
        moving_avg_rewards,
        color="blue",
        linewidth=2,
        label=f"{window_size}-Ep Moving Average",
    )
    ax1.set_ylabel("Total Reward")
    ax1.set_title(
        "DQN Agent Performance: Dynamic Credit Limit Management",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(
        range(window_size - 1, len(defaults)),
        moving_avg_defaults,
        color="crimson",
        linewidth=2,
        label=f"Default Rate ({window_size}-Ep Moving Avg %)",
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Default Rate (%)")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rl_training_results.png"
    plt.savefig(output_path, dpi=300)
    print(f"[OK] Figure saved as '{output_path}'")
    plt.show()
