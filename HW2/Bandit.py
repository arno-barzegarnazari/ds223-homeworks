"""
A/B Testing with Multi-Armed Bandits: Epsilon-Greedy vs Thompson Sampling.

Scenario
--------
Four advertisement options (bandits / arms) return Gaussian rewards with true
means ``BANDIT_REWARD = [1, 2, 3, 4]`` and a known precision ``TAU``.
Both algorithms run for ``NUM_TRIALS`` trials on identical arms, so their
results are directly comparable.

Outputs (saved in the ``outputs/`` folder next to this file)
------------------------------------------------------------
* ``bandit_rewards.csv``       - every trial: {Bandit, Reward, Algorithm}
* ``plot1_learning.png``       - learning process of each algorithm (linear & log)
* ``plot2_cumulative.png``     - cumulative rewards and regrets of both algorithms
* ``comparison_selection.png`` - how often each algorithm chose each ad

Run
---
    python Bandit.py
"""
############################### LOGGER
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# ============================== CONSTANTS ============================== #
BANDIT_REWARD = [1, 2, 3, 4] # true mean reward of each advertisement
NUM_TRIALS = 20_000 # number of trials per algorithm
EPSILON_0 = 5.0 # epsilon schedule: eps_t = min(1, EPSILON_0 / t)
TAU = 1.0 # known precision of rewards (variance = 1 / TAU)
PRIOR_MEAN = 0.0 # Thompson prior mean of mu (m0)
PRIOR_PRECISION = 0.01 # Thompson prior precision of mu (lambda0)
SEED = 12 # random seed for reproducibility

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CSV_PATH = OUTPUT_DIR / "bandit_rewards.csv"

# Plot palette: one fixed colour per algorithm, neutral grey for references
COLORS = {"EpsilonGreedy": "#2a78d6", "ThompsonSampling": "#eb6834"}
REFERENCE_COLOR = "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#b5b4ae",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "legend.frameon": False,
    "font.size": 10,
})


class Bandit(ABC):
    """Abstract base class for a multi-armed bandit algorithm.

    The abstract methods come from the assignment template and are kept
    unchanged. The concrete helper methods below them were added so that
    both algorithms share the same reward model, bookkeeping and reporting.
    """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        """Initialize the algorithm.

        :param p: true mean rewards of the arms
        :type p: list
        """
        pass

    @abstractmethod
    def __repr__(self):
        """Return a readable description of the algorithm."""
        pass

    @abstractmethod
    def pull(self):
        """Pull an arm and return the observed reward."""
        pass

    @abstractmethod
    def update(self):
        """Update the algorithm's beliefs after observing a reward."""
        pass

    @abstractmethod
    def experiment(self):
        """Run the full experiment over all trials."""
        pass

    @abstractmethod
    def report(self):
        """Store results in a CSV file and log the summary statistics."""
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

    # --------------- shared helpers (added to the template) --------------- #
    def _setup(self, p, num_trials, tau, seed):
        """Create the attributes shared by every algorithm.

        :param p: true mean rewards of the arms
        :type p: list
        :param num_trials: number of trials to run
        :type num_trials: int
        :param tau: known precision of the reward distribution
        :type tau: float
        :param seed: random seed for reproducibility
        :type seed: int
        """
        self.true_means = np.asarray(p, dtype=float)
        self.n_arms = len(self.true_means)
        self.num_trials = num_trials
        self.tau = tau
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.chosen_arms = np.zeros(num_trials, dtype=int)
        self.rewards = np.zeros(num_trials)

    def _draw_reward(self, arm):
        """Draw a Gaussian reward ``N(mu_arm, 1 / tau)`` for the given arm.

        :param arm: index of the arm
        :type arm: int
        :return: observed reward
        :rtype: float
        """
        return self.rng.normal(self.true_means[arm], 1 / np.sqrt(self.tau))

    @property
    def cumulative_reward(self):
        """Cumulative reward after each trial.

        :rtype: numpy.ndarray
        """
        return np.cumsum(self.rewards)

    @property
    def cumulative_regret(self):
        """Cumulative expected regret after each trial.

        Regret per trial = best true mean - true mean of the chosen arm.

        :rtype: numpy.ndarray
        """
        return np.cumsum(self.true_means.max() - self.true_means[self.chosen_arms])

    def to_dataframe(self):
        """Return the trial-level results as a DataFrame.

        :return: columns {Bandit, Reward, Algorithm}
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame({
            "Bandit": [f"Ad {arm + 1}" for arm in self.chosen_arms],
            "Reward": self.rewards,
            "Algorithm": self.__class__.__name__,
        })

    def _report(self, csv_path, append):
        """Save results to CSV and log the summary (used by ``report``).

        :param csv_path: path of the output CSV file
        :type csv_path: pathlib.Path
        :param append: append to an existing file instead of overwriting it
        :type append: bool
        :return: the trial-level results
        :rtype: pandas.DataFrame
        """
        name = self.__class__.__name__
        df = self.to_dataframe()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, mode="a" if append else "w",
                  header=not append, index=False)
        logger.debug(f"{name}: {len(df):,} rows written to {csv_path.name}")

        best_arm = int(self.true_means.argmax())
        logger.info(f"========== {name} ==========")
        logger.info(f"{name} | cumulative reward: {self.cumulative_reward[-1]:,.2f}")
        logger.info(f"{name} | cumulative regret: {self.cumulative_regret[-1]:,.2f}")
        logger.info(f"{name} | average reward per trial: {self.rewards.mean():.4f}")
        logger.info(f"{name} | average regret per trial: {self.cumulative_regret[-1] / self.num_trials:.4f}")
        logger.info(f"{name} | best ad (Ad {best_arm + 1}) chosen "
                    f"{self.counts[best_arm] / self.num_trials:.2%} of the time")
        logger.debug(f"{name} | pulls per ad: {self.counts.tolist()}")
        return df

#--------------------------------------#


class Visualization():
    """Plots that compare the two algorithms.

    :param eg: finished Epsilon-Greedy experiment
    :type eg: EpsilonGreedy
    :param ts: finished Thompson Sampling experiment
    :type ts: ThompsonSampling
    """

    def __init__(self, eg, ts):
        """Store the finished experiments and make sure the output folder exists."""
        self.algorithms = [eg, ts]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def plot1(self):
        """Visualize the learning process of each algorithm (linear and log scale).

        Plots the running average reward against the optimal reward,
        one row per algorithm, saved as ``plot1_learning.png``.
        """
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
        trials = np.arange(1, NUM_TRIALS + 1)
        optimal = max(BANDIT_REWARD)

        for row, algo in enumerate(self.algorithms):
            name = algo.__class__.__name__
            running_avg = algo.cumulative_reward / trials
            for col, scale in enumerate(["linear", "log"]):
                ax = axes[row, col]
                ax.plot(trials, running_avg, color=COLORS[name], lw=2,
                        label="Average reward")
                ax.axhline(optimal, color=REFERENCE_COLOR, ls="--", lw=1.2,
                           label=f"Optimal reward ({optimal})")
                ax.set_xscale(scale)
                ax.set_title(f"{name} - {scale} scale")
                ax.set_xlabel("Number of trials")
                ax.set_ylabel("Average reward")
                ax.legend(loc="lower right")

        fig.suptitle("Learning process: average reward converges to the best ad",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "plot1_learning.png", dpi=150)
        logger.debug("plot1 saved")

    def plot2(self):
        """Compare cumulative rewards and cumulative regrets of both algorithms.

        Saved as ``plot2_cumulative.png``.
        """
        fig, (ax_reward, ax_regret) = plt.subplots(1, 2, figsize=(13, 5))
        trials = np.arange(1, NUM_TRIALS + 1)

        for algo in self.algorithms:
            name = algo.__class__.__name__
            ax_reward.plot(trials, algo.cumulative_reward, color=COLORS[name],
                           lw=2, label=name)
            ax_regret.plot(trials, algo.cumulative_regret, color=COLORS[name],
                           lw=2, label=name)
            # direct label at the end of each regret line
            ax_regret.annotate(f"{algo.cumulative_regret[-1]:,.1f}",
                               (trials[-1], algo.cumulative_regret[-1]),
                               xytext=(5, 0), textcoords="offset points",
                               va="center", color="#0b0b0b")

        ax_reward.plot(trials, trials * max(BANDIT_REWARD), color=REFERENCE_COLOR,
                       ls="--", lw=1.2, label="Optimal (always best ad)")
        ax_reward.set_title("Cumulative reward")
        ax_reward.set_ylabel("Cumulative reward")
        ax_regret.set_title("Cumulative regret (lower is better)")
        ax_regret.set_ylabel("Cumulative regret")
        ax_reward.legend(loc="upper left")
        ax_regret.legend(loc="center right")
        for ax in (ax_reward, ax_regret):
            ax.set_xlabel("Number of trials")

        fig.suptitle("Epsilon-Greedy vs Thompson Sampling",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "plot2_cumulative.png", dpi=150)
        logger.debug("plot2 saved")

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """Epsilon-Greedy algorithm with epsilon decaying as 1/t.

    With probability ``eps_t = min(1, epsilon_0 / t)`` a random ad is shown
    (explore); otherwise the ad with the highest sample mean is shown (exploit).

    :param p: true mean rewards of the arms
    :type p: list
    :param epsilon_0: scale of the 1/t decay (``epsilon_0 = 1`` gives plain 1/t)
    :type epsilon_0: float
    :param num_trials: number of trials
    :type num_trials: int
    :param tau: known precision of rewards
    :type tau: float
    :param seed: random seed
    :type seed: int
    """

    def __init__(self, p, epsilon_0=EPSILON_0, num_trials=NUM_TRIALS,
                 tau=TAU, seed=SEED):
        """Initialize sample-mean estimates and counters."""
        self._setup(p, num_trials, tau, seed)
        self.epsilon_0 = epsilon_0
        self.estimates = np.zeros(self.n_arms)   # sample mean of each arm
        self.num_explored = 0

    def __repr__(self):
        """Return a readable description of the algorithm."""
        return (f"EpsilonGreedy(arms={self.true_means.tolist()}, "
                f"epsilon=min(1, {self.epsilon_0}/t), trials={self.num_trials})")

    def pull(self, arm):
        """Show an ad and observe its reward.

        :param arm: index of the arm to pull
        :type arm: int
        :return: observed reward
        :rtype: float
        """
        return self._draw_reward(arm)

    def update(self, arm, reward):
        """Update the sample mean of the pulled arm in O(1).

        :param arm: index of the pulled arm
        :type arm: int
        :param reward: observed reward
        :type reward: float
        """
        self.counts[arm] += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]

    def experiment(self):
        """Run Epsilon-Greedy for ``num_trials`` trials.

        :return: self, so calls can be chained
        :rtype: EpsilonGreedy
        """
        logger.info(f"Running {self!r}")
        for t in range(1, self.num_trials + 1):
            epsilon = min(1.0, self.epsilon_0 / t)
            if self.rng.random() < epsilon:
                arm = int(self.rng.integers(self.n_arms))   # explore
                self.num_explored += 1
            else:
                arm = int(np.argmax(self.estimates))         # exploit
            reward = self.pull(arm)
            self.update(arm, reward)
            self.chosen_arms[t - 1] = arm
            self.rewards[t - 1] = reward

        logger.debug(f"EpsilonGreedy explored {self.num_explored} times")
        logger.debug(f"EpsilonGreedy estimated means: {np.round(self.estimates, 3).tolist()}")
        return self

    def report(self, csv_path=CSV_PATH, append=False):
        """Store results in a CSV file and log cumulative reward and regret.

        :param csv_path: path of the output CSV file
        :type csv_path: pathlib.Path
        :param append: append to an existing CSV instead of overwriting it
        :type append: bool
        :return: the trial-level results
        :rtype: pandas.DataFrame
        """
        return self._report(csv_path, append)

#--------------------------------------#

class ThompsonSampling(Bandit):
    """Thompson Sampling for Gaussian rewards with known precision.

    Each ad's mean has a Normal prior ``N(m0, 1/lambda0)``. After observing
    rewards the posterior is ``N(m, 1/lambda)`` with
    ``lambda = lambda0 + tau * N`` and ``m = (tau * sum(x) + lambda0 * m0) / lambda``.
    Every trial, one value is sampled from each posterior and the ad with the
    largest sample is shown.

    :param p: true mean rewards of the arms
    :type p: list
    :param tau: known precision of rewards
    :type tau: float
    :param m0: prior mean of each arm's mean
    :type m0: float
    :param lambda0: prior precision of each arm's mean
    :type lambda0: float
    :param num_trials: number of trials
    :type num_trials: int
    :param seed: random seed
    :type seed: int
    """

    def __init__(self, p, tau=TAU, m0=PRIOR_MEAN, lambda0=PRIOR_PRECISION,
                 num_trials=NUM_TRIALS, seed=SEED):
        """Initialize the Normal prior of every arm."""
        self._setup(p, num_trials, tau, seed)
        self.m0, self.lambda0 = m0, lambda0
        self.m = np.full(self.n_arms, m0, dtype=float)              # posterior means
        self.lambda_ = np.full(self.n_arms, lambda0, dtype=float)   # posterior precisions
        self.sum_x = np.zeros(self.n_arms)                          # reward sums

    def __repr__(self):
        """Return a readable description of the algorithm."""
        return (f"ThompsonSampling(arms={self.true_means.tolist()}, tau={self.tau}, "
                f"prior=N({self.m0}, 1/{self.lambda0}), trials={self.num_trials})")

    def pull(self, arm):
        """Show an ad and observe its reward.

        :param arm: index of the arm to pull
        :type arm: int
        :return: observed reward
        :rtype: float
        """
        return self._draw_reward(arm)

    def sample(self):
        """Draw one sample from every arm's posterior.

        :return: one sampled mean per arm
        :rtype: numpy.ndarray
        """
        return self.rng.normal(self.m, 1 / np.sqrt(self.lambda_))

    def update(self, arm, reward):
        """Update the posterior of the pulled arm (Normal-Normal conjugacy).

        :param arm: index of the pulled arm
        :type arm: int
        :param reward: observed reward
        :type reward: float
        """
        self.counts[arm] += 1
        self.sum_x[arm] += reward
        self.lambda_[arm] = self.lambda0 + self.tau * self.counts[arm]
        self.m[arm] = (self.tau * self.sum_x[arm] + self.lambda0 * self.m0) / self.lambda_[arm]

    def experiment(self):
        """Run Thompson Sampling for ``num_trials`` trials.

        :return: self, so calls can be chained
        :rtype: ThompsonSampling
        """
        logger.info(f"Running {self!r}")
        for t in range(self.num_trials):
            arm = int(np.argmax(self.sample()))
            reward = self.pull(arm)
            self.update(arm, reward)
            self.chosen_arms[t] = arm
            self.rewards[t] = reward

        logger.debug(f"ThompsonSampling posterior means: {np.round(self.m, 3).tolist()}")
        return self

    def report(self, csv_path=CSV_PATH, append=False):
        """Store results in a CSV file and log cumulative reward and regret.

        :param csv_path: path of the output CSV file
        :type csv_path: pathlib.Path
        :param append: append to an existing CSV instead of overwriting it
        :type append: bool
        :return: the trial-level results
        :rtype: pandas.DataFrame
        """
        return self._report(csv_path, append)




def comparison(eg, ts):
    """Compare the two algorithms visually and in a summary table.

    Draws how many times each algorithm showed each ad (saved as
    ``comparison_selection.png``) and logs a side-by-side summary.

    :param eg: finished Epsilon-Greedy experiment
    :type eg: EpsilonGreedy
    :param ts: finished Thompson Sampling experiment
    :type ts: ThompsonSampling
    :return: summary table, one row per algorithm
    :rtype: pandas.DataFrame
    """
    # think of a way to compare the performances of the two algorithms VISUALLY and
    algorithms = [eg, ts]
    labels = [f"Ad {i + 1}\n(mean {m})" for i, m in enumerate(BANDIT_REWARD)]
    x = np.arange(len(BANDIT_REWARD))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, algo in enumerate(algorithms):
        name = algo.__class__.__name__
        bars = ax.bar(x + (i - 0.5) * width, algo.counts, width * 0.95,
                      color=COLORS[name], label=name)
        ax.bar_label(bars, labels=[f"{c:,}" for c in algo.counts],
                     padding=3, fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Times shown (log scale)")
    ax.set_title("How often each algorithm showed each ad")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_selection.png", dpi=150)

    summary = pd.DataFrame({
        "Cumulative reward": [a.cumulative_reward[-1] for a in algorithms],
        "Cumulative regret": [a.cumulative_regret[-1] for a in algorithms],
        "Best ad share": [a.counts[a.true_means.argmax()] / a.num_trials for a in algorithms],
    }, index=[a.__class__.__name__ for a in algorithms]).round(3)
    logger.info(f"Comparison summary:\n{summary.to_string()}")

    winner = summary["Cumulative regret"].idxmin()
    logger.success(f"{winner} achieved the lower cumulative regret")
    return summary


if __name__=='__main__':

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eg = EpsilonGreedy(BANDIT_REWARD).experiment()
    ts = ThompsonSampling(BANDIT_REWARD).experiment()

    eg.report(CSV_PATH)                 # creates a fresh CSV
    ts.report(CSV_PATH, append=True)    # adds Thompson Sampling rows
    logger.info(f"Rewards stored in {CSV_PATH}")

    viz = Visualization(eg, ts)
    viz.plot1()
    viz.plot2()
    comparison(eg, ts)

    logger.info(f"All figures saved in {OUTPUT_DIR}")
    plt.show()