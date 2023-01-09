import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from tqdm import tqdm

argparser = ArgumentParser()
argparser.add_argument(
    "-P",
    "--prey",
    default=[500, 500],
    type=int,
    nargs="+",
    help="Population size of prey species",
)
argparser.add_argument(
    "-T",
    "--toxicity",
    default=[0.9, 0.0],
    type=float,
    nargs="+",
    help="Toxicity of prey species",
)
argparser.add_argument(
    "-H",
    "--predator",
    default=100,
    type=int,
    help="Population size of predator species",
)
argparser.add_argument(
    "-G",
    "--genes",
    default=10,
    type=int,
    help="Length of genes",
)
argparser.add_argument(
    "-N",
    "--generations",
    default=1000,
    type=int,
    help="Number of generations",
)
argparser.add_argument(
    "-R",
    "--mutation_rate",
    default=0.05,
    type=float,
    help="Probability of mutation",
)
argparser.add_argument(
    "-K",
    "--mutation_scale",
    default=0.05,
    type=float,
    help="Scale of mutation",
)
argparser.add_argument(
    "-I",
    "--initial_scale",
    default=0.1,
    type=float,
    help="Initial scale of deviation",
)
argparser.add_argument(
    "-S",
    "--seed",
    default=0,
    type=int,
    help="Random seed",
)
argparser.add_argument(
    "-O",
    "--output",
    default="fig",
    type=str,
    help="Output directory",
)
argparser.add_argument(
    "-C",
    "--cmap",
    default="rainbow",
    type=str,
    help="color map to use",
)
args = argparser.parse_args()
print(args)

np.random.seed(args.seed)

DENSE_ALPHA = 0.2
MEDIUM_ALPHA = 0.5
FRAME_DPI = 150

cmap = get_cmap(args.cmap)

P: list[int] = args.prey
H: int = args.predator
T: list[float] = args.toxicity

if len(T) < len(P):
    T = T + [0.0] * (len(P) - len(T))
elif len(P) < len(T):
    T = T[: len(P)]

GENES: int = args.genes
GENERANTIONS: int = args.generations

PROB_MUT: float = args.mutation_rate
AMP_MUT: float = args.mutation_scale
AMP_INIT: float = args.initial_scale

PREY_SEEDS = np.random.normal(0, AMP_INIT, size=(len(P), GENES))
PREDATOR_SEED = np.random.normal(0, AMP_INIT, size=GENES)
PREYS = [
    np.random.normal(seed, AMP_MUT, size=(p, GENES)) for seed, p in zip(PREY_SEEDS, P)
]
PREDATOR = np.random.normal(PREDATOR_SEED, AMP_MUT, size=(H, GENES))

ROOT_PATH = args.output
FRAME_PATH = os.path.join(ROOT_PATH, "frames")

if not os.path.exists(FRAME_PATH):
    os.makedirs(FRAME_PATH)


def softmax(*args):
    return np.exp(np.clip(args, -500, 500)) / np.sum(np.exp(np.clip(args, -500, 500)))


def repopulate(curr, k, mean):
    global GENES, PROB_MUT, AMP_MUT
    if len(curr) == 0:
        curr = (np.random.uniform(size=(k, GENES)) < PROB_MUT) * np.random.normal(
            scale=AMP_MUT, size=(k, GENES)
        ) + mean
    else:
        PARENTS = curr[np.random.randint(len(curr), size=(2, k - len(curr))), :]
        MATE = np.random.uniform(size=(k - len(curr), GENES)) < 0.5
        NEW = (
            PARENTS[0] * MATE
            + PARENTS[1] * (1 - MATE)
            + (
                np.random.uniform(
                    size=(k - len(curr), GENES),
                )
                < PROB_MUT
            )
            * np.random.normal(scale=AMP_MUT, size=(k - len(curr), GENES))
        ) * (
            1
            + (
                np.random.uniform(
                    size=(k - len(curr), GENES),
                )
                < PROB_MUT
            )
            * np.random.normal(scale=AMP_MUT, size=(k - len(curr), GENES))
        )
        curr = np.vstack((curr, NEW))
    return curr


last_frame_pca_buffer = None


def plot(gen):
    global PREYS, PREDATOR, last_frame_pca_buffer, cmap
    p = PREDATOR.mean(axis=0)
    pca = PCA(n_components=2).fit(np.vstack(PREYS))
    pca_results = [pca.transform(pop) for pop in PREYS]
    if last_frame_pca_buffer is None:
        last_frame_pca_buffer = np.array([np.mean(res, axis=0) for res in pca_results])
        x_sign = 1
        y_sign = 1
    else:
        this_frame_pca_buffer = np.array([np.mean(res, axis=0) for res in pca_results])
        flip = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        x_sign, y_sign = flip[
            np.argmin(
                [
                    np.sum(
                        (last_frame_pca_buffer - this_frame_pca_buffer * f[None, :])
                        ** 2
                    )
                    for f in flip
                ]
            )
        ]
        last_frame_pca_buffer = this_frame_pca_buffer * np.array([[x_sign, y_sign]])
    fig = plt.figure(figsize=(6, 6))
    for i, result in enumerate(pca_results):
        plt.scatter(
            x_sign * result[:, 0],
            y_sign * result[:, 1],
            color=cmap(i / (len(pca_results) - 1)),
            alpha=DENSE_ALPHA,
            label=f"Prey {i}",
        )
    X = x_sign * np.linspace(*plt.xlim(), 10)
    Y = y_sign * np.linspace(*plt.ylim(), 10)
    X, Y = np.meshgrid(X, Y)
    genes = pca.inverse_transform(np.vstack((X.ravel(), Y.ravel())).T)
    Z = softmax(np.dot(genes, p)).reshape(X.shape)
    plt.imshow(
        Z,
        cmap="gray_r",
        alpha=MEDIUM_ALPHA,
        extent=[*plt.xlim(), *plt.ylim()],
        origin="lower",
        aspect="auto",
    )
    plt.title(f"Generation {gen}")
    plt.legend(loc="upper center", ncol=min(len(PREYS), 5))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(FRAME_PATH, f"GEN-{gen}.png"), dpi=FRAME_DPI)
    plt.close()

    prey_means = np.array([np.mean(p, axis=0) for p in PREYS])
    pred_mean = np.mean(PREDATOR, axis=0)
    prey_dist_mat = np.sqrt(
        np.sum((prey_means[:, None, :] - prey_means[None, :, :]) ** 2, axis=-1)
    )
    prey_pref_vec = softmax(*(prey_means @ pred_mean))
    fig, axs = plt.subplots(
        figsize=(len(P) + 1, len(P)),
        nrows=1,
        ncols=2,
        sharey=True,
        gridspec_kw={"width_ratios": [len(P), 1]},
    )
    dist_cmap = get_cmap("coolwarm")
    dist_cmap.set_bad("w")
    dist_im = axs[0].imshow(
        np.ma.array(prey_dist_mat, mask=np.tri(prey_dist_mat.shape[0], k=-1)),
        cmap=dist_cmap,
    )
    axs[0].set_xticks(np.arange(len(P)), labels=[f"T{t:.01f}" for p, t in zip(P, T)])
    axs[0].set_yticks(np.arange(len(P)), labels=[f"T{t:.01f}" for p, t in zip(P, T)])
    plt.colorbar(dist_im)
    plt.setp(axs[0].get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    # axs[0].set_title("Distance")

    pref_im = axs[1].imshow(prey_pref_vec[:, None], cmap="coolwarm", vmin=0.0, vmax=1.0)
    axs[1].set_xticks([])
    plt.colorbar(pref_im)
    # axs[1].set_title("Preference")

    plt.suptitle(f"Generation {gen}")
    plt.tight_layout()
    plt.savefig(os.path.join(FRAME_PATH, f"DIST-{gen}.png"), dpi=FRAME_DPI)
    plt.close()


SURVIVAL_RATES = list()
POP_HISTORY = list()

for gen in tqdm(range(GENERANTIONS)):
    # Plot the population
    plot(gen)
    # Record mean population in case of death
    PREY_MEANS = [pop.mean(axis=0) for pop in PREYS]
    PREDATOR_MEAN = PREDATOR.mean(axis=0)
    POP_HISTORY.append((*PREY_MEANS, PREDATOR_MEAN))
    # Each predator hunts preys
    for p in PREDATOR:
        preference = softmax(*np.concatenate([np.dot(pop, p) for pop in PREYS]))
        idx = np.random.choice(sum(map(len, PREYS)), p=preference)
        cumsum = np.cumsum(list(map(len, PREYS)))
        pop_idx = (idx < cumsum).argmax()
        # prey eaten by predator
        PREYS[pop_idx] = np.delete(
            PREYS[pop_idx], idx - (cumsum[pop_idx - 1] if pop_idx else 0), axis=0
        )
        if np.random.uniform() < T[pop_idx]:
            PREDATOR = np.delete(
                PREDATOR, np.argmin(np.sum((PREDATOR - p) ** 2, axis=1)), axis=0
            )
    # Survival rates
    SR = [len(PREYS[i]) / P[i] for i in range(len(P))] + [len(PREDATOR) / H]
    SURVIVAL_RATES.append(SR)
    # Repopulate
    for i in range(len(P)):
        if len(PREYS[i]) < P[i]:
            PREYS[i] = repopulate(PREYS[i], P[i], PREY_MEANS[i])
    if len(PREDATOR) < H:
        PREDATOR = repopulate(PREDATOR, H, PREDATOR_MEAN)

PREY_MEANS = [pop.mean(axis=0) for pop in PREYS]
PREDATOR_MEAN = PREDATOR.mean(axis=0)
POP_HISTORY.append((*PREY_MEANS, PREDATOR_MEAN))
plot(GENERANTIONS)

# Plot survival rates
SURVIVAL_RATES = np.array(SURVIVAL_RATES)
for i in range(len(P)):
    plt.plot(
        SURVIVAL_RATES[:, i],
        label=f"Prey {i}",
        color=cmap(i / (len(P) - 1)),
        alpha=MEDIUM_ALPHA,
    )
plt.plot(SURVIVAL_RATES[:, len(P)], label=f"Predator", color="k", alpha=MEDIUM_ALPHA)
plt.legend()
plt.savefig(os.path.join(ROOT_PATH, "Survival.pdf"))
plt.close()
pd.DataFrame(SURVIVAL_RATES).rename(
    columns={i: f"Prey {i}" for i in range(len(P))} | {len(P): "Predator"}
).to_csv(os.path.join(ROOT_PATH, "Survival.csv"))

# Plot population evolution history
POP_HISTORY = np.array(POP_HISTORY)
pca = PCA(n_components=2).fit(POP_HISTORY[:, :-1, :].reshape(-1, GENES))

for i in range(len(P)):
    pop_history = pca.transform(POP_HISTORY[:, i, :])
    plt.plot(
        pop_history[:, 0],
        pop_history[:, 1],
        label=f"Prey ({P[i]}; {T[i]:.01f})",
        color=cmap(i / (len(P) - 1)),
        alpha=MEDIUM_ALPHA,
    )
    plt.plot(
        pop_history[:: len(pop_history) // 10, 0],
        pop_history[:: len(pop_history) // 10, 1],
        marker=".",
        color=cmap(i / (len(P) - 1)),
    )
    plt.scatter(
        pop_history[0, 0],
        pop_history[0, 1],
        marker="o",
        color=cmap(i / (len(P) - 1)),
    )
    plt.scatter(
        pop_history[-1, 0],
        pop_history[-1, 1],
        marker="X",
        color=cmap(i / (len(P) - 1)),
    )
preferences = np.array(
    [
        softmax(*np.dot(pop, pred))
        for pop, pred in zip(POP_HISTORY[:, :-1, :], POP_HISTORY[:, -1, :])
    ]
)
pred_target = (preferences[:, None, :] @ POP_HISTORY[:, :-1, :]).squeeze()
pop_history = pca.transform(pred_target)
plt.plot(
    pop_history[:, 0],
    pop_history[:, 1],
    label=f"Preference ({H})",
    color="k",
    alpha=MEDIUM_ALPHA,
)
plt.plot(
    pop_history[:: len(pop_history) // 10, 0],
    pop_history[:: len(pop_history) // 10, 1],
    marker=".",
    color="k",
)
plt.scatter(
    pop_history[0, 0],
    pop_history[0, 1],
    marker="o",
    color="k",
)
plt.scatter(
    pop_history[-1, 0],
    pop_history[-1, 1],
    marker="X",
    color="k",
)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT_PATH, "Evolution.pdf"))
plt.close()
pd.DataFrame(
    {
        (t, f"Prey {i}" if i != len(history) - 1 else "Predator"): p
        for t, history in enumerate(POP_HISTORY)
        for i, p in enumerate(history)
    }
).T.to_csv(os.path.join(ROOT_PATH, "Evolution.csv"))
