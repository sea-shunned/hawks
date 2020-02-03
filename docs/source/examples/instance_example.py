"""This example shows a quick instance space for some uninteresting datasets from HAWKS and :mod:`sklearn.datasets`.
"""
from sklearn.datasets import make_blobs, make_moons
import seaborn as sns

import hawks

SEED_NUM = 10
NUM_RUNS = 10 # May take a few minutes
NUM_CLUSTERS = 5

generator = hawks.create_generator({
    "hawks": {
        "seed_num": SEED_NUM,
        "num_runs": int(NUM_RUNS/2) # for parity
    },
    "dataset": {
        "num_clusters": NUM_CLUSTERS
    },
    "objectives": {
        "silhouette": {
            "target": [0.5, 0.9]
        }
    }
})
generator.run()
# Analyse the hawks datasets
df, _ = hawks.analysis.analyse_datasets(
    generator=generator,
    source="HAWKS",
    seed=SEED_NUM,
    save=False
)
# Make the blobs datasets
datasets = []
label_sets = []
for run in range(NUM_RUNS):
    data, labels = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=NUM_CLUSTERS,
        random_state=SEED_NUM+run
    )
    datasets.append(data)
    label_sets.append(labels)
# Analyse the blobs datasets
df, _ = hawks.analysis.analyse_datasets(
    datasets=datasets,
    label_sets=label_sets,
    source="SK-Blobs",
    seed=SEED_NUM,
    save=False,
    prev_df=df
)
# Make the moons datasets
datasets = []
label_sets = []
for run in range(NUM_RUNS):
    data, labels = make_moons(
        n_samples=1000,
        noise=2,
        random_state=SEED_NUM+run
    )
    datasets.append(data)
    label_sets.append(labels)
# Analyse the moons datasets
df, _ = hawks.analysis.analyse_datasets(
    datasets=datasets,
    label_sets=label_sets,
    source="SK-Moons",
    seed=SEED_NUM,
    save=False,
    prev_df=df
)
# Make the font etc. larger
sns.set_context("talk")
# Make the boxplot
hawks.plotting.instance_space(
    df=df,
    color_highlight="source",
    marker_highlight="source",
    show=True,
    seed=SEED_NUM,
    cmap=sns.cubehelix_palette(3)
)
