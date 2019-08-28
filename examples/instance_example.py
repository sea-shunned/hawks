# Use a HAWKS and sklearn example, and show the space for HAWKS, moons, and blobs
from pathlib import Path
from sklearn.datasets import make_blobs, make_moons
import hawks

SEED_NUM = 10
SAVE_FOLDER = Path.cwd()
NUM_RUNS = 10 # May take a few minutes
NUM_CLUSTERS = 5

generator = hawks.create_generator({
    "hawks": {
        "seed_num": SEED_NUM,
        "num_runs": NUM_RUNS
    },
    "dataset": {
        "num_clusters": NUM_CLUSTERS
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
# Make the boxplot
hawks.plotting.instance_space(
    df=df,
    color_highlight="source",
    marker_highlight="source",
    show=True,
    save_folder=SAVE_FOLDER,
    filename="instance_space",
    seed=SEED_NUM,
    cmap="viridis"
)
