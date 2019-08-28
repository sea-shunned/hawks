# Use a HAWKS and sklearn example, and show the performance (boxplot) for HAWKS, moons, and blobs
from pathlib import Path
from sklearn.datasets import make_blobs, make_moons
import hawks

SEED_NUM = 42
SAVE_FOLDER = Path.cwd()
NUM_RUNS = 5
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
# Get the clustering algorithms into one column
df = df.melt(
    id_vars=[col for col in df if not col.startswith("c_")],
    value_vars=[col for col in df if col.startswith("c_")],
    var_name="Algorithm",
    value_name="ARI"
)
# Remove the c_ prefix to algorithm names
df['Algorithm'] = df['Algorithm'].map(lambda x: str(x)[2:])
# Make the boxplot
hawks.plotting.create_boxplot(
    df=df,
    x="source",
    y="ARI",
    hue="Algorithm",
    show=True,
    fpath=SAVE_FOLDER / "clustering_performance",
    xlabel="Source"
)