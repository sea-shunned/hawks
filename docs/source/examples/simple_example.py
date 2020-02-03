"""Single, simple HAWKS run, with KMeans applied to the best dataset
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import hawks

# Set the magic seed number
SEED_NUM = 42
# Set the seed number in the config
config = {
    "hawks": {
        "folder_name": "simple_example",
        "seed_num": SEED_NUM
    },
    "dataset": {
        "num_clusters": 5
    },
    "objectives": {
        "silhouette": {
            "target": 0.9
        }
    }
}
# Any missing parameters will take from hawks/defaults.json
generator = hawks.create_generator(config)
# Run the generator
generator.run()
# Let's plot the best individual found
generator.plot_best_indivs(show=True)
# Get the best dataset found and it's labels
datasets, label_sets = generator.get_best_dataset()
# Stored as a list for multiple runs
data, labels = datasets[0], label_sets[0]
# Run KMeans on the data
km = KMeans(
    n_clusters=len(np.unique(labels)), random_state=SEED_NUM
).fit(data)
# Plot the output of KMeans
hawks.plotting.scatter_prediction(data, km.labels_)
# Get the Adjusted Rand Index for KMeans on the data
ari = adjusted_rand_score(labels, km.labels_)
print(f"ARI: {ari}")
