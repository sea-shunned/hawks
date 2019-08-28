import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import hawks

SEED_NUM = 42

# Fix the seed number
config = {
    "hawks": {
        "seed_num": SEED_NUM
    }
}
# Any missing parameters will take the default seen in configs/defaults.json
generator = hawks.create_generator(config)
# Run the generator
generator.run()
# Get the best dataset found and it's labels
datasets, label_sets = generator.get_best_dataset()
# Stored as a list for multiple runs
data, labels = datasets[0], label_sets[0]
# Run KMeans on the data
km = KMeans(
    n_clusters=len(np.unique(labels)), random_state=SEED_NUM
).fit(data)
# Get the Adjusted Rand Index for KMeans on the data
ari = adjusted_rand_score(labels, km.labels_)
print(f"ARI: {ari}")
