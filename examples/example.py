from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import hawks

# Fix the seed number
config = {
    "hawks": {
        "seed_num": 42
    }
}
# Any missing parameters will take the default seen in configs/defaults.json
generator = hawks.create_generator(config)
# Run the generator
generator.run()
# Get the best dataset found and it's labels
data, labels = generator.get_best_dataset()
# # Plot the best dataset to see how it looks
# generator.plot_best_indiv()
# Run KMeans on the data
km = KMeans(
    n_clusters=len(np.unique(labels)), random_state=42
).fit(data)
# Get the Adjusted Rand Index for KMeans on the data
ari = adjusted_rand_score(labels, km.labels_)
print(f"ARI: {ari}")
