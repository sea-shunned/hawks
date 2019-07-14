# HAWKS Data Generator

![HAWKS Animation](examples/hawks_anim.gif)

HAWKS is a tool for generating controllably difficult synthetic data, used primarily for clustering. This repo is associated with the following paper:

1. Shand, C, Allmendinger, R, Handl, J, Webb, A & Keane, J 2019, Evolving Controllably Difficult Datasets for Clustering. in Proceedings of the Annual Conference on Genetic and Evolutionary Computation (GECCO '19) . The Genetic and Evolutionary Computation Conference, Prague, Czech Republic, 13/07/19. [https://doi.org/10.1145/3321707.3321761](https://doi.org/10.1145/3321707.3321761)

The academic/technical details can be found there. What follows here is a practical guide to using this tool to generate synthetic data.

If you use this tool to generate data that forms part of a paper, please consider either linking to this work or citing the paper above.

## Installation
Installation is available through pip by:
```
pip install hawks
```
or by cloning this repo (and installing locally using `pip install .`). 

## Running HAWKS
Like any other package, you need to `import hawks` in order to use it. The parameters of hawks are configured via a config file system. Details of the parameters are found in the [user guide](https://github.com/sea-shunned/hawks/blob/master/user_guide.md). For any parameters that are not specified, default values will be used (as defined in `hawks/defaults.json`).

The example below illustrates how to run `hawks`. Either a dictionary or a path to a JSON config can be provided to override any of the default values.

```python
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
```

## User Guide
For a more detailed explanation of the parameters and how to use HAWKS, please read the [user guide](https://github.com/sea-shunned/hawks/blob/master/user_guide.md).

## Issues
As this work is still in development, plain sailing is not guaranteed. If you encounter an issue, first ensure that `hawks` is running as intended by navigating to the tests directory, and running `python tests.py`. If any test fails, please add details of this alongside your original problem to an issue on the [GitHub repo](https://github.com/sea-shunned/hawks/).

## Feature Requests
At present, this is primarily academic work, so future developments will be released here after they have been published. If you have any suggestions or simple feature requests for HAWKS as a tool to use, please raise that on the [GitHub repo](https://github.com/sea-shunned/hawks/).

If you are interested in extending this work or collaborating in an academic nature, please email cameron(dot)shand(at)manchester(dot)ac(dot)uk. 