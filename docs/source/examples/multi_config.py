"""Multi-config example script
"""
import hawks

# Create a multi-config
# Try three different silhouette width target values
# Try two different K (number of clusters) values
# 6 parameters combinations, 2 runs each = 12 runs
config = {
    "hawks": {
        "num_runs": 2,
        "seed_num": 42
    },
    "objectives": {
        "silhouette": {
            "target": [0.2, 0.5, 0.8]
        }
    },
    "dataset": {
        "num_clusters": [3, 10]
    }
}
# Any missing parameters will take from hawks/defaults.json
generator = hawks.create_generator(config)
# Run the generator
generator.run()
# Let's show the stats for the best individuals
print(generator.stats[generator.stats["best_indiv"] ==1])