"""This example demonstrates generating convergence plots for the fitness and overlap (constraint).
"""
import seaborn as sns

import hawks

# Create the generator
generator = hawks.create_generator({
    "hawks": {
        "seed_num": 42,
        "num_runs": 5
    },
    "dataset": {
        "num_clusters": 5
    },
    "objectives": {
        "silhouette": {
            "target": 0.9
        }
    },
    "constraints": {
        "overlap": {
            "threshold": 0.05,
            "limit": "lower"
        }
    }
})
# Run HAWKS!
generator.run()
# Make a dictionary of options common to both plots
converg_kws = dict(
    show=True,
    xlabel="Generation",
    ci="sd",
    legend_type=None
)
# Make the font etc. larger
sns.set_context("talk")
# Plot the fitness (proximity to silhouette width target)
hawks.plotting.convergence_plot(
    generator.stats,
    y="fitness_silhouette",
    ylabel="Average Fitness",
    clean_props={
        "legend_loc": "center left"
    },
    **converg_kws
)
# Plot the overlap constraint
hawks.plotting.convergence_plot(
    generator.stats,
    y="overlap",
    clean_props={
        "clean_labels": True, # Capitalize the 'overlap' y-axis label
        "legend_loc": "center left"
    },
    **converg_kws
)
