"""This script was used in our GECCO'19 `paper <https://doi.org/10.1145/3321707.3321761>`_. A range of plots are shown at the end (though only a subset can be found in the linked paper). The full results will be shown in my thesis, when it is released.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import hawks

NAME = "suite_exp"
SAVE_FOLDER = Path.cwd() / "hawks_experiments" / NAME
FILENAME = "suite_analysis_gecco19" # Base filename to important files (e.g. stats csv)
SEED_NUM = 0 # The magic number zero!
ANALYSIS_PATH = SAVE_FOLDER / (FILENAME+".csv")
LOAD_ANALYSIS = ANALYSIS_PATH.is_file()
LOAD_SUITE = SAVE_FOLDER.is_dir()

def process_QJ(files):
    """Custom function for loading QJ datasets
    """
    filenames = []
    datasets = []
    label_sets = []

    for file in files:
        filenames.append(file.name)
        data = np.genfromtxt(file, skip_header=1, delimiter=" ")
        datasets.append(data)
        labels = np.loadtxt(f"{file.parent / file.stem}.mem", delimiter="\n").astype(int)
        label_sets.append(labels)
    return filenames, datasets, label_sets

# Load the analysis if we have previously run it
if LOAD_ANALYSIS:
    print("Loading previous analysis")
    df = pd.read_csv(ANALYSIS_PATH, index_col=False)
# Otherwise perform it
else:
    # Define and run HAWKS if not previously done
    if not LOAD_SUITE:
        print("Generating datasets")
        config = {
            "hawks": {
                "folder_name": NAME,
                "save_best_data": True,
                "save_stats": True,
                "seed_num": SEED_NUM,
                "save_config": True,
                "num_runs": 30,
                # "comparison": "ranking"
            },
            "dataset": {
                "num_examples": 2000,
                "num_dims": [2, 20],
                "num_clusters": [5, 30],
                "min_clust_size": 20
            },
            "objectives": {
                "silhouette": {
                    "target": [0.2, 0.5, 0.8]
                }
            },
            "ga": {
                "num_gens": 100,
                "prob_fitness": 0.5,
                "elites": 0
            },
            "constraints": {
                "overlap": {
                    "threshold": 0.0,
                    "limit": "upper"
                },
                "eigenval_ratio": {
                    "threshold": 1,
                    "limit": "lower"
                }
            }
        }
        # Create the gen
        gen = hawks.create_generator(config)
        # Run the gen(s)
        gen.run()
    # Otherwise load it
    else:
        print("Loading datasets")
        gen = hawks.load_folder(SAVE_FOLDER)
    print("Analyzing datasets")
    # Analyse the HAWKS datasets
    df, _ = hawks.analysis.analyse_datasets(
        generator=gen,
        source="HAWKS",
        seed=SEED_NUM,
        save_folder=SAVE_FOLDER,
        filename=FILENAME+"_incomp"
    )
    print("HAWKS Done!")
    # Run on the HK datasets
    filenames, datasets, label_sets = hawks.load_datasets(
        Path("../other_generators/HK_data"),
        labels_last_column=True,
        glob_filter="*.dat"
    )
    df, _ = hawks.analysis.analyse_datasets(
        datasets=datasets,
        label_sets=label_sets,
        source="HK",
        seed=SEED_NUM,
        prev_df=df,
        save_folder=SAVE_FOLDER,
        filename=FILENAME+"_incomp"
    )
    print("HK Done!")
    # Run on the QJ datasets
    filenames, datasets, label_sets = hawks.load_datasets(
        Path("../other_generators/QJ_data"),
        custom_func=process_QJ,
        glob_filter="*.dat"
    )
    df, _ = hawks.analysis.analyse_datasets(
        datasets=datasets,
        label_sets=label_sets,
        source="QJ",
        seed=SEED_NUM,
        prev_df=df,
        save_folder=SAVE_FOLDER,
        filename=FILENAME
    )
    print("QJ Done!")

# Set style/font sizes etc.
sns.set_context("notebook")
sns.set_style("ticks")

# Plot the instance space according to algorithm
hawks.plotting.instance_space(
    df=df,
    color_highlight="algorithm",
    marker_highlight="source",
    show=False,
    filename="instance_space_gecco19",
    save_folder=SAVE_FOLDER,
    seed=SEED_NUM,
    alpha=0.6,
    cmap=sns.color_palette("cubehelix", 6),
    clean_props={
        "clean_labels": False,
        "clean_legend": True,
        "legend_loc": "center left"
    }
)

hawks.plotting.instance_space(
    df=df,
    color_highlight="source",
    marker_highlight="source",
    show=False,
    filename="instance_space_gecco19",
    save_folder=SAVE_FOLDER,
    seed=SEED_NUM,
    cmap=sns.color_palette("cubehelix", 3),
    alpha=0.6,
    clean_props={
        "clean_labels": False,
        "clean_legend": True
    }
)

hawks.plotting.scatter_plot(
    df=df.rename(columns={
        "f_silhouette": "silhouette",
        "f_overlap": "overlap"
    }),
    x="silhouette",
    y="overlap",
    show=False,
    fpath=SAVE_FOLDER / "suite_sw-olap",
    hue="source",
    style="source",
    s=35,
    alpha=0.7,
    clean_props={
        "clean_labels": True,
        "clean_legend": True
    }
)
# Prettify the names
temp_df = df.rename(
    columns={
        "f_silhouette": "Silhouette",
        "f_overlap": "Overlap"
    }
)
# Adjust the fontscale for these graphs
sns.set_context("notebook", font_scale=1.15)
# Split the silh vs overlap into multiple plots to better visualize
# Create the grid
g = sns.FacetGrid(temp_df, col="source", hue="source", palette="inferno", height=5)
# Use scatter for each
g.map(plt.scatter, "Silhouette", "Overlap", alpha=0.8, s=15)
# Set better titles
for ax, name in zip(g.axes.flatten(), ["HAWKS", "HK", "QJ"]):
    ax.set_title(name, fontdict={"fontsize":"x-large"})
hawks.plotting.save_plot(
    g.fig,
    SAVE_FOLDER / "instance_space_gecco19_sw-olap-3fig",
    fig_format="pdf"
)
plt.close(g.fig)
# Reset the font size
sns.set_context("notebook", font_scale=1)
# Get the problem features
problem_features = [col for col in df if col.startswith("f_")]
# Instance space for the problem features
for problem_feat in problem_features:
    # Special case, need to truncate legend
    if "dimensionality" in problem_feat:
        hawks.plotting.instance_space(
            df=df,
            color_highlight=problem_feat,
            marker_highlight="source",
            show=False,
            filename="instance_space_gecco19",
            save_folder=SAVE_FOLDER,
            seed=SEED_NUM,
            cmap="viridis",
            alpha=0.6,
            legend_type="brief",
            clean_props={
                "clean_labels": False,
                "clean_legend": True,
                "legend_loc": "center left",
                "legend_truncate": True
            }
        )
    else:
        hawks.plotting.instance_space(
            df=df,
            color_highlight=problem_feat,
            marker_highlight="source",
            show=False,
            filename="instance_space_gecco19",
            save_folder=SAVE_FOLDER,
            seed=SEED_NUM,
            cmap="viridis",
            alpha=0.6,
            legend_type="brief",
            clean_props={
                "clean_labels": False,
                "clean_legend": True,
                "legend_loc": "center left"
            }
        )
# Reshape the alg performance
output_df = df.melt(
    id_vars=[col for col in df if not col.startswith("c_")],
    value_vars=[col for col in df if col.startswith("c_")],
    var_name="Algorithm",
    value_name="ARI"
)
# Remove the c_ prefix to algorithm names
output_df['Algorithm'] = output_df['Algorithm'].map(lambda x: str(x)[2:])
# Plot the instance space according to best ARI
hawks.plotting.instance_space(
    df=output_df.loc[output_df.groupby(["source", "dataset_num"])["ARI"].idxmax()].reset_index(drop=True),
    color_highlight="ARI",
    marker_highlight="source",
    show=False,
    filename="instance_space_gecco19",
    save_folder=SAVE_FOLDER,
    seed=SEED_NUM,
    cmap="viridis",
    alpha=0.6,
    legend_type="brief",
    clean_props={
        "clean_labels": False,
        "clean_legend": True,
        "legend_truncate": True,
        "legend_loc": "center left"
    }
)
# Create boxplots for algorithm performance
sns.set_style("whitegrid")
hawks.plotting.create_boxplot(
    df=output_df,
    x="source",
    y="ARI",
    hue="Algorithm",
    cmap=sns.color_palette("colorblind"),
    xlabel="",
    ylabel="ARI",
    fpath=SAVE_FOLDER / f"{FILENAME}_clustering",
    hatching=True,
    clean_props={
        "clean_labels": False,
        "legend_loc": "center left"
    },
    fliersize=3
)
# Make the critical difference diagrams
hawks.plotting.cluster_alg_ranking(
    df=df[["source"]+[col for col in df if col.startswith("c_")]],
    save_folder=SAVE_FOLDER,
    filename=FILENAME+"-ranking"
)
