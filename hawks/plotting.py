from pathlib import Path
from itertools import cycle
from copy import deepcopy
import textwrap

import numpy as np
from scipy.stats import norm, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.cm as cm
import seaborn as sns
import Orange.evaluation

# plt.style.use('seaborn-paper')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['axes.labelsize'] = 12

# sns.set_style("whitegrid")

def plot_pop(indivs, nrows=None, ncols=None, fpath=None, cmap="inferno", fig_format="png", global_seed=None, save=False, show=True, remove_axis=False, fig_title=None, **kwargs):
    # If no guidance is given, set it close to a square
    if nrows is None and ncols is None:
        # Get the square root for how many plots we need
        n = np.sqrt(len(indivs))
        # Rounds the number
        nrows = int(n)
        # Always take the ceil so we have either a square plot or slightly rectangular
        ncols = np.ceil(n).astype(int)
        # Ensure we have enough subplots
        while nrows * ncols < len(indivs):
            ncols += 1
    # Otherwise work it out
    elif nrows is None and ncols is not None:
        nrows = (len(indivs) // ncols) + 1
    elif nrows is not None and ncols is None:
        ncols = (len(indivs) // nrows) + 1
    # Create the subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
    try:
        axes_list = axes.flatten()
    # If we only have one subplot, don't need to flatten
    except AttributeError:
        axes_list = [axes]
    # Loop over the indivs
    for i, indiv in enumerate(indivs):
        # I don't like kwargs
        plot_indiv(
            indiv,
            ax=axes_list[i],
            multiple=True,
            cmap=cmap,
            global_seed=global_seed,
            remove_axis=remove_axis,
            **kwargs
        )
    # Clean up spare axes
    if (ncols * nrows) > len(indivs):
        spares = ncols * nrows - len(indivs)
        for ax in axes_list[-spares:]:
            ax.axis('off')
    if fig_title is not None:
        fig.suptitle(fig_title)
    # save the plot if specified
    if save or fpath is not None:
        save_plot(fig, fpath, fig_format)
    # Show the plot if specified
    if show:
        plt.show()
    # Close the figure (and its window)
    plt.close(fig)

def plot_indiv(indiv, ax=None, multiple=False, save=False, show=True, fpath=None, cmap="inferno", fig_format="png", global_seed=None, remove_axis=False, **kwargs):
    if multiple and ax is None:
        raise ValueError(f"An axis object must be supplied if plotting multiple indivs")
    # Create the figure and axis if called in isolation
    if not multiple:
        # Create the mpl objects
        fig, ax = plt.subplots(1, 1)
    try:
        # Get the colormap and apply
        cmap = cm.get_cmap(cmap)
        colors = cmap(np.linspace(0, 1, len(indiv)))
    # Not a valid colourmap
    except (ValueError, TypeError):
        # A list of colours may have been given
        if isinstance(cmap, list):
            # Not of right length
            if len(cmap) != len(indiv):
                raise ValueError(f"If a list of colours is given, must be of length {len(indiv)}")
            # Set the list of colours
            colors = cmap
        # Duplicate the colour, as cmap wasn't given
        elif matplotlib.colors.is_color_like(cmap):
            colors = [cmap] * len(indiv)
        # Check if a seaborn palette was given
        elif isinstance(cmap, str):
            try:
                colors = sns.color_palette(cmap)
            except ValueError:
                cmap = cm.get_cmap("inferno")
                colors = cmap(np.linspace(0, 1, len(indiv)))
        # Bad input, revert to default
        else:
            cmap = cm.get_cmap("inferno")
            colors = cmap(np.linspace(0, 1, len(indiv)))
    # Copy the indiv so we don't modify the original
    indiv = deepcopy(indiv)
    # Perform PCA if needed
    if indiv.all_values.shape[1] > 2:
        # Perform PCA to get something we can plot
        pca = PCA(
            n_components=2,
            random_state=global_seed
        )
        # Transform the data with the PCA
        indiv.all_values = pca.fit_transform(indiv.all_values)
        # Recreate the views so that the clusters represent the real value
        indiv.recreate_views()
    else:
        pca = None
    # Plot each cluster
    for i, cluster in enumerate(indiv):
        plot_cluster(ax, cluster, colors[i], pca=pca, **kwargs)
    # Remove the axis if need be
    if remove_axis:
        ax.axis('off')
    # Either save or show if called once
    if not multiple:
        # Save plot (or not)
        if save:
            save_plot(fig, fpath, fig_format)
        if show:
            plt.show()
        # Close the figure (and its window)
        plt.close(fig)

def plot_cluster(ax, cluster, color, add_patch=True, add_data=True, patch_color=None, hatch=None, pca=None, patch_alpha=0.25, point_size=20):
    # Add the patch to show the area of the Gaussian
    if add_patch:
        # Rotate the cov
        cov = cluster.rotate_cov()
        # Transform the cluster mean and cov if PCA is applied
        if pca is not None:
            P = pca.components_
            cluster.mean = P.dot(cluster.mean - pca.mean_)
            cov = P.dot(cov).dot(P.T)
        # Set patch color to be the same as the points if not set
        if patch_color is None:
            patch_color = color
        # Add the patch to the axes
        ax = add_ellipse(
            ax, cluster.mean, cov, patch_color,
            hatch, patch_alpha
        )
        # Needs to be called when adding patches
        # https://github.com/matplotlib/matplotlib/pull/3936
        ax.autoscale_view()
        ax.axis('equal')
    # Whether to add the data (or just plot ellipse)
    if add_data:
        ax.scatter(
            cluster.values[:, 0], cluster.values[:, 1],
            alpha=0.9, s=point_size, c=[color]
        )
    else:
        # We need the data to be there, just make it invisible
        ax.scatter(
            cluster.values[:, 0], cluster.values[:, 1],
            alpha=0, s=point_size, c=[color]
        )
    return ax

def cov_ellipse(cov, q=None, nsig=None):
    """ Creates an ellipse for a given covariance (with a given significance level)
    """
    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)

    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))
    return width, height, rotation

def add_ellipse(ax, mean, cov, patch_color='grey', hatch=None, patch_alpha=0.25):
    # Get the properties of the ellipse
    width, height, rotation = cov_ellipse(cov, nsig=3)
    # Add the patch with the provided args
    ax.add_patch(Ellipse(
        xy=(mean[0], mean[1]),
        width=width,
        height=height,
        angle=rotation,
        alpha=patch_alpha,
        facecolor=patch_color,
        edgecolor="k",
        hatch=hatch,
    ))
    return ax

def save_plot(fig, fpath, fig_format):
    if fig_format == "png":
        # Presentation style
        fig.savefig(
            f"{fpath}.{fig_format}",
            format=fig_format,
            transparent=False,
            bbox_inches='tight',
            pad_inches=0,
            dpi=200,
            figsize=(15, 10)
        )
    elif fig_format == "pdf":
        # Paper style
        fig.savefig(
            f"{fpath}.{fig_format}",
            format=fig_format,
            dpi=300,
            # transparent=True,
            bbox_inches='tight',
            figsize=(15, 10)
        )

def create_boxplot(df, x, y, cmap="viridis", xlabel=None, ylabel=None, fpath=None, show=False, fig_format="pdf", clean_props=None, hatching=False, **kwargs):
    # Create the fig and ax objects
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", None))
    loc = kwargs.pop("loc", "best")
    # Create the boxplot using seaborn
    ax = sns.boxplot(
        x=x,
        y=y,
        data=df,
        palette=cmap,
        ax=ax,
        **kwargs
    )
    if hatching:
        hatches = cycle(["/", "\\", ".", "x", "o", ""])
        # hatches = cycle(["/", "\\", ".", "x"])
        for i, box in enumerate(ax.patches):
            box.set_hatch(next(hatches)*3)
        for i, box in enumerate(ax.artists):
            box.set_hatch(next(hatches)*3)
    # Set labels if given
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # Format the graph
    if clean_props is not None:
        # Assume any input that isn't a dict signifies cleaning is desired
        if not isinstance(clean_props, dict):
            clean_props = {}
        # Add in the legend type if not passed (not used in sns.boxplot)
        if "legend_type" not in clean_props:
            clean_props["legend_type"] = "full"
        ax = clean_graph(ax, clean_props)
    # Save the graph if specified
    if fpath is not None:
        save_plot(fig, fpath, fig_format)
    # Show the graph if specified
    if show:
        plt.show()
    # Close the figure (and its window)
    plt.close(fig)

def instance_space(df, color_highlight, marker_highlight=None, show=True, save_folder=None, seed=None, filename="instance_space", cmap="inferno", make_heatmap=False, legend_type="brief", clean_props=None, **kwargs):
    # Make the fig and ax
    # Pop the figsize if provided, otherwise use default (None)
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", None))
    # Set the markers
    markers = cycle(["o", "D", "X", "v", "P", "s", "*", "^"])
    # Set the colours by unique items in the highlight
    # Select the problem feature columns/values
    feature_cols = [col for col in df if col.startswith("f_")]
    prob_feat_vals = df[feature_cols].values
    # PCA the problem features
    pca = PCA(n_components=2, random_state=seed)
    pca_feat_vals = pca.fit_transform(
        StandardScaler().fit_transform(prob_feat_vals)
    )
    # Add the PCs to the df
    df["PC1"] = pca_feat_vals[:, 0]
    df["PC2"] = pca_feat_vals[:, 1]
    # Convert column names
    if color_highlight is not None:
        if color_highlight.lower() == "generator":
            color_highlight = "source"
        elif color_highlight.lower() == "algorithm":
            color_highlight = "Algorithm"
    if marker_highlight is not None:
        if marker_highlight.lower() == "generator":
            marker_highlight = "source"
        elif marker_highlight.lower() == "algorithm":
            marker_highlight = "Algorithm"
    # Make the conversions needed for plotting clustering algorithms
    if color_highlight == "Algorithm" or marker_highlight == "Algorithm":
        all_algs = [col[2:] for col in df if col.startswith("c_")]
        # Reshape the df to put cluster algs into single column
        df = df.melt(
            id_vars=[col for col in df if not col.startswith("c_")],
            value_vars=[col for col in df if col.startswith("c_")],
            var_name="Algorithm",
            value_name="ARI"
        )
        # Remove the c_ prefix to algorithm names
        df['Algorithm'] = df['Algorithm'].map(lambda x: str(x)[2:])
        # Then we need to take the best algorithm for each dataset
        df = df.loc[df.groupby(["source", "dataset_num"])["ARI"].idxmax()].reset_index(drop=True)
        # if make_heatmap:
        #     fig2, ax2 = plt.subplots()
        #     # Count unique occurences, unstack it, and fill in missing values with 0
        #     norm_perf = df.groupby("source")["Algorithm"].value_counts(normalize=True, sort=False).unstack().fillna(0)
        #     # Create the heatmap
        #     ax2 = sns.heatmap(norm_perf, vmin=0, vmax=1, annot=True, square=False, ax=ax2)
        #     ax2.set_ylabel("")
        #     ax2.set_xlabel("")
        #     # ax2.set_yticks([i-0.5 for i in ax2.get_yticks()])
        #     ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
        #     # text.set_text(textwrap.fill(text.get_text(), 20))
        #     # import pdb; pdb.set_trace()
        #     ax2.set_xticklabels(
        #         [textwrap.fill(t.get_text(), 20) for t in ax2.get_xticklabels()]
        #     )
        #     ax2.set_ylim(len(norm_perf)+0.5, -0.5)
        #     fpath = Path(save_folder) / f"{filename}_heatmap"
        #     save_plot(fig2, fpath, fig_format="pdf")
    # Use seaborn's scatterplot for ease (otherwise groupby)
    ax = sns.scatterplot(
        x="PC1",
        y="PC2",
        data=df,
        hue=color_highlight,
        style=marker_highlight,
        palette=cmap,
        ax=ax,
        legend=legend_type, # Some scenarios may need brief, some full
        markers=markers,
        edgecolor="none", # Remove seaborn's white border
        **kwargs
    )
    # Format the graph
    if clean_props is not None:
        # Assume any input that isn't a dict signifies cleaning is desired
        if not isinstance(clean_props, dict):
            clean_props = {}
        # Specific fix for issues with these continuous values
        if "overlap" in color_highlight or "silhouette" in color_highlight:
            clean_props["legend_truncate"] = True
        # Add in the legend type
        clean_props["legend_type"] = legend_type
        ax = clean_graph(ax, clean_props)
    # Save if a location is given
    if save_folder is not None:
        # I like lowercase filenames, and remove f_/c_
        if marker_highlight is not None:
            marker_highlight = marker_highlight.lower().replace("c_","").replace("f_","")
        if color_highlight is not None:
            color_highlight = color_highlight.lower().replace("c_","").replace("f_","")
        # Construct filename using what has been varied
        if marker_highlight == color_highlight:
            fname = f"{filename}_{marker_highlight}"
        elif color_highlight is None and marker_highlight is not None:
            fname = f"{filename}_{marker_highlight}"
        elif marker_highlight is None and color_highlight is not None:
            fname = f"{filename}_{color_highlight}"
        else:
            fname = f"{filename}_{marker_highlight}-{color_highlight}"
        fpath = Path(save_folder) / fname
        save_plot(fig, fpath, fig_format="pdf")
    # Show the graph
    if show:
        plt.show()
    # Close the figure (and its window)
    plt.close(fig)

def convergence_plot(stats_df, y="fitness_silhouette", xlabel=None, ylabel=None, cmap="inferno", show=True, fpath=None, legend_type="brief", clean_props=None, **kwargs):
    # Create the fig and axes
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", None))
    loc = kwargs.pop("loc", "best")
    # Filter certain columns
    filter_list = ["gen", y, "run"]
    # Add the column to also separate on colour if specified
    if "hue" in kwargs:
        filter_list.append(kwargs["hue"])
    if "style" in kwargs:
        filter_list.append(kwargs["style"])
    # Filter the dataframe
    df = stats_df.filter(items=filter_list)
    # Make the lineplot
    ax = sns.lineplot(
        x="gen",
        y=y,
        data=df,
        ax=ax,
        palette=cmap,
        legend=legend_type,
        **kwargs
    )
    # Set labels if given
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # Format the graph
    if clean_props is not None:
        # Assume any input that isn't a dict signifies cleaning is desired
        if not isinstance(clean_props, dict):
            clean_props = {}
        # Add in the legend type
        clean_props["legend_type"] = legend_type
        ax = clean_graph(ax, clean_props)
    # Save if a filepath is given
    if fpath is not None:
        save_plot(fig, fpath, fig_format="pdf")
    # Show if specified
    if show:
        plt.show()
    # Close the figure (and its window)
    plt.close(fig)

def scatter_plot(df, x, y, cmap="inferno", show=True, fpath=None, clean_props=None, legend_type="full", **kwargs):
    # Create the fig and axes
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", None))
    # import pdb; pdb.set_trace()
    loc = kwargs.pop("loc", "best")
    # Call seaborn's scatter plot
    ax = sns.scatterplot(
        x=x,
        y=y,
        data=df,
        palette=cmap,
        ax=ax,
        legend=legend_type, # Some scenarios may need brief instead
        edgecolor="none", # Remove seaborn's white border
        **kwargs
    )
    # Format the graph
    if clean_props is not None:
        # Assume any input that isn't a dict signifies cleaning is desired
        if not isinstance(clean_props, dict):
            clean_props = {}
        # Add in the legend type
        clean_props["legend_type"] = legend_type
        ax = clean_graph(ax, clean_props)
    # Save if a filepath is given
    if fpath is not None:
        save_plot(fig, fpath, fig_format="pdf")
    # Show if specified
    if show:
        plt.show()
    # Close the figure (and its window)
    plt.close(fig)

def clean_graph(ax, clean_props=None):
    defaults = {
        "legend_type": "brief",
        "clean_legend": False,
        "legend_loc": "best",
        "legend_truncate": False,
        "truncate_amount": 4,
        "legend_wrap": True,
        "clean_labels": True
    }
    # Update defaults with any given values
    if clean_props is not None:
        for key, value in clean_props.items():
            defaults[key] = value
    if defaults["legend_type"] is not None:
        # Get the legend details
        handles, labels = ax.get_legend_handles_labels()
        # Modify the labels if desired
        if defaults["clean_legend"]:
            labels = [label.replace("_", " ") if label.isupper() else label.replace("_", " ").title() for label in labels]
        # Place the legend
        if defaults["legend_loc"] == "best":
            leg = ax.legend(handles, labels, loc="best")
        else:
            leg = ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.025, 0.5))
        # Truncate the legend (useful for long decimal numbers)
        if defaults["legend_truncate"]:
            for text in leg.texts:
                try:
                    txt = f"{float(text.get_text()):.2f}"
                except ValueError:
                    txt = text.get_text()
                # Remove leading f_ for features or c_ for alg
                if txt[:2].lower() == "f " or txt[:2].lower() == "c ":
                    txt = txt[2:]
                text.set_text(txt)
                # if isinstance(defaults["truncate_amount"], int):
                #     text.set_text(txt[:defaults["truncate_amount"]])
                # else:
                #     text.set_text(txt[:4])
        # Wrap the text if longer than 20 characters
        if defaults["legend_wrap"]:
            for text in leg.texts:
                text.set_text(textwrap.fill(text.get_text(), 20))
    # else:
    #     ax.get_legend().remove()
    # Title case the x and y labels
    if defaults["clean_labels"]:
        # Clean the y_axis label
        ax.set_ylabel(ax.get_ylabel().replace("_", " ").title())
        # Clean the x_axis label
        ax.set_xlabel(ax.get_xlabel().replace("_", " ").title())
    return ax

def cluster_alg_ranking(df, significance=0.05, show=False, save_folder=None, filename="alg-ranking"):
    # Replace the leading c_ for the algorithms
    df.columns = df.columns.str.replace("c_","")
    # Loop over the generators
    for source_name, group in df.groupby("source"):
        # Calculate the average ranks for each column (algorithm)
        avg_ranks = group.drop("source", axis=1).rank(axis=1, ascending=False).mean(0).values
        # Get the rows (N) and cols (k)
        rows, cols = group.drop("source", axis=1).shape
        # Calculate the friedman statistic
        friedman = (12*rows)/(cols*(cols+1)) * (np.sum(avg_ranks**2) - ((cols*(cols+1)**2)/4))
        # Get the associated p-value
        p = chi2.sf(friedman, cols-1)
        print(f"Source={source_name}, Friedman stat={friedman}, p-value={p}")
        # Check if this result is at the specified level of significance
        if p < significance:
            # Compute the critical distance
            cd = Orange.evaluation.compute_CD(
                avg_ranks, rows, alpha=str(significance), test="nemenyi"
            )
            # Plot the critical distance
            Orange.evaluation.graph_ranks(avg_ranks, group.columns[1:], cd=cd)
            # Get the figure created in the above
            fig = plt.gcf()
            # Save the results
            if save_folder is not None:
                fname = f"{filename}_{source_name}"
                fpath = Path(save_folder) / fname
                save_plot(fig, fpath, fig_format="pdf")
            if show or save_folder is None:
                plt.show()
        else:
            print(f"{source_name} has no statistically significant differences at p={significance}")
