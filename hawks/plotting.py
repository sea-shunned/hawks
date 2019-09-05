from pathlib import Path
from itertools import cycle
from copy import deepcopy

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

plt.style.use('seaborn-paper')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['axes.labelsize'] = 12

sns.set_style("whitegrid")

def plot_pop(indivs, nrows=None, ncols=None, fpath=None, cmap="inferno", fig_format="png", global_seed=None, save=False, show=True, remove_axis=False, fig_title=None):
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
            remove_axis=remove_axis
        )
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

def plot_indiv(indiv, ax=None, multiple=False, save=False, show=True, fpath=None, cmap="inferno", fig_format="png", global_seed=None, remove_axis=False):
    if multiple and ax is None:
        raise ValueError(f"An axis object must be supplied if plotting multiple indivs")
    # Create the figure and axis if called in isolation
    if not multiple:
        # Create the mpl objects
        fig, ax = plt.subplots(1, 1)
    # Get the colormap and apply
    cmap = cm.get_cmap(cmap)
    # colors = cmap(np.linspace(0, 1, len(indiv)+1))
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
        plot_cluster(ax, cluster, colors[i], pca=pca)
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

def plot_cluster(ax, cluster, color, add_patch=True, add_data=True, patch_color=None, hatch=None, pca=None):
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
        ax = add_ellipse(ax, cluster.mean, cov, patch_color, hatch)
        # Needs to be called when adding patches
        # https://github.com/matplotlib/matplotlib/pull/3936
        ax.autoscale_view()
    # Whether to add the data (or just plot ellipse)
    if add_data:
        ax.scatter(
            cluster.values[:, 0], cluster.values[:, 1],
            alpha=0.9, s=20, c=[color]
        )
    else:
        # We need the data to be there, just make it invisible
        ax.scatter(
            cluster.values[:, 0], cluster.values[:, 1],
            alpha=0, s=20, c=[color]
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

def add_ellipse(ax, mean, cov, patch_color='grey', hatch=None):
    # Get the properties of the ellipse
    width, height, rotation = cov_ellipse(cov, nsig=3)
    # Add the patch with the provided args
    ax.add_patch(Ellipse(
        xy=(mean[0], mean[1]),
        width=width,
        height=height,
        angle=rotation,
        alpha=0.25,
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
            transparent=True,
            bbox_inches='tight',
            figsize=(15, 10)
        )

def create_boxplot(df, x, y, cmap="viridis", xlabel=None, ylabel=None, fpath=None, show=False, fig_format="pdf", **kwargs):
    # Create the fig and ax objects
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", None))
    # Create the boxplot using seaborn
    ax = sns.boxplot(
        x=x,
        y=y,
        data=df,
        palette=cmap,
        ax=ax,
        **kwargs
    )
    # Set labels if given
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # Save the graph if specified
    if fpath is not None:
        save_plot(fig, fpath, fig_format)
    # Show the graph if specified
    if show:
        plt.show()

def instance_space(df, color_highlight, marker_highlight=None, show=True, save_folder=None, seed=None, filename="instance_space", cmap="inferno", **kwargs):
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
    if color_highlight.lower() == "generator":
        color_highlight = "source"
    elif color_highlight.lower() == "algorithm":
        color_highlight = "Algorithm"
    if marker_highlight is not None and marker_highlight.lower() == "generator":
        marker_highlight = "source"
    elif marker_highlight is not None and marker_highlight.lower() == "algorithm":
        marker_highlight = "Algorithm"
    # Make the conversions needed for plotting clustering algorithms
    if color_highlight == "Algorithm" or marker_highlight == "Algorithm":
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
    # Use seaborn's scatterplot for ease (otherwise groupby)
    ax = sns.scatterplot(
        x="PC1",
        y="PC2",
        data=df,
        hue=color_highlight,
        style=marker_highlight,
        palette=cmap,
        ax=ax,
        legend="full", # Some scenarios may need brief instead
        markers=markers,
        edgecolor="none", # Remove seaborn's white border
        s=20,
        **kwargs
    )
    # Save if a location is given
    if save_folder is not None:
        # Construct filename using what has been varied
        if marker_highlight == color_highlight:
            fname = f"{filename}_{marker_highlight}"
        else:
            fname = f"{filename}_{marker_highlight}-{color_highlight}"
        fpath = Path(save_folder) / fname
        save_plot(fig, fpath, fig_format="pdf")
    # Show the graph
    if show:
        plt.show()
    # Close the figure (and its window)
    plt.close(fig)
