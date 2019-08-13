from pathlib import Path
from copy import deepcopy

import numpy as np
import matplotlib
from matplotlib.patches import Ellipse
from scipy.stats import norm, chi2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.cm as cm

plt.style.use('seaborn-paper')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['axes.labelsize'] = 12

def plot_pop(indivs, nrows=None, ncols=None, fpath=None, cmap="rainbow", fig_format="png", global_seed=None, save=False, show=True, remove_axis=False, fig_title=None):
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

def plot_indiv(indiv, ax=None, multiple=False, save=False, fpath=None, cmap="rainbow", fig_format="png", global_seed=None, remove_axis=False):
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
        else:
            plt.show()

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
            alpha=0.9, s=20, c=color
        )
    else:
        # We need the data to be there, just make it invisible
        ax.scatter(
            cluster.values[:, 0], cluster.values[:, 1],
            alpha=0, s=20, c=color
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
