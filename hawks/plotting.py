from pathlib import Path
from copy import deepcopy

import numpy as np
import matplotlib
from matplotlib.patches import Ellipse
from scipy.stats import norm, chi2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.style.use('seaborn-paper')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['axes.labelsize'] = 12

def plot_indiv(indiv, save=False, fname=None, cmap="rainbow", fig_format="png", global_seed=None):
    # Create the mpl objects
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    # Get the colormap and apply
    cmap = cm.get_cmap(cmap)
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
    # Save plot (or not)
    if save:
        plt.axis('off')
        save_plot(fig, fname, fig_format)
    else:
        plt.show()

def plot_cluster(ax, cluster, color, add_patch=True, add_data=True, patch_color="grey", hatch=None, pca=None):
    # Whether to add the data (or just plot ellipse)
    if add_data:
        ax.scatter(
            cluster.values[:, 0], cluster.values[:, 1],
            alpha=0.8, s=20, c=color
        )
    else:
        # We need the data to be there, just make it invisible
        ax.scatter(
            cluster.values[:, 0], cluster.values[:, 1],
            alpha=0, s=20, c=color
        )
    # Add the patch to show the area of the Gaussian
    if add_patch:
        # Rotate the cov
        cov = cluster.rotate_cov()
        # Transform the cluster mean and cov if PCA is applied
        if pca is not None:
            P = pca.components_
            cluster.mean = P.dot(cluster.mean - pca.mean_)
            cov = P.dot(cov).dot(P.T)
        # Add the patch to the axes
        ax = add_ellipse(ax, cluster.mean, cov, patch_color, hatch)
    return ax

def cov_ellipse(cov, q=None, nsig=None, **kwargs):
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
        alpha=0.3,
        facecolor=patch_color,
        edgecolor="k",
        hatch=hatch,
    ))
    return ax

def plot_population(pop, save=False, fig_format="png"):
    # Get number of rows and cols from user
    # (automated approach is so-so here, but difficult to fix well)
    nrows, ncols = input(
        'Please input "<nrows>,<ncols>" for graph: ').split(",")
    nrows, ncols = int(nrows), int(ncols)
    # Set up the fig/axes
    fig, axes = plt.subplots(nrows, ncols)
    # Create the colormap
    colors = cm.rainbow(np.linspace(0, 1, len(pop[0])))
    # Loop over the axes (flattening them to avoid index issues)
    for i, ax in enumerate(axes.flatten()):
        # Select the individual from the pop
        indiv = pop[i]
        # Loop over to get the clusters
        for j, cluster in enumerate(indiv):
            plot_cluster(ax, cluster, colors[j])
        ax.axis('off')
    # Save plot (or not)
    if save:
        save_plot(fig, "final_population", fig_format)
    else:
        plt.show()

def save_plot(fig, fname, fig_format):
    if fig_format == "png":
        # Presentation style
        fig.savefig(
            fname+"."+fig_format,
            format=fig_format,
            transparent=False,
            bbox_inches='tight',
            pad_inches=0,
            dpi=200,
            figsize=(18,12)
        )
    elif fig_format == "pdf":
        # Paper style
        fig.savefig(
            fname+"."+fig_format,
            format=fig_format,
            dpi=300,
            bbox_inches='tight',
            figsize=(18,12)
        )
