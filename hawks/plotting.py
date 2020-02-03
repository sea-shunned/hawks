"""Defines all the functions for plotting, allowing easier generation of results. Flexible, general functions remain a constant issue with plotting, so for more complex plots some tweaking may be needed.

Examples of these functions can be found in the :ref:`plotting_page` guide.
"""

from pathlib import Path
from itertools import cycle
from copy import deepcopy
import textwrap
import pickle

import numpy as np
import pandas as pd
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


def plot_pop(indivs, nrows=None, ncols=None, fpath=None, cmap="inferno", fig_format="png", global_seed=None, save=False, show=True, remove_axis=False, fig_title=None, **kwargs):
    """Plotting a population of individuals. Wrapper function for :func:`~hawks.plotting.plot_indiv`.

    The ``nrows`` and ``ncols`` options allow for specification of the layout of the plotting grid. If not given, it's made as square as possible.

    Args:
        indivs (list): A list of individuals (:class:`~hawks.genotype.Genotype`) to be plotted.
        nrows (int, optional): Number of rows for the plots. Defaults to None.
        ncols (int, optional): Number of columns for the plots. Defaults to None.
        fpath (:obj:`str`, :class:`pathlib.Path`, optional): The path to save the plot in. Defaults to None.
        cmap (str, optional): A colourmap for the plots. See `here <https://matplotlib.org/tutorials/colors/colormaps.html>`_ for options. Defaults to "inferno".
        fig_format (str, optional): Whether to save the plot as a "png" or as a "pdf". Defaults to "png".
        global_seed (int, optional): Seed used for PCA if the data is more than 2-dimensions. Defaults to None.
        save (bool, optional): Whether to save the plot or not. Defaults to False.
        show (bool, optional): Whether to show the plot or not. Defaults to True.
        remove_axis (bool, optional): Whether to remove the axis lines (and just show the data). Defaults to False.
        fig_title (str, optional): Adds a title to the figure if desired. Defaults to None.
    """
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
    """Function to plot a single individual. Sequentially calls :func:`hawks.plotting.plot_cluster`. PCA is applied if the data is more than 2-dimensions.
    
    Args:
        indiv (:class:`~hawks.genotype.Genotype`): A single individual to be plotted.
        ax (:mod:`matplotlib.axes`, optional): The axis object to use. Defaults to None, where it is created.
        multiple (bool, optional): Whether multiple plots are being made (i.e. adding a subplot onto a larger plot). Defaults to False.
        save (bool, optional): Whether to save the plot or not. Defaults to False.
        show (bool, optional): Whether to show the plot or not. Defaults to True.
        fpath (:obj:`str`, :class:`pathlib.Path`, optional): The path to save the plot in. Defaults to None.
        cmap (str, optional): A colourmap for the plots. See `here <https://matplotlib.org/tutorials/colors/colormaps.html>`_ for options. Defaults to "inferno".
        fig_format (str, optional): Whether to save the plot as a "png" or as a "pdf". Defaults to "png".
        global_seed (int, optional): Seed used for PCA if the data is more than 2-dimensions. Defaults to None.
        remove_axis (bool, optional): Whether to remove the axis lines (and just show the data). Defaults to False.
    """
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
    """Function to plot a single cluster.

    Args:
        ax (:mod:`matplotlib.axes`): The axis object to use.
        cluster (:class:`~hawks.cluster.Cluster`): A single cluster to plot.
        color (list): The colour to be used for this cluster. The type can vary based on what was used to generate the colours, but must be accepted by ``matplotlib``.
        add_patch (bool, optional): Whether to add an ellipse that shows the true cluster boundary. Defaults to True.
        add_data (bool, optional): Whether to add the actual data/samples for the cluster. Defaults to True.
        patch_color (list, optional): The colour to use for the cluster boundary (must be in an accepted format for ``matplotlib``). Defaults to None.
        hatch (str, optional): Whether to hatch the ellipse. Defaults to None.
        pca (:class:`sklearn.decomposition.PCA`, optional): The PCA object, use to transform the data if it exists. Defaults to None.
        patch_alpha (float, optional): Transparency of the ellipse. Defaults to 0.25.
        point_size (int, optional): Size of the data points. Defaults to 20.

    Returns:
        :mod:`matplotlib.axes`: The axis object with the cluster added.
    """
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
    """Creates an ellipse for a given covariance (with a given significance level).

    Args:
        cov (:class:`numpy.ndarray`): The covariance matrix.
        q (float, optional): The amount of variance to account for. Defaults to None.
        nsig (int, optional): The number of confidence intervals to use. Defaults to None.

    Returns:
        tuple: A 3-element tuple containing:

            width (float): The width of the ellipse.

            height (float): The height of the ellipse.

            rotation (float): The rotation of the ellipse.
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
    """Save the given :class:`matplotlib.figure.Figure`.
    
    Args:
        fig (:class:`~matplotlib.figure.Figure`): The figure to be saved.
        fpath (:obj:`str`, :class:`pathlib.Path`): The path (full path if a folder other than the working directory is to be used) to save the figure.
        fig_format (str): The format of the figure to save in, either 'png' or 'pdf'.
    """
    if fig_format == "png":
        # Presentation style
        fig.savefig(
            f"{fpath}.{fig_format}",
            format=fig_format,
            transparent=False,
            bbox_inches='tight',
            pad_inches=0,
            dpi=600,
            figsize=(15, 10)
        )
    elif fig_format == "pdf":
        # Paper style
        fig.savefig(
            f"{fpath}.{fig_format}",
            format=fig_format,
            dpi=300,
            # transparent=True,
            # figsize=(15, 10),
            bbox_inches='tight'
        )

def create_boxplot(df, x, y, cmap="viridis", xlabel=None, ylabel=None, fpath=None, show=False, fig_format="pdf", clean_props=None, hatching=False, remove_xticks=False, remove_legend=False, **kwargs):
    """General function for plotting boxplots (wrapper for :func:`seaborn.boxplot`).
    
    Args:
        df (:class:`pandas.DataFrame`): The DataFrame with results to plot from.
        x (str): The name of the column in the DataFrame to use as the x-axis.
        y (str): The name of the column in the DataFrame to use as the y-axis.
        cmap (str, optional): A colourmap for the plots. See `here <https://matplotlib.org/tutorials/colors/colormaps.html>`_ for options. Defaults to "viridis".
        xlabel (str, optional): The xlabel to add to the plot. Defaults to None.
        ylabel (str, optional): The ylabel to add to the plot. Defaults to None.
        fpath (:obj:`str`, :class:`pathlib.Path`, optional): The location to save the plot. Defaults to None.
        show (bool, optional): Whether to show the plot or not. Defaults to False.
        fig_format (str, optional): The file format to save the plot in. Defaults to "pdf".
        clean_props (dict, optional): The properties to use for cleaning the plot (calls :func:`~hawks.plotting.clean_graph`). Defaults to None.
        hatching (bool, optional): Whether to add hatches to the boxes for better visualization. Defaults to False.
        remove_xticks (bool, optional): Whether to remove the tick labels on the x-axis. Defaults to False.
        remove_legend (bool, optional): Whether to remove the legend. Defaults to False.
    """
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
        hatches = cycle(["/", "\\", ".", "x", "O", ""])
        # hatches = cycle(["/", "\\", ".", "x"])
        for i, box in enumerate(ax.patches):
            box.set_hatch(next(hatches)*3)
        for i, box in enumerate(ax.artists):
            box.set_hatch(next(hatches)*3)
    # Remove the legend
    if remove_legend:
        ax.legend_.remove()
    # May want to remove the xticks
    if remove_xticks:
        ax.set_xticklabels(
            ["" for _ in ax.get_xticklabels()]
        )
    # Format the graph
    if clean_props is not None:
        # Assume any input that isn't a dict signifies cleaning is desired
        if not isinstance(clean_props, dict):
            clean_props = {}
        # Add in the legend type if not passed (not used in sns.boxplot)
        if "legend_type" not in clean_props:
            clean_props["legend_type"] = "full"
        ax = clean_graph(ax, clean_props)
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
    # Close the figure (and its window)
    plt.close(fig)

def instance_space(df, color_highlight, marker_highlight=None, show=True, save_folder=None, seed=None, filename="instance_space", cmap="inferno", legend_type="brief", clean_props=None, plot_data=True, plot_components=False, save_pca=False, feat_label_placement=None, **kwargs):
    """Function to create the instance space. For information on usage, see :ref:`instance_space_example`.
    
    Args:
        df (:class:`pandas.DataFrame`): The DataFrame with results to plot from.
        color_highlight (str): The name of the column in the DataFrame to differentiate using colour.
        marker_highlight (str, optional): The name of the column in the DataFrame to differentiate using different markers. Defaults to None.
        show (bool, optional): Whether to show the plot or not. Defaults to True.
        save_folder (:obj:`str`, :class:`pathlib.Path`, optional): The folder to save the results in. Defaults to None.
        seed (int, optional): The random seed to pass to the clustering algorithms or problem features (if needed). Defaults to None.
        filename (str, optional): The filename to save the plot as. Defaults to "instance_space".
        cmap (str, optional): A colourmap for the plots. See `here <https://matplotlib.org/tutorials/colors/colormaps.html>`_ for options. Defaults to "inferno".
        legend_type (str, optional): The type of legend to use, governed by ``seaborn``. Defaults to "brief".
        clean_props (dict, optional): The properties to use for cleaning the plot (calls :func:`~hawks.plotting.clean_graph`). Defaults to None.
        plot_data (bool, optional): Whether the datasets should be plotted or not. Defaults to True.
        plot_components (bool, optional): Whether the components of the projection should be plotted or not. Defaults to False.
        save_pca (bool, optional): Whether to save the PCA object (for future use). Defaults to False.
        feat_label_placement (dict, optional): For customization of the location of the feature labels. Useful when ``plot_components = True``, and the labels need manual tweaking. Defaults to None.
    """
    # Make the fig and ax
    # Pop the figsize if provided, otherwise use default (None)
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", None))
    # Set the markers
    markers = cycle(["o", "D", "X", "v", "s", ">", "P", "*", "^"])
    if plot_components and plot_data:
        markers = cycle([""])
    # Set the colours by unique items in the highlight
    # Select the problem feature columns/values
    feature_cols = [col for col in df if col.startswith("f_")]
    # Dict of nicer names for plotting
    pretty_names = {
        "f_num_clusters": "Num\nClusters",
        "f_overlap": "Overlap",
        "f_silhouette": "Silhouette",
        "f_dimensionality": "Dimensions"
    }
    feature_names = [pretty_names[i] for i in feature_cols]
    # Extract the relevant array of values
    prob_feat_vals = df[feature_cols].values
    # PCA the problem features
    pca = PCA(n_components=2, random_state=seed)
    pca_feat_vals = pca.fit_transform(
        StandardScaler().fit_transform(prob_feat_vals)
    )
    # Add the PCs to the df
    df["PC1"] = pca_feat_vals[:, 0]
    df["PC2"] = pca_feat_vals[:, 1]
    # Save the PCA object to use later (if needed)
    if save_pca:
        with open(save_folder / "pca_obj.pkl", "wb") as pkl_file:
            pickle.dump(pca, pkl_file)
    # Remap input names for colour/marker
    remap_names = {
        "generator": "Source",
        "source": "Source",
        "algorithm": "Algorithm"
    }
    # Convert column names
    if color_highlight is not None:
        color_highlight = remap_names.get(color_highlight.lower(), color_highlight)
    if marker_highlight is not None:
        marker_highlight = remap_names.get(marker_highlight.lower(), marker_highlight)
    # Rename the source column (we don't always want this to be handled by the clean props)
    if "source" in df.columns and color_highlight == "Source" or marker_highlight == "Source":
        df = df.rename(columns={"source": "Source"})
    # Make the conversions needed for plotting clustering algorithms
    if color_highlight == "Algorithm" or marker_highlight == "Algorithm":
        # Select the cols for the cluster algs
        a = df[[col for col in df if col.startswith("c_")]]
        df["Algorithm"] = a.idxmax(1)
        # Remove the c_ prefix to algorithm names
        df['Algorithm'] = df['Algorithm'].map(lambda x: str(x)[2:])
        # Check if there are any ties
        s = a.eq(a.max(1), axis=0).sum(axis=1)
        # Replace the winner name with "Tied if so"
        df["Algorithm"] = df["Algorithm"].mask(s>1, "Tied")
        # Rename
        df = df.rename(columns={"source": "Source"})        
        # Then we need to take the best algorithm for each dataset
        # df = df.loc[df.groupby(["Source", "dataset_num"])["ARI"].idxmax()].reset_index(drop=True)
        # Adjust the cmap if needed, as it may have been specified without expectation of ties
        if isinstance(cmap, str) or len(cmap) != len(df["Algorithm"].unique()):
            if isinstance(cmap, str):
                cmap = sns.color_palette(cmap, len(df["Algorithm"].unique()))
            else:
                cmap = sns.color_palette("cubehelix", len(df["Algorithm"].unique()))
        # import pdb; pdb.set_trace()
    # Use seaborn's scatterplot for ease (otherwise groupby)
    if plot_data:
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
    # *TODO*: Split into a different function
    if plot_components:
        # Transpose components (multiply for better visualization)
        comps = pca.components_.T * 1.5
        # https://stackoverflow.com/a/50845697/9963224
        # This requires some manual adjustment...easier than an automated version (and another dependency!)
        if feat_label_placement is None:
            feat_label_placement = {
                "Dimensionality": (1, 1.2),
                "Overlap": (1.15, 1.15),
                "Silhouette": (1.15, 1.15)
            }
        for i in range(comps.shape[0]):
            # Add the feature
            ax.arrow(0, 0, comps[i, 0], comps[i, 1], color="red", width=0.01)
            xshift, yshift = feat_label_placement[feature_names[i]]
            # Add the feature label
            ax.text(
                comps[i, 0] * xshift, comps[i, 1] * yshift, feature_names[i],
                color='k', ha='center', va='center'
            )
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
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

def instance_parameters(gen, df, color_highlight=None, marker_highlight=None, save_folder=None, cmap="viridis", filename="instance_highlight", show=False, legend_type="brief", clean_props=None, **kwargs):
    # Get the stats dataframe
    stats_df = gen.stats[gen.stats["best_indiv"] == 1].reset_index(drop=True)
    # Only plot HAWKS data
    df = df[df["source"] == "HAWKS"]
    df["config_num"] = df["config_num"].astype(int)
    # Pop the figsize if provided, otherwise use default (None)
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", None))
    # Set the markers
    markers = cycle(["o", "X", "D", "v", "P", "s", "*", "^"])
    # Check PCA has been done
    if "PC1" not in df.columns:
        raise ValueError("Need a dataframe where the PCA has already been done")
    if color_highlight not in stats_df and color_highlight not in df:
        raise ValueError(f"{color_highlight} is not a parameter that was varied or measured")
    if marker_highlight not in stats_df and marker_highlight not in df:
        raise ValueError(f"{marker_highlight} is not a parameter that was varied or measured")
    # Insert columns if needed
    if color_highlight not in df:
        df[color_highlight] = stats_df[color_highlight]
    if marker_highlight not in df:
        df[marker_highlight] = stats_df[marker_highlight]
    # Create the plot
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

def convergence_plot(stats_df, y="fitness_silhouette", xlabel=None, ylabel=None, cmap="inferno", show=True, fpath=None, legend_type="brief", clean_props=None, ci=None, **kwargs):
    """Function to show the convergence of the e.g. fitness (though it can be anything). Generic wrapper function for :func:`seaborn.lineplot` (with error bars).

    Args:
        stats_df (:class:`pandas.DataFrame`): The DataFrame with results to plot from.
        y (str, optional): The column of the DataFrame for the y-axis. Defaults to "fitness_silhouette".
        xlabel (str, optional): The xlabel to add to the plot. Defaults to None.
        ylabel (str, optional): The ylabel to add to the plot. Defaults to None.
        cmap (str, optional): A colourmap for the plots. See `here <https://matplotlib.org/tutorials/colors/colormaps.html>`_ for options. Defaults to "viridis".
        show (bool, optional): Whether to show the plot or not. Defaults to False.
        fpath (:obj:`str`, :class:`pathlib.Path`, optional): The location to save the plot. Defaults to None.
        legend_type (str, optional): The type of legend to use, governed by ``seaborn``. Defaults to "brief".
        clean_props (dict, optional): The properties to use for cleaning the plot (calls :func:`~hawks.plotting.clean_graph`). Defaults to None.
        ci (:obj:`int`, :obj:`str`, optional): The confidence interval to use when calculating the estimated error. Can pass 'sd' instead to use the standard deviation. Defaults to None.
    """
    
    # Create the fig and axes
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", None))
    loc = kwargs.pop("loc", "best")
    # Filter certain columns
    filter_list = ["gen", y, "run"]
    # Add the column to also separate on colour if specified
    if "hue" in kwargs:
        filter_list.append(kwargs["hue"])
    if "style" in kwargs and kwargs["style"] != kwargs["hue"]:
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
        ci=ci,
        **kwargs
    )
    ax.set_xlim(
        (0, df["gen"].max()+1)
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
    """Generic wrapper function for :func:`seaborn.scatterplot`.

    Args:
        df (:class:`pandas.DataFrame`): The DataFrame with results to plot from.
        x (str): The name of the column in the DataFrame to use as the x-axis.
        y (str): The name of the column in the DataFrame to use as the y-axis.
        cmap (str, optional): A colourmap for the plots. See `here <https://matplotlib.org/tutorials/colors/colormaps.html>`_ for options. Defaults to "inferno".
        show (bool, optional): Whether to show the plot or not. Defaults to True.
        fpath (:obj:`str`, :class:`pathlib.Path`, optional): The location to save the plot. Defaults to None.
        clean_props (dict, optional): The properties to use for cleaning the plot (calls :func:`~hawks.plotting.clean_graph`). Defaults to None.
        legend_type (str, optional): The type of legend to use, governed by ``seaborn``. Defaults to "full".
    """
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

def scatter_prediction(data, preds, seed=None, ax=None, cmap="inferno", show=False, fpath=None, fig_format="png", **kwargs):
    """Function to plot the predictions (useful for working with external clustering algorithms).

    Args:
        data (:class:`numpy.ndarray`): The dataset to be plotted.
        preds (list, or :class:`numpy.ndarray`): The predicted labels for the given dataset.
        seed (int): The seed used to initialize the RNG. Defaults to None.
        ax (:mod:`matplotlib.axes`): The axis object to use (created if not).
        cmap (str, optional): A colourmap for the plots. See `here <https://matplotlib.org/tutorials/colors/colormaps.html>`_ for options. Defaults to "inferno".

    Returns:
        :mod:`matplotlib.axes`: The axis with the plot added.
    """
    # Create the figure and axis if called in isolation
    if ax is None:
        # Create the mpl objects
        fig, ax = plt.subplots(1, 1)
    # Perform PCA if needed
    if data.shape[1] > 2:
        # Perform PCA to get something we can plot
        pca = PCA(
            n_components=2,
            random_state=seed
        )
        # Transform the data with the PCA
        data = pca.fit_transform(data)
        col_names = ["PC1", "PC2"]
    else:
        col_names = ["1", "2"]
    # Create the DataFrame
    df = pd.DataFrame(data=data, columns=col_names)
    df["labels"] = preds
    # Create the plot
    ax = sns.scatterplot(
        data=df,
        x=col_names[0],
        y=col_names[1],
        hue="labels",
        style="labels",
        palette=cmap,
        ax=ax,
        legend=None,
        edgecolor="none", # Remove seaborn's white border
        **kwargs
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Save if a filepath is given
    if fpath is not None:
        save_plot(fig, fpath, fig_format=fig_format)
    # Show if specified
    if show:
        plt.show()
    return ax

def clean_graph(ax, clean_props=None):
    """Helper function to clean up graphs. Primarily designed for my own usage and preferences, but it should be more broadly useful. This function is called by many of the plots, though it can be called directly if working interactively.

    Args:
        ax (:mod:`matplotlib.axes`): The axis object to be cleaned.
        clean_props (dict, optional): A dictionary of options to use. Defaults to None, which uses the defaults specified in this function.

    Returns:
        :mod:`matplotlib.axes`: The cleaned axis object.
    """
    defaults = {
        "legend_type": "brief",
        "clean_legend": False,
        "legend_loc": "best",
        "legend_truncate": False,
        "truncate_amount": 4,
        "legend_wrap": True,
        "clean_labels": True,
        "wrap_ticks": False
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
                text.set_text(textwrap.fill(text.get_text(), 16))
    elif defaults["legend_type"] is None:
        try:
            ax.get_legend().remove()
        except AttributeError:
            pass
    # Title case the x and y labels
    if defaults["clean_labels"]:
        # Clean the y_axis label
        ax.set_ylabel(ax.get_ylabel().replace("_", " ").title())
        # Clean the x_axis label
        ax.set_xlabel(ax.get_xlabel().replace("_", " ").title())
    # Wrap the x ticks if need be
    if defaults["wrap_ticks"]:
        if ax.get_xticklabels()[0].get_text() != "":
            ax.set_xticklabels(
                [textwrap.fill(i.get_text(), 10) for i in ax.get_xticklabels()]
            )
    return ax

def cluster_alg_ranking(df, significance=0.05, show=False, save_folder=None, filename="alg-ranking"):
    """Produce critical difference (CD) diagrams (see `this paper <http://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf>`_ for further details).

    Args:
        df (:class:`pandas.DataFrame`): The DataFrame with results to plot from.
        significance (float, optional): The significance level to use for the statistical test. Defaults to 0.05.
        show (bool, optional): Whether to show the plot or not. Defaults to False.
        save_folder (:obj:`str`, :class:`pathlib.Path`, optional): The folder to save the plot in. Defaults to None.
        filename (str, optional): The filename to use for saving. Defaults to "alg-ranking".
    """
    # Replace the leading c_ for the algorithms
    df.columns = df.columns.str.replace("c_", "")
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
            Orange.evaluation.graph_ranks(avg_ranks, group.columns[1:], cd=cd, width=5.5)
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
