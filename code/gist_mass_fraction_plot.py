# -*- coding: utf-8 -*-

"""
@Time        : 9/1/2023
@Author      : purmortal
@File        : gist_mass_fraction_plot
@Description :
"""

from utils import cal_pearsonr_corr
import numpy as np
import matplotlib.pyplot as plt
import ppxf.ppxf_util as ppxf_util
from astropy.io import fits
from astropy.table import Table
from scipy.stats import pearsonr
from spectres import spectres
from mpl_toolkits.axes_grid1 import AxesGrid

from scipy.optimize import curve_fit
from scipy.special import hermite


# ----- GAUSS-HERMITE MODEL -----------------------------------------------------------------------
def gauss_hermite(x, A, mu, sigma, h3, h4):
    """
    Fourth-order Gauss-Hermite parameterization of an MDF.

    Parameters
    ----------
    x : array-like
        Metallicity [M/H].
    A : float
        Amplitude of the base Gaussian.
    mu : float
        Center of the base Gaussian.
    sigma : float
        Width of the base Gaussian.
    h3 : float
        Asymmetric Gauss-Hermite coefficient.
    h4 : float
        Symmetric peak/wing Gauss-Hermite coefficient.
    """
    x = np.asarray(x, dtype=float)
    w = (x - mu) / sigma

    H3 = (2.0 * w**3 - 3.0 * w) / np.sqrt(3.0)
    H4 = (4.0 * w**4 - 12.0 * w**2 + 3.0) / np.sqrt(24.0)

    return A * np.exp(-0.5 * w**2) * (1.0 + h3 * H3 + h4 * H4)

# ----------------------------------------------------------------------------------------------------------------
def plot_age_metal_grid_2d(distri1, distri2, title1, title2, age_grid_2d, metal_grid_2d, linear_age=False, threshold=0, **kwargs):
    '''
    plot two series values along the age/metallicity grid with different title
    :param distri1:
    :param distri2:
    :param title1:
    :param title2:
    :param age_bins:
    :param metal_bins:
    :param age_grid_2d:cg
    :param metal_grid_2d:
    :param linear_age:
    :param kwargs:
    :return:
    '''

    if linear_age:
        xgrid = age_grid_2d
        xlabel = "Age (Gyr)"
    else:
        xgrid = np.log10(age_grid_2d) + 9
        xlabel = "log Age (yr)"
    ygrid = metal_grid_2d

    distri1[distri1<threshold] = np.nan
    distri2[distri2<threshold] = np.nan

    fig = plt.figure(figsize=[12, 4])
    plt.subplot(121)
    ppxf_util.plot_weights_2d(xgrid, ygrid, distri1, nodots=False, colorbar=True, title=title1, xlabel=xlabel, **kwargs)
    plt.subplot(122)
    ppxf_util.plot_weights_2d(xgrid, ygrid, distri2, nodots=False, colorbar=True, title=title2, xlabel=xlabel, **kwargs)
    # plt.tight_layout()
    return fig

# ----------------------------------------------------------------------------------------------------------------
def plot_sfh_list(ax, mass_frac_list, age_grid_list, labels, colors, text=None, title=None, linear_age=True,
                  plot_xlabel=True, plot_ylabel=True, plot_legend=True, logmass=True, plot_ylim=None, perbinsize=True,
                  fraction_type='mass fraction', **kwargs):
    '''
    Plot the mass fraction in 1D, age/metallicity/alpha
    :param mass_alpha00_list:
    :param mass_alpha04_list:
    :param labels:
    :return:
    '''

    if type(mass_frac_list) != list:
        mass_frac_list = [mass_frac_list]
        age_grid_list = [age_grid_list]
        labels = [labels]
        colors = [colors]
        
    if linear_age:
        xlabel = "Age (Gyr)"
    else:
        for i, age_grid in enumerate(age_grid_list):
            age_grid_list[i] = np.log10(age_grid) + 9
        xlabel = "log Age (yr)"

    for mass_frac, age_grid, label, color in zip(mass_frac_list, age_grid_list, labels, colors):
        if logmass == True:
            ax.plot(age_grid, np.log10(mass_frac), label=label, c=color, alpha=0.75, **kwargs)
        else:
            ax.plot(age_grid, mass_frac, label=label, c=color, alpha=0.75, **kwargs)
    # if logmass == True:
    #     ax.set_ylim(-0.05 * np.nanmax(np.log10(mass_frac_list)), np.nanmax(np.log10(mass_frac_list)) * 1.25)
    if logmass != True and plot_ylim == None:
        if linear_age:
            max_mass = np.max([np.nanmax(mass_frac[age_grid > 1]) for mass_frac, age_grid in zip(mass_frac_list, age_grid_list)])
        else:
            max_mass = np.max([np.nanmax(mass_frac[10 ** (age_grid - 9) > 1]) for mass_frac, age_grid in zip(mass_frac_list, age_grid_list)])
        ax.set_ylim(-0.05 * max_mass, max_mass * 1.1)

    if plot_ylabel:
        if logmass == True:
            if perbinsize == True:
                ax.set_ylabel(r'log10 %s ($\rm Gyr^{-1}$)' % fraction_type)
            else:
                ax.set_ylabel(r'log10 %s' % fraction_type)
        else:
            if perbinsize == True:
                ax.set_ylabel(r'%s ($\rm Gyr^{-1}$)' % fraction_type)
            elif perbinsize == 'log':
                ax.set_ylabel(r'%s ($\rm log Gyr^{-1}$)' % fraction_type)
            else:
                ax.set_ylabel(r'%s' % fraction_type)
    if plot_xlabel:
        ax.set_xlabel(xlabel)
    # if linear_age: ax1.set_xticks(np.linspace(0, 12, 7))
    if plot_legend:
        ax.legend()
    if title != None:
        ax.set_title(title)
    if text != None:
        ax.text(0.05, 0.95, text, ha="left", va="top", rotation=0, size=12, transform=ax.transAxes)
    return ax

# ----------------------------------------------------------------------------------------------------------------
def plot_sfh_stairs_list(ax, mass_frac_list, age_grid_list, age_border_list, labels, colors, text=None, title=None, linear_age=True,
                  plot_xlabel=True, plot_ylabel=True, plot_legend=True, logmass=True, plot_ylim=None, perbinsize=True, fraction_type='mass fraction'):
    '''
    Plot the mass fraction in 1D, age/metallicity/alpha
    :param mass_alpha00_list:
    :param mass_alpha04_list:
    :param labels:
    :return:
    '''

    if type(mass_frac_list) != list:
        mass_frac_list = [mass_frac_list]
        age_border_list = [age_border_list]
        age_grid = [age_grid_list]
        labels = [labels]
        colors = [colors]
        
    if linear_age:
        xlabel = "Age (Gyr)"
    else:
        for i, (age_border, age_grid) in enumerate(zip(age_border_list, age_grid_list)):
            age_border_list[i] = np.log10(age_border) + 9
            age_grid_list[i] = np.log10(age_grid) + 9
        xlabel = "log Age (yr)"

    for mass_frac, age_border, label, color in zip(mass_frac_list, age_border_list, labels, colors):
        # age_coor = age_border[:-1]
        # age_width = age_border[1:] - age_border[:-1]
        if logmass == True:
            ax.stairs(np.log10(mass_frac), age_border, label=label, fill=False, baseline=None, color=color, alpha=0.75, lw=1.5)
        else:
            ax.stairs(mass_frac, age_border, label=label, fill=False, baseline=None, color=color, alpha=0.75, lw=1.5)
    # if logmass == True:
    #     ax.set_ylim(-0.05 * np.nanmax(np.log10(mass_frac_list)), np.nanmax(np.log10(mass_frac_list)) * 1.25)
    if logmass != True and plot_ylim == None:
        if linear_age:
            max_mass = np.max([np.nanmax(mass_frac[age_grid > 1]) for mass_frac, age_grid in zip(mass_frac_list, age_grid_list)])
        else:
            max_mass = np.max([np.nanmax(mass_frac[10 ** (age_grid - 9) > 1]) for mass_frac, age_grid in zip(mass_frac_list, age_grid_list)])
        ax.set_ylim(-0.05 * max_mass, max_mass * 1.25)
    if plot_ylabel:
        if logmass == True:
            if perbinsize == True:
                ax.set_ylabel(r'log10 %s ($\rm Gyr^{-1}$)' % fraction_type)
            else:
                ax.set_ylabel(r'log10 %s' % fraction_type)
        else:
            if perbinsize == True:
                ax.set_ylabel(r'%s ($\rm Gyr^{-1}$)' % fraction_type)
            elif perbinsize == 'log':
                ax.set_ylabel(r'%s ($\rm log Gyr^{-1}$)' % fraction_type)
            else:
                ax.set_ylabel(r'%s' % fraction_type)
    if plot_xlabel:
        ax.set_xlabel(xlabel)
    # if linear_age: ax1.set_xticks(np.linspace(0, 12, 7))
    if plot_legend:
        ax.legend()
    if title != None:
        ax.set_title(title)
    if text != None:
        ax.text(0.05, 0.95, text, ha="left", va="top", rotation=0, size=12, transform=ax.transAxes)
    # ax.xaxis.get_major_formatter().set_scientific(False)
    # ax.yaxis.get_major_formatter().set_scientific(False)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    return ax

# ----------------------------------------------------------------------------------------------------------------
def plot_metal_list(ax, mass_frac_list, metal_grid_list, labels, colors, text=None, title=None, 
                    plot_xlabel=True, plot_ylabel=True, plot_legend=True, logmass=True, perbinsize=True, fraction_type='mass fraction', **kwargs):
    '''
    Plot the mass fraction in 1D, age/metallicity/alpha
    :param mass_alpha00_list:
    :param mass_alpha04_list:
    :param labels:
    :return:
    '''

    if type(mass_frac_list) != list:
        mass_frac_list = [mass_frac_list]
        metal_grid_list = [metal_grid_list]
        labels = [labels]
        colors = [colors]

    for mass_frac, metal_grid, label, color in zip(mass_frac_list, metal_grid_list, labels, colors):
        if logmass == True:
            ax.plot(metal_grid, np.log10(mass_frac), label=label, c=color, alpha=0.75, **kwargs)
        else:
            ax.plot(metal_grid, mass_frac, label=label, c=color, alpha=0.75, **kwargs)
    if plot_ylabel:
        if logmass == True:
            if perbinsize == True:
                ax.set_ylabel(r'log10 %s ($\rm dex^{-1}$)' % fraction_type)
            else:
                ax.set_ylabel(r'log10 %s' % fraction_type)
        else:
            if perbinsize == True:
                ax.set_ylabel(r'%s ($\rm dex^{-1}$)' % fraction_type)
            else:
                ax.set_ylabel(r'%s' % fraction_type)
    if plot_xlabel:
        ax.set_xlabel("[M/H]")
    # if linear_age: ax1.set_xticks(np.linspace(0, 12, 7))
    if plot_legend:
        ax.legend()
    if title != None:
        ax.set_title(title)
    if text != None:
        ax.text(0.05, 0.95, text, ha="left", va="top", rotation=0, size=12, transform=ax.transAxes)
    return ax

# ----------------------------------------------------------------------------------------------------------------
def plot_metal_stairs_list(ax, mass_frac_list, metal_grid_list, metal_border_list, labels, colors, text=None, title=None, 
                    plot_xlabel=True, plot_ylabel=True, plot_legend=True, logmass=True, perbinsize=True, fraction_type='mass fraction'):
    '''
    Plot the mass fraction in 1D, age/metallicity/alpha
    :param mass_alpha00_list:
    :param mass_alpha04_list:
    :param labels:
    :return:
    '''

    if type(mass_frac_list) != list:
        mass_frac_list = [mass_frac_list]
        metal_grid_list = [metal_grid_list]
        metal_border_list = [metal_border_list]
        labels = [labels]
        colors = [colors]

    for mass_frac, metal_border, label, color in zip(mass_frac_list, metal_border_list, labels, colors):
        if logmass == True:
            ax.stairs(np.log10(mass_frac), metal_border, fill=False, baseline=None, label=label, color=color, alpha=0.75, lw=1.5)
        else:
            ax.stairs(mass_frac, metal_border, fill=False, baseline=None, label=label, color=color, alpha=0.75, lw=1.5)
    if plot_ylabel:
        if logmass == True:
            if perbinsize == True:
                ax.set_ylabel(r'log10 %s ($\rm dex^{-1}$)' % fraction_type)
            else:
                ax.set_ylabel(r'log10 %s' % fraction_type)
        else:
            if perbinsize == True:
                ax.set_ylabel(r'%s ($\rm dex^{-1}$)' % fraction_type)
            else:
                ax.set_ylabel(r'%s' % fraction_type)
    if plot_xlabel:
        ax.set_xlabel("[M/H]")
    # if linear_age: ax1.set_xticks(np.linspace(0, 12, 7))
    if plot_legend:
        ax.legend()
    if title != None:
        ax.set_title(title)
    if text != None:
        ax.text(0.05, 0.95, text, ha="left", va="top", rotation=0, size=12, transform=ax.transAxes)
    return ax

# ----------------------------------------------------------------------------------------------------------------
def plot_cumulative_sfh_list(ax, cumulative_mass_list, age_grid_list, age_border_list, labels, colors, text=None, title=None, linear_age=True,
                  plot_xlabel=True, plot_ylabel=True, plot_legend=True, plot_ylim=None):
    '''
    Plot the mass fraction in 1D, age/metallicity/alpha
    :param mass_alpha00_list:
    :param mass_alpha04_list:
    :param labels:
    :return:
    '''

    if type(cumulative_mass_list) != list:
        cumulative_mass_list = [cumulative_mass_list]
        age_grid_list = [age_grid_list]
        age_border_list = [age_border_list]
        labels = [labels]
        colors = [colors]
        
    if linear_age:
        xlabel = "Age (Gyr)"
    else:
        for i, (age_border, age_grid) in enumerate(zip(age_border_list, age_grid_list)):
            age_border_list[i] = np.log10(age_border) + 9
            age_grid_list[i] = np.log10(age_grid) + 9
        xlabel = "log Age (yr)"

    for cumulative_mass, age_border, label, color in zip(cumulative_mass_list, age_border_list, labels, colors):
        # ax.step(age_grid, np.cumsum(cumulative_mass[::-1])[::-1], '-', label=label, c=color, alpha=0.75, lw=0.8)
        ax.stairs(np.cumsum(cumulative_mass[::-1])[::-1], age_border, fill=False, baseline=None, label=label, color=color, alpha=0.75, lw=0.8)
    ax.set_ylim(-0.05, 1.05)
    if plot_ylabel:
        ax.set_ylabel(r'Cum. weights')
    if plot_xlabel:
        ax.set_xlabel(xlabel)
    if plot_legend:
        ax.legend()
    if title != None:
        ax.set_title(title)
    if text != None:
        ax.text(0.05, 0.95, text, ha="left", va="top", rotation=0, size=12, transform=ax.transAxes)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    return ax

# ----------------------------------------------------------------------------------------------------------------
def plot_cumulative_metal_list(ax, cumulative_mass_list, metal_grid_list, metal_border_list, labels, colors, text=None, title=None,
                    plot_xlabel=True, plot_ylabel=True, plot_legend=True, fraction_type='mass fraction'):
    '''
    Plot the mass fraction in 1D, age/metallicity/alpha
    :param mass_alpha00_list:
    :param mass_alpha04_list:
    :param labels:
    :return:
    '''

    if type(cumulative_mass_list) != list:
        cumulative_mass_list = [cumulative_mass_list]
        metal_grid_list = [metal_grid_list]
        metal_border_list = [metal_border_list]
        labels = [labels]
        colors = [colors]

    for cumulative_mass, metal_border, label, color in zip(cumulative_mass_list, metal_border_list, labels, colors):
        # ax.step(metal_grid, np.cumsum(cumulative_mass[::-1])[::-1], '-', label=label, c=color, alpha=0.75, lw=0.8)
        ax.stairs(np.cumsum(cumulative_mass[::-1])[::-1], metal_border, fill=False, baseline=None, label=label, color=color, alpha=0.75, lw=0.8)
    if plot_ylabel:
        ax.set_ylabel(r'Cum. weights')
    if plot_xlabel:
        ax.set_xlabel("[M/H]")
    if plot_legend:
        ax.legend()
    if title != None:
        ax.set_title(title)
    if text != None:
        ax.text(0.05, 0.95, text, ha="left", va="top", rotation=0, size=12, transform=ax.transAxes)
    return ax

# ----------------------------------------------------------------------------------------------------------------
def plot_weights_2d(ax, xgrid, ygrid, weights, threshold, title, plot_xlabel,
                    plot_ylabel, xlabel="log Age (yr)", ylabel="[M/H]",
                    nodots=False, colorbar=True, **kwargs):
    weights[weights < threshold] = np.nan

    x = xgrid[:, 0]  # Grid centers
    y = ygrid[0, :]
    xb = (x[1:] + x[:-1]) / 2  # internal grid borders
    yb = (y[1:] + y[:-1]) / 2
    xb = np.hstack([1.5 * x[0] - x[1] / 2, xb, 1.5 * x[-1] - x[-2] / 2])  # 1st/last border
    yb = np.hstack([1.5 * y[0] - y[1] / 2, yb, 1.5 * y[-1] - y[-2] / 2])

    pc = ax.pcolormesh(xb, yb, weights.T, edgecolors='face', **kwargs)
    if plot_xlabel:
        ax.set_xlabel(xlabel)
    if plot_ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if not nodots:
        ax.plot(xgrid, ygrid, 'w,')
    if colorbar:
        plt.colorbar(pc, ax=ax)
        # ax.sca(ax)  # Activate main plot before returning

    return pc

# ----------------------------------------------------------------------------------------------------------------
def cal_rz_distri_bin(rbin, zbin, d, results_table, weights_values, reg_dim):
    r_range_list = []
    z_range_list = []
    distri_bin_list = []
    i_list = []
    j_list = []

    for i in range(len(rbin) - 1):
        for j in range(len(zbin) - 1):

            r_range = rbin[[i, i + 1]]
            z_range = zbin[[j, j + 1]]
            binid = np.unique(results_table[(results_table['XBIN'] > r_range[0] / d * 180 / np.pi * 3600) &
                                            (results_table['XBIN'] < r_range[1] / d * 180 / np.pi * 3600) &
                                            (np.abs(results_table['YBIN']) > z_range[0] / d * 180 / np.pi * 3600) &
                                            (np.abs(results_table['YBIN']) < z_range[1] / d * 180 / np.pi * 3600) &
                                            (results_table['BIN_ID'] >= 0)]['BIN_ID'])

            mass_frac_bin = weights_values[binid, :]
            distri_bin = np.sum(np.sum(mass_frac_bin, axis=0).reshape(reg_dim), axis=0)
            r_range_list.append(r_range)
            z_range_list.append(z_range)
            distri_bin_list.append(distri_bin)
            i_list.append(i)
            j_list.append(j)
    return r_range_list, z_range_list, distri_bin_list, i_list, j_list

#-----------------------------------------------------------------------------------------------------------
def plot_mh_alpha_rz_hist(weights_values, weights_true_values, rbin, zbin, d,
                          reg_dim, reg_dim_true, results_table,
                          metal_grid, metal_grid_true,
                          fraction_type='mass fraction',
                          figure_title='100% Migration Efficiency MDFs'):

    """
    Plot pPXF-recovered and true/input MDFs in projected R-|z| bins.

    For each spatial bin:
        - dark-blue solid line: pPXF-recovered MDF
        - dark-blue dashed line: Gauss-Hermite fit to pPXF MDF
        - orange solid line: true/input MDF
        - orange dashed line: Gauss-Hermite fit to true/input MDF

    Only the [alpha/Fe] = 0.0 slice (index 0) is displayed.
    Each MDF is independently normalized so that its metallicity-bin
    mass fractions sum to one within the spatial bin.
    """

    # ---------------- COLORS ----------------
    true_color = '#1f5a94' # Blue 
    ppxf_color = '#e67e22' # Orange
    constrained_color = '#7b2cbf' # Purple 

    # ---------------- TUNABLE TEXT SIZES ----------------
    panel_fs = 12
    moment_fs = 7.5
    moment_dy = 0.055
    title_y_z = 0.975
    title_y_r = 0.895

    fit_results = []

    # ---------------- GRID SETUP ----------------
    nz = len(zbin) - 1
    nr = len(rbin) - 1

    fig, axes = plt.subplots(
        nz,
        nr,
        figsize=(3.1 * nr, 2.9 * nz),
        sharey=True,
        squeeze=False
    )

    # ---------------- BIN DISTRIBUTIONS ----------------
    r_range_list, z_range_list, distri_bin_list, i_list, j_list = cal_rz_distri_bin(
        rbin,
        zbin,
        d,
        results_table,
        weights_values,
        reg_dim
    )

    _, _, distri_bin_true_list, _, _ = cal_rz_distri_bin(
        rbin,
        zbin,
        d,
        results_table,
        weights_true_values,
        reg_dim_true
    )

    # ---------------- MAP EACH SPATIAL BIN TO ITS INDEX ----------------
    bin_map = {}

    for k in range(len(distri_bin_list)):
        r0, r1 = r_range_list[k]
        z0, z1 = z_range_list[k]
        bin_map[(z0, z1, r0, r1)] = k

    # ---------------- GAUSS-HERMITE FITTING ----------------
    def _gh_fit_and_plot(ax, xgrid, ydata, color, label, do_label):
        """
        Fit the same Gauss-Hermite model independently to one MDF.

        Important fitting choices:
            - no smoothing;
            - no peak-based threshold;
            - all finite, non-negative MDF bins are retained;
            - the fitted curve is not renormalized after optimization;
            - starting mu and sigma adapt automatically to each MDF;
            - multiple h3/h4 starting points are tested;
            - the same bounds and fitting method are used in every bin.
        """

        xgrid = np.asarray(xgrid, dtype=float)
        data = np.asarray(ydata, dtype=float)

        # Keep only valid metallicity bins.
        valid = (
            np.isfinite(xgrid)
            & np.isfinite(data)
            & (data >= 0.0)
        )

        x = xgrid[valid]
        y = data[valid]

        if len(x) < 6:
            print(
                "GH fit skipped: fewer than six valid metallicity bins."
            )
            return None

        total = np.sum(y)

        if not np.isfinite(total) or total <= 0.0:
            print(
                "GH fit skipped: MDF has a non-positive or invalid total."
            )
            return None

        # Normalize the actual MDF used in the fit.
        y = y / total

        # Automatically adapt the initial center and width to this MDF.
        mu0 = np.sum(x * y)

        variance0 = np.sum(y * (x - mu0)**2)
        sigma0 = np.sqrt(max(variance0, 0.05**2))

        sigma0 = np.clip(
            sigma0,
            0.06,
            0.55
        )

        A0 = np.max(y)

        # Use bounds broad enough for all spatial bins while applying
        # the same constraints everywhere.
        lower_bounds = [
            0.0,
            np.min(x) - 0.25,
            0.04,
            -0.30,
            -0.30
        ]

        upper_bounds = [
            max(2.0, 3.0 * A0),
            np.max(x) + 0.25,
            0.75,
            0.30,
            0.30
        ]

        # Multiple starting points reduce sensitivity to a single
        # local minimum without manually tuning individual spatial bins.
        h_start_values = [-0.10, 0.0, 0.10]

        best_popt = None
        best_score = np.inf

        for h3_start in h_start_values:
            for h4_start in h_start_values:

                p0 = [
                    A0,
                    mu0,
                    sigma0,
                    h3_start,
                    h4_start
                ]

                try:
                    popt, _ = curve_fit(
                        gauss_hermite,
                        x,
                        y,
                        p0=p0,
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=100000
                    )

                    model_at_data = gauss_hermite(x, *popt)

                    if not np.all(np.isfinite(model_at_data)):
                        continue

                    residual_score = np.sum(
                        (y - model_at_data)**2
                    )

                    # Penalize substantially negative GH curves because
                    # an MDF represents a non-negative mass fraction.
                    dense_test_grid = np.linspace(
                        np.min(x),
                        np.max(x),
                        500
                    )

                    dense_test_model = gauss_hermite(
                        dense_test_grid,
                        *popt
                    )

                    negative_values = np.minimum(
                        dense_test_model,
                        0.0
                    )

                    negativity_penalty = 100.0 * np.sum(
                        negative_values**2
                    )

                    score = residual_score + negativity_penalty

                    if score < best_score:
                        best_score = score
                        best_popt = popt

                except (RuntimeError, ValueError, FloatingPointError):
                    continue

        if best_popt is None:
            print(
                "GH fit failed for one MDF after all starting points."
            )
            return None

        A, mu, sigma, h3, h4 = best_popt

        # Draw a smooth fitted curve while fitting only the real MDF bins.
        x_dense = np.linspace(
            np.min(xgrid),
            np.max(xgrid),
            500
        )

        fit_y_dense = gauss_hermite(
            x_dense,
            A,
            mu,
            sigma,
            h3,
            h4
        )

        ax.plot(
            x_dense,
            fit_y_dense,
            linestyle='--',
            linewidth=2.2,
            color=color,
            alpha=0.95,
            label=label if do_label else None,
            zorder=4
        )

        model_at_data = gauss_hermite(
            x,
            A,
            mu,
            sigma,
            h3,
            h4
        )

        rmse = np.sqrt(
            np.mean((y - model_at_data)**2)
        )

        return {
            'A': float(A),
            'mu': float(mu),
            'sigma': float(sigma),
            'h3': float(h3),
            'h4': float(h4),
            'rmse': float(rmse)
        }

    def _true_constrained_gh_fit_and_plot(ax, xgrid, ydata, color,
                                          label, do_label):
        """
        Experimental stability-constrained GH fit for the true MDF.

        This fit:
            - retains all finite, non-negative true-MDF bins;
            - uses the direct MDF mean and width as starting values;
            - requires sigma >= 0.12 dex;
            - restricts |h3| and |h4| to 0.15;
            - penalizes negative model values;
            - penalizes artificial between-bin peaks;
            - does not alter the original true-MDF GH fit.
        """

        from scipy.optimize import least_squares

        xgrid = np.asarray(xgrid, dtype=float)
        data = np.asarray(ydata, dtype=float)

        valid = (
            np.isfinite(xgrid)
            & np.isfinite(data)
            & (data >= 0.0)
        )

        x = xgrid[valid]
        y = data[valid]

        if len(x) < 6:
            print(
                "Constrained true GH fit skipped: "
                "fewer than six valid metallicity bins."
            )
            return None

        total = np.sum(y)

        if not np.isfinite(total) or total <= 0.0:
            print(
                "Constrained true GH fit skipped: "
                "invalid MDF normalization."
            )
            return None

        y = y / total

        # Direct MDF estimates provide bin-specific starting values.
        mu0 = np.sum(x * y)

        variance0 = np.sum(
            y * (x - mu0)**2
        )

        sigma0 = np.sqrt(
            max(variance0, 0.12**2)
        )

        sigma0 = np.clip(
            sigma0,
            0.12,
            0.55
        )

        A0 = np.max(y)

        sigma_min = 0.12
        h_limit = 0.15
        maximum_allowed_peak = 1.35 * np.max(y)

        lower_bounds = np.array([
            0.0,
            np.min(x) - 0.25,
            sigma_min,
            -h_limit,
            -h_limit
        ])

        upper_bounds = np.array([
            1.35 * A0,
            np.max(x) + 0.25,
            0.75,
            h_limit,
            h_limit
        ])

        dense_test_grid = np.linspace(
            np.min(x),
            np.max(x),
            500
        )

        def constrained_residuals(parameters):
            model_at_data = gauss_hermite(
                x,
                *parameters
            )

            model_dense = gauss_hermite(
                dense_test_grid,
                *parameters
            )

            data_residuals = (
                model_at_data - y
            )

            # Penalize non-physical negative MDF values.
            negative_penalty = np.sqrt(100.0) * np.minimum(
                model_dense,
                0.0
            )

            # Penalize artificial peaks between metallicity bins.
            peak_penalty = np.sqrt(100.0) * np.maximum(
                model_dense - maximum_allowed_peak,
                0.0
            )

            return np.concatenate([
                data_residuals,
                negative_penalty,
                peak_penalty
            ])

        h_start_values = [
            -0.075,
            0.0,
            0.075
        ]

        best_result = None
        best_score = np.inf

        for h3_start in h_start_values:
            for h4_start in h_start_values:

                initial_parameters = np.array([
                    min(A0, 0.95 * upper_bounds[0]),
                    mu0,
                    sigma0,
                    h3_start,
                    h4_start
                ])

                try:
                    result = least_squares(
                        constrained_residuals,
                        initial_parameters,
                        bounds=(lower_bounds, upper_bounds),
                        max_nfev=100000,
                        x_scale='jac'
                    )

                    if not result.success:
                        continue

                    score = np.sum(
                        constrained_residuals(result.x)**2
                    )

                    if score < best_score:
                        best_score = score
                        best_result = result

                except (RuntimeError, ValueError, FloatingPointError):
                    continue

        if best_result is None:
            print(
                "Constrained true GH fit failed after "
                "all starting points."
            )
            return None

        A, mu, sigma, h3, h4 = best_result.x

        x_dense = np.linspace(
            np.min(xgrid),
            np.max(xgrid),
            500
        )

        fit_y_dense = gauss_hermite(
            x_dense,
            A,
            mu,
            sigma,
            h3,
            h4
        )

        ax.plot(
            x_dense,
            fit_y_dense,
            color=color,
            linestyle=':',
            linewidth=2.4,
            alpha=0.95,
            label=label if do_label else None,
            zorder=5
        )

        model_at_data = gauss_hermite(
            x,
            A,
            mu,
            sigma,
            h3,
            h4
        )

        rmse = np.sqrt(
            np.mean((y - model_at_data)**2)
        )

        return {
            'A': float(A),
            'mu': float(mu),
            'sigma': float(sigma),
            'h3': float(h3),
            'h4': float(h4),
            'rmse': float(rmse),
            'fitting_strategy': 'true constrained diagnostic'
        }

    # ---------------- LOOP OVER SPATIAL BINS ----------------
    for j in range(nz):
        z0 = zbin[j]
        z1 = zbin[j + 1]

        for col in range(nr):
            r0 = rbin[col]
            r1 = rbin[col + 1]

            ax = axes[-j - 1, col]
            ax.tick_params(direction='in')

            # Hide x tick labels except along the bottom row.
            if j != 0:
                ax.tick_params(labelbottom=False)

            key = (z0, z1, r0, r1)

            if key not in bin_map:
                ax.set_axis_off()
                continue

            k = bin_map[key]

            # ---------------- PANEL LABELS ----------------
            ax.text(
                0.5,
                title_y_z,
                rf'$\mathbf{{{z0}<|z|/\mathrm{{kpc}}<{z1}}}$',
                ha='center',
                va='top',
                transform=ax.transAxes,
                fontsize=panel_fs,
                fontweight='bold'
            )

            ax.text(
                0.5,
                title_y_r,
                rf'$\mathbf{{{r0}<R_{{\mathrm{{proj}}}}/\mathrm{{kpc}}<{r1}}}$',
                ha='center',
                va='top',
                transform=ax.transAxes,
                fontsize=panel_fs,
                fontweight='bold'
            )

            # ---------------- MDFs FOR [alpha/Fe] = 0.0 ----------------
            ppxf_mdf = np.asarray(
                distri_bin_list[k][:, 0],
                dtype=float
            ).copy()

            true_mdf = np.asarray(
                distri_bin_true_list[k][:, 0],
                dtype=float
            ).copy()

            ppxf_total = np.nansum(ppxf_mdf)
            true_total = np.nansum(true_mdf)

            if (
                not np.isfinite(ppxf_total)
                or not np.isfinite(true_total)
                or ppxf_total <= 0.0
                or true_total <= 0.0
            ):
                ax.set_axis_off()
                continue

            ppxf_mdf /= ppxf_total
            true_mdf /= true_total

            # Actual MDFs are solid; color identifies their origin.
            ax.plot(
                metal_grid,
                ppxf_mdf,
                color=ppxf_color,
                linewidth=2.0,
                linestyle='-',
                label=(
                    r'pPXF MDF ([$\alpha$/Fe]=0.0)'
                    if (j == 0 and col == 0)
                    else None
                ),
                zorder=3
            )

            ax.plot(
                metal_grid_true,
                true_mdf,
                color=true_color,
                linewidth=2.0,
                linestyle='-',
                label=(
                    r'True MDF ([$\alpha$/Fe]=0.0)'
                    if (j == 0 and col == 0)
                    else None
                ),
                zorder=2
            )

            ax.set_xlim(-1.0, 0.5)
            ax.set_ylim(0.0, 0.7)

            # ---------------- FIT BOTH MDFs ----------------
            do_leg = (j == 0 and col == 0)

            ppxf_fit = _gh_fit_and_plot(
                ax,
                metal_grid,
                ppxf_mdf,
                ppxf_color,
                r'pPXF MDF GH fit',
                do_leg
            )

            true_fit = _gh_fit_and_plot(
                ax,
                metal_grid_true,
                true_mdf,
                true_color,
                r'True MDF GH fit',
                do_leg
            )


            true_constrained_fit = _true_constrained_gh_fit_and_plot(
                ax,
                metal_grid_true,
                true_mdf,
                constrained_color,
                r'True MDF constrained GH fit',
                do_leg
            )


            # Store both fits so the notebook can use them later.
            fit_results.append({
                'zbin': (float(z0), float(z1)),
                'rbin': (float(r0), float(r1)),
                'ppxf': ppxf_fit,
                'true': true_fit,
                'true_constrained': true_constrained_fit
            })


            # ---------------- GH PARAMETER ANNOTATIONS ----------------
            # These are explicitly labeled as GH fit parameters.
            txt_y0 = 0.765

            if true_fit is not None:
                true_lines = [
                    r'$\mathbf{True\ GH}$',
                    rf'$\mu={true_fit["mu"]:+.3f}$',
                    rf'$\sigma={true_fit["sigma"]:.3f}$',
                    rf'$h_3={true_fit["h3"]:+.3f}$',
                    rf'$h_4={true_fit["h4"]:+.3f}$'
                ]

                for line_index, text_line in enumerate(true_lines):
                    ax.text(
                        0.03,
                        txt_y0 - line_index * moment_dy,
                        text_line,
                        transform=ax.transAxes,
                        ha='left',
                        va='top',
                        fontsize=moment_fs,
                        fontweight='bold',
                        color=true_color,
                        bbox={
                            'facecolor': 'white',
                            'edgecolor': 'none',
                            'alpha': 0.72,
                            'pad': 0.4
                        }
                    )

            if ppxf_fit is not None:
                ppxf_lines = [
                    r'$\mathbf{pPXF\ GH}$',
                    rf'$\mu={ppxf_fit["mu"]:+.3f}$',
                    rf'$\sigma={ppxf_fit["sigma"]:.3f}$',
                    rf'$h_3={ppxf_fit["h3"]:+.3f}$',
                    rf'$h_4={ppxf_fit["h4"]:+.3f}$'
                ]

                for line_index, text_line in enumerate(ppxf_lines):
                    ax.text(
                        0.97,
                        txt_y0 - line_index * moment_dy,
                        text_line,
                        transform=ax.transAxes,
                        ha='right',
                        va='top',
                        fontsize=moment_fs,
                        fontweight='bold',
                        color=ppxf_color,
                        bbox={
                            'facecolor': 'white',
                            'edgecolor': 'none',
                            'alpha': 0.72,
                            'pad': 0.4
                        }
                    )

    # ---------------- GLOBAL FORMATTING ----------------
    fig.add_subplot(111, frame_on=False)

    plt.tick_params(
        labelcolor='none',
        bottom=False,
        left=False
    )

    plt.xlabel(
        '[M/H]',
        fontsize=17,
        fontweight='bold',
        labelpad=22
    )

    plt.ylabel(
        fraction_type,
        fontsize=17,
        fontweight='bold',
        labelpad=22
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.88,
        bottom=0.12,
        hspace=0.14,
        wspace=0.18
    )

    for ax in axes.flatten():
        if not ax.axison:
            continue

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=13,
            width=1.6,
            length=6
        )

        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')

        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')

        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

    # ---------------- CLEAR FOUR-ITEM LEGEND ----------------
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=true_color,
            linewidth=2.0,
            linestyle='-',
            label=r'True MDF ([$\alpha$/Fe]=0.0)'
        ),
        Line2D(
            [0],
            [0],
            color=true_color,
            linewidth=2.2,
            linestyle='--',
            label=r'True MDF GH fit'
        ),
        Line2D(
            [0],
            [0],
            color=constrained_color,
            linewidth=2.4,
            linestyle=':',
            label=r'True MDF constrained GH fit (test)'
        ),
        Line2D(
            [0],
            [0],
            color=ppxf_color,
            linewidth=2.0,
            linestyle='-',
            label=r'pPXF MDF ([$\alpha$/Fe]=0.0)'
        ),
        Line2D(
            [0],
            [0],
            color=ppxf_color,
            linewidth=2.2,
            linestyle='--',
            label=r'pPXF MDF GH fit'
        )
    ]

    fig.suptitle(
        figure_title,
        fontsize=18,
        fontweight='bold',
        y=0.99
    )

    legend = fig.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=5,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, 0.955)
    )

    for legend_text in legend.get_texts():
        legend_text.set_fontweight('bold')

    return fig, fit_results




# =================================================================================================
# FACE-ON MDF ROUTINES
# =================================================================================================

def cal_r_distri_bin_faceon(
        rbin,
        d,
        results_table,
        weights_values,
        reg_dim,
        x_center_arcsec=0.0,
        y_center_arcsec=0.0
):
    """
    Combine stellar-population weights in face-on galactocentric radial bins.

    Unlike the edge-on routine, this function does not use vertical |z| bins.
    XBIN and YBIN are treated as the two Cartesian coordinates in the
    face-on image plane. The true in-plane galactocentric radius is

        R = sqrt((XBIN - X0)^2 + (YBIN - Y0)^2).

    Parameters
    ----------
    rbin : array-like
        Radial-bin boundaries in kpc.

    d : float
        Distance to the mock galaxy in kpc.

    results_table : astropy.table.Table
        GIST table containing XBIN, YBIN, and BIN_ID.

    weights_values : ndarray
        Population weights with shape
        (number of spatial bins, number of SSP templates).

    reg_dim : array-like
        SSP-grid dimensions:
        [number of ages, number of metallicities, number of alpha values].

    x_center_arcsec, y_center_arcsec : float
        Galaxy-center coordinates in arcseconds. For the current face-on
        mock cube, the diagnostics support using (0, 0).

    Returns
    -------
    r_range_list : list
        Radial intervals in kpc.

    distri_bin_list : list
        Metallicity-alpha distributions for every radial interval.

    binid_list : list
        Unique GIST BIN_ID values included in every interval.

    coverage_list : list
        Number of unique GIST bins included in every interval.
    """

    rbin = np.asarray(rbin, dtype=float)

    x_arcsec = np.asarray(
        results_table['XBIN'],
        dtype=float
    ) - float(x_center_arcsec)

    y_arcsec = np.asarray(
        results_table['YBIN'],
        dtype=float
    ) - float(y_center_arcsec)

    bin_id_column = np.asarray(
        results_table['BIN_ID'],
        dtype=int
    )

    # Face-on angular galactocentric radius.
    radius_arcsec = np.sqrt(
        x_arcsec**2 + y_arcsec**2
    )

    # Convert angular radius from arcseconds to physical radius in kpc.
    radius_kpc = (
        radius_arcsec
        * np.pi
        / 180.0
        / 3600.0
        * float(d)
    )

    r_range_list = []
    distri_bin_list = []
    binid_list = []
    coverage_list = []

    for i in range(len(rbin) - 1):

        r0 = float(rbin[i])
        r1 = float(rbin[i + 1])

        radial_mask = (
            (radius_kpc >= r0)
            & (radius_kpc < r1)
            & (bin_id_column >= 0)
        )

        bin_ids = np.unique(
            bin_id_column[radial_mask]
        ).astype(int)

        # Protect against malformed or out-of-range BIN_ID values.
        bin_ids = bin_ids[
            (bin_ids >= 0)
            & (bin_ids < weights_values.shape[0])
        ]

        r_range_list.append(
            np.array([r0, r1], dtype=float)
        )

        binid_list.append(bin_ids)
        coverage_list.append(len(bin_ids))

        if len(bin_ids) == 0:
            distri_bin_list.append(None)
            continue

        # Select all GIST spatial bins in this radial annulus.
        selected_weights = np.asarray(
            weights_values[bin_ids, :],
            dtype=float
        )

        # Add the population weights from every selected spatial bin.
        summed_weights = np.nansum(
            selected_weights,
            axis=0
        )

        if (
            not np.all(np.isfinite(summed_weights))
            or np.nansum(summed_weights) <= 0.0
        ):
            distri_bin_list.append(None)
            continue

        # Restore the age-metallicity-alpha population grid.
        population_grid = summed_weights.reshape(
            tuple(np.asarray(reg_dim, dtype=int))
        )

        # Sum over stellar age. The result has shape:
        # metallicity x alpha.
        metal_alpha_distribution = np.nansum(
            population_grid,
            axis=0
        )

        distri_bin_list.append(
            metal_alpha_distribution
        )

    return (
        r_range_list,
        distri_bin_list,
        binid_list,
        coverage_list
    )


def plot_mh_alpha_r_hist_faceon(
        weights_values,
        weights_true_values,
        rbin,
        d,
        reg_dim,
        reg_dim_true,
        results_table,
        metal_grid,
        metal_grid_true,
        fraction_type='mass fraction',
        figure_title='Migration Efficiency Face-on MDFs',        # FACEON MDF TITLE FIGURE IS HERE!!!!
        alpha_index=0,
        x_center_arcsec=0.0,
        y_center_arcsec=0.0,
        include_true_constrained_fit=False
):
    """
    Plot face-on true/input and pPXF-recovered MDFs in true radial bins.

    Face-on geometry
    ----------------
    Each panel contains every observed GIST spatial bin satisfying

        r_min <= sqrt(XBIN^2 + YBIN^2) < r_max.

    There is no vertical |z| selection.

    Curves
    ------
    Blue solid:
        True/input MDF.

    Blue dashed:
        Gauss-Hermite fit to the true/input MDF.

    Orange solid:
        pPXF-recovered MDF.

    Orange dashed:
        Gauss-Hermite fit to the pPXF MDF.

    Purple dotted, optional:
        Stability-constrained diagnostic fit to the true MDF.
    """

    # ---------------------------------------------------------------------------------------------
    # COLORS AND TEXT
    # ---------------------------------------------------------------------------------------------

    true_color = '#1f5a94' # Blue 
    ppxf_color = '#e67e22' # Orange 
    constrained_color = '#7b2cbf' # Purple 

    panel_fs = 11
    moment_fs = 7.2
    moment_dy = 0.055

    rbin = np.asarray(rbin, dtype=float)
    metal_grid = np.asarray(metal_grid, dtype=float)
    metal_grid_true = np.asarray(metal_grid_true, dtype=float)

    fit_results = []

    # ---------------------------------------------------------------------------------------------
    # CREATE FACE-ON RADIAL DISTRIBUTIONS
    # ---------------------------------------------------------------------------------------------

    (
        r_range_list,
        distri_bin_list,
        binid_list,
        coverage_list
    ) = cal_r_distri_bin_faceon(
        rbin=rbin,
        d=d,
        results_table=results_table,
        weights_values=weights_values,
        reg_dim=reg_dim,
        x_center_arcsec=x_center_arcsec,
        y_center_arcsec=y_center_arcsec
    )

    (
        r_range_true_list,
        distri_bin_true_list,
        binid_true_list,
        coverage_true_list
    ) = cal_r_distri_bin_faceon(
        rbin=rbin,
        d=d,
        results_table=results_table,
        weights_values=weights_true_values,
        reg_dim=reg_dim_true,
        x_center_arcsec=x_center_arcsec,
        y_center_arcsec=y_center_arcsec
    )

    nr = len(rbin) - 1

    fig, axes = plt.subplots(
        1,
        nr,
        figsize=(3.1 * nr, 3.6),
        sharey=True,
        squeeze=False
    )

    # ---------------------------------------------------------------------------------------------
    # STANDARD GAUSS-HERMITE FIT
    # ---------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    # FACE-ON STABILITY-CONSTRAINED GAUSS-HERMITE FIT
    # ---------------------------------------------------------------------------------------------

    def _gh_fit_and_plot(
            ax,
            xgrid,
            ydata,
            color,
            label,
            do_label
    ):
        """
        Fit a stability-constrained fourth-order Gauss-Hermite model
        to one face-on MDF.

        This fitter is used identically for the true and pPXF MDFs.

        Face-on stability choices
        -------------------------
        1. Uses all finite, non-negative metallicity bins.
        2. Normalizes the MDF before fitting.
        3. Initializes mu and sigma from the direct MDF moments.
        4. Prevents sigma from collapsing into an artificial narrow spike.
        5. Restricts h3 and h4 to moderate values.
        6. Penalizes negative fitted mass fractions.
        7. Penalizes peaks far above the observed MDF.
        8. Penalizes solutions that move too far from the direct MDF
           mean and width.
        9. Tests several starting values and retains the best solution.
        10. Falls back to a Gaussian-shaped solution when no stable
            Gauss-Hermite solution is found.
        """

        from scipy.optimize import least_squares

        xgrid = np.asarray(
            xgrid,
            dtype=float
        )

        data = np.asarray(
            ydata,
            dtype=float
        )

        valid = (
            np.isfinite(xgrid)
            & np.isfinite(data)
            & (data >= 0.0)
        )

        x = xgrid[valid]
        y = data[valid]

        if len(x) < 6:
            print(
                "Face-on GH fit skipped: "
                "fewer than six valid metallicity bins."
            )
            return None

        total = np.sum(y)

        if (
            not np.isfinite(total)
            or total <= 0.0
        ):
            print(
                "Face-on GH fit skipped: "
                "invalid MDF normalization."
            )
            return None

        # Normalize the MDF used by the fitter.
        y = y / total

        # -----------------------------------------------------------------------------------------
        # DIRECT MDF MOMENTS
        # -----------------------------------------------------------------------------------------

        direct_mu = np.sum(
            x * y
        )

        direct_variance = np.sum(
            y * (x - direct_mu)**2
        )

        direct_sigma = np.sqrt(
            max(
                direct_variance,
                0.12**2
            )
        )

        direct_sigma = np.clip(
            direct_sigma,
            0.12,
            0.55
        )

        observed_peak = np.max(y)

        # -----------------------------------------------------------------------------------------
        # FACE-ON FITTING CONSTRAINTS
        # -----------------------------------------------------------------------------------------

        sigma_min = 0.10
        sigma_max = 0.65

        h_limit = 0.18

        maximum_allowed_peak = (
            1.35 * observed_peak
        )

        minimum_allowed_mu = max(
            np.min(x) - 0.10,
            direct_mu - 0.30
        )

        maximum_allowed_mu = min(
            np.max(x) + 0.10,
            direct_mu + 0.30
        )

        amplitude_upper = max(
            1.50 * observed_peak,
            observed_peak + 1.0e-8
        )

        lower_bounds = np.array([
            0.0,
            minimum_allowed_mu,
            sigma_min,
            -h_limit,
            -h_limit
        ])

        upper_bounds = np.array([
            amplitude_upper,
            maximum_allowed_mu,
            sigma_max,
            h_limit,
            h_limit
        ])

        dense_x = np.linspace(
            np.min(x),
            np.max(x),
            600
        )

        # -----------------------------------------------------------------------------------------
        # PENALIZED RESIDUAL FUNCTION
        # -----------------------------------------------------------------------------------------

        def residuals(parameters):

            amplitude, mu, sigma, h3, h4 = parameters

            model_at_data = gauss_hermite(
                x,
                amplitude,
                mu,
                sigma,
                h3,
                h4
            )

            model_dense = gauss_hermite(
                dense_x,
                amplitude,
                mu,
                sigma,
                h3,
                h4
            )

            # Ordinary data residuals.
            data_residuals = (
                model_at_data - y
            )

            # MDFs cannot be negative.
            negative_penalty = (
                np.sqrt(250.0)
                * np.minimum(
                    model_dense,
                    0.0
                )
            )

            # Prevent artificial peaks between discrete metallicity bins.
            peak_penalty = (
                np.sqrt(150.0)
                * np.maximum(
                    model_dense
                    - maximum_allowed_peak,
                    0.0
                )
            )

            # Keep fitted mu reasonably close to the direct MDF mean.
            mu_penalty = np.array([
                np.sqrt(2.0)
                * (
                    mu - direct_mu
                )
                / 0.15
            ])

            # Keep fitted sigma reasonably close to the direct MDF width.
            sigma_penalty = np.array([
                np.sqrt(3.0)
                * (
                    sigma - direct_sigma
                )
                / 0.12
            ])

            # Mild regularization discourages unnecessary extreme
            # higher-order terms while still allowing real asymmetry.
            h3_penalty = np.array([
                np.sqrt(0.75)
                * h3
                / 0.12
            ])

            h4_penalty = np.array([
                np.sqrt(0.75)
                * h4
                / 0.12
            ])

            return np.concatenate([
                data_residuals,
                negative_penalty,
                peak_penalty,
                mu_penalty,
                sigma_penalty,
                h3_penalty,
                h4_penalty
            ])

        # -----------------------------------------------------------------------------------------
        # MULTIPLE STARTING POINTS
        # -----------------------------------------------------------------------------------------

        h_start_values = [
            -0.10,
            -0.05,
            0.0,
            0.05,
            0.10
        ]

        sigma_start_values = np.unique(
            np.clip(
                np.array([
                    direct_sigma,
                    0.85 * direct_sigma,
                    1.15 * direct_sigma
                ]),
                sigma_min,
                sigma_max
            )
        )

        best_result = None
        best_score = np.inf

        for sigma_start in sigma_start_values:
            for h3_start in h_start_values:
                for h4_start in h_start_values:

                    initial_parameters = np.array([
                        min(
                            observed_peak,
                            0.95 * amplitude_upper
                        ),
                        np.clip(
                            direct_mu,
                            minimum_allowed_mu,
                            maximum_allowed_mu
                        ),
                        sigma_start,
                        h3_start,
                        h4_start
                    ])

                    try:
                        result = least_squares(
                            residuals,
                            initial_parameters,
                            bounds=(
                                lower_bounds,
                                upper_bounds
                            ),
                            max_nfev=100000,
                            x_scale="jac",
                            ftol=1.0e-10,
                            xtol=1.0e-10,
                            gtol=1.0e-10
                        )

                    except (
                        RuntimeError,
                        ValueError,
                        FloatingPointError
                    ):
                        continue

                    if (
                        not result.success
                        or not np.all(
                            np.isfinite(result.x)
                        )
                    ):
                        continue

                    candidate_model = gauss_hermite(
                        dense_x,
                        *result.x
                    )

                    if not np.all(
                        np.isfinite(candidate_model)
                    ):
                        continue

                    amplitude, mu, sigma, h3, h4 = result.x

                    # Reject clearly unstable solutions.
                    if np.min(candidate_model) < -0.005:
                        continue

                    if (
                        np.max(candidate_model)
                        > 1.40 * observed_peak
                    ):
                        continue

                    if (
                        sigma
                        <= sigma_min + 0.005
                    ):
                        continue

                    if (
                        abs(h3)
                        >= h_limit - 0.005
                    ):
                        continue

                    if (
                        abs(h4)
                        >= h_limit - 0.005
                    ):
                        continue

                    score = np.sum(
                        residuals(result.x)**2
                    )

                    if score < best_score:
                        best_score = score
                        best_result = result

        # -----------------------------------------------------------------------------------------
        # GAUSSIAN FALLBACK
        # -----------------------------------------------------------------------------------------

        fitting_strategy = (
            "face-on constrained Gauss-Hermite"
        )

        if best_result is None:

            print(
                "No stable face-on GH solution found; "
                "using Gaussian fallback with h3 = h4 = 0."
            )

            def gaussian_residuals(parameters):

                amplitude, mu, sigma = parameters

                model_at_data = gauss_hermite(
                    x,
                    amplitude,
                    mu,
                    sigma,
                    0.0,
                    0.0
                )

                model_dense = gauss_hermite(
                    dense_x,
                    amplitude,
                    mu,
                    sigma,
                    0.0,
                    0.0
                )

                data_residuals = (
                    model_at_data - y
                )

                negative_penalty = (
                    np.sqrt(250.0)
                    * np.minimum(
                        model_dense,
                        0.0
                    )
                )

                peak_penalty = (
                    np.sqrt(150.0)
                    * np.maximum(
                        model_dense
                        - maximum_allowed_peak,
                        0.0
                    )
                )

                mu_penalty = np.array([
                    np.sqrt(2.0)
                    * (
                        mu - direct_mu
                    )
                    / 0.15
                ])

                sigma_penalty = np.array([
                    np.sqrt(3.0)
                    * (
                        sigma - direct_sigma
                    )
                    / 0.12
                ])

                return np.concatenate([
                    data_residuals,
                    negative_penalty,
                    peak_penalty,
                    mu_penalty,
                    sigma_penalty
                ])

            gaussian_initial = np.array([
                min(
                    observed_peak,
                    0.95 * amplitude_upper
                ),
                np.clip(
                    direct_mu,
                    minimum_allowed_mu,
                    maximum_allowed_mu
                ),
                direct_sigma
            ])

            try:
                gaussian_result = least_squares(
                    gaussian_residuals,
                    gaussian_initial,
                    bounds=(
                        np.array([
                            0.0,
                            minimum_allowed_mu,
                            sigma_min
                        ]),
                        np.array([
                            amplitude_upper,
                            maximum_allowed_mu,
                            sigma_max
                        ])
                    ),
                    max_nfev=100000,
                    x_scale="jac",
                    ftol=1.0e-10,
                    xtol=1.0e-10,
                    gtol=1.0e-10
                )

            except (
                RuntimeError,
                ValueError,
                FloatingPointError
            ):
                print(
                    "Face-on Gaussian fallback also failed."
                )
                return None

            if (
                not gaussian_result.success
                or not np.all(
                    np.isfinite(
                        gaussian_result.x
                    )
                )
            ):
                print(
                    "Face-on Gaussian fallback returned "
                    "an invalid solution."
                )
                return None

            amplitude, mu, sigma = (
                gaussian_result.x
            )

            h3 = 0.0
            h4 = 0.0

            fitting_strategy = (
                "Gaussian fallback"
            )

        else:

            amplitude, mu, sigma, h3, h4 = (
                best_result.x
            )

        # -----------------------------------------------------------------------------------------
        # DRAW THE FINAL FIT
        # -----------------------------------------------------------------------------------------

        plot_x = np.linspace(
            np.min(xgrid),
            np.max(xgrid),
            600
        )

        plot_y = gauss_hermite(
            plot_x,
            amplitude,
            mu,
            sigma,
            h3,
            h4
        )

        ax.plot(
            plot_x,
            plot_y,
            linestyle="--",
            linewidth=2.2,
            color=color,
            alpha=0.95,
            label=(
                label
                if do_label
                else None
            ),
            zorder=4
        )

        model_at_data = gauss_hermite(
            x,
            amplitude,
            mu,
            sigma,
            h3,
            h4
        )

        rmse = np.sqrt(
            np.mean(
                (
                    y - model_at_data
                )**2
            )
        )

        # Direct discrete moments are also saved for validation.
        direct_h3 = np.sum(
            y
            * (
                (
                    x - direct_mu
                )
                / direct_sigma
            )**3
        )

        direct_h4 = (
            np.sum(
                y
                * (
                    (
                        x - direct_mu
                    )
                    / direct_sigma
                )**4
            )
            - 3.0
        )

        return {
            "A": float(amplitude),
            "mu": float(mu),
            "sigma": float(sigma),
            "h3": float(h3),
            "h4": float(h4),
            "rmse": float(rmse),
            "direct_mu": float(direct_mu),
            "direct_sigma": float(direct_sigma),
            "direct_skewness": float(direct_h3),
            "direct_excess_kurtosis": float(direct_h4),
            "fitting_strategy": fitting_strategy
        }



    # ---------------------------------------------------------------------------------------------
    # CONSTRAINED TRUE-MDF DIAGNOSTIC FIT
    # ---------------------------------------------------------------------------------------------

    def _true_constrained_fit_and_plot(
            ax,
            xgrid,
            ydata,
            color,
            label,
            do_label
    ):
        """
        Fit an extra-conservative Gauss-Hermite model to the true face-on MDF.

        This purple dotted curve is a diagnostic comparison only.

        It is intentionally more restrictive than the primary face-on
        Gauss-Hermite fit:

        1. Requires a broad, physically plausible width.
        2. Restricts h3 and h4 to small values.
        3. Strongly penalizes negative MDF values.
        4. Strongly penalizes peaks above the actual true MDF.
        5. Anchors mu and sigma to the direct discrete MDF moments.
        6. Rejects solutions that sit against parameter boundaries.
        7. Falls back to a Gaussian when no stable constrained GH
           solution is available.
        """

        if not include_true_constrained_fit:
            return None

        from scipy.optimize import least_squares

        xgrid = np.asarray(
            xgrid,
            dtype=float
        )

        data = np.asarray(
            ydata,
            dtype=float
        )

        valid = (
            np.isfinite(xgrid)
            & np.isfinite(data)
            & (data >= 0.0)
        )

        x = xgrid[valid]
        y = data[valid]

        if len(x) < 6:
            print(
                "Purple face-on true GH fit skipped: "
                "fewer than six valid metallicity bins."
            )
            return None

        total = np.sum(y)

        if (
            not np.isfinite(total)
            or total <= 0.0
        ):
            print(
                "Purple face-on true GH fit skipped: "
                "invalid MDF normalization."
            )
            return None

        y = y / total

        # -----------------------------------------------------------------------------------------
        # DIRECT TRUE-MDF MOMENTS
        # -----------------------------------------------------------------------------------------

        direct_mu = np.sum(
            x * y
        )

        direct_variance = np.sum(
            y * (x - direct_mu)**2
        )

        direct_sigma = np.sqrt(
            max(
                direct_variance,
                0.14**2
            )
        )

        direct_sigma = np.clip(
            direct_sigma,
            0.14,
            0.60
        )

        observed_peak = np.max(y)

        # -----------------------------------------------------------------------------------------
        # EXTRA-CONSERVATIVE TRUE-MDF CONSTRAINTS
        # -----------------------------------------------------------------------------------------

        sigma_min = 0.13
        sigma_max = 0.65

        h_limit = 0.12

        maximum_allowed_peak = (
            1.20 * observed_peak
        )

        minimum_allowed_mu = max(
            np.min(x) - 0.05,
            direct_mu - 0.22
        )

        maximum_allowed_mu = min(
            np.max(x) + 0.05,
            direct_mu + 0.22
        )

        amplitude_upper = max(
            1.25 * observed_peak,
            observed_peak + 1.0e-8
        )

        lower_bounds = np.array([
            0.0,
            minimum_allowed_mu,
            sigma_min,
            -h_limit,
            -h_limit
        ])

        upper_bounds = np.array([
            amplitude_upper,
            maximum_allowed_mu,
            sigma_max,
            h_limit,
            h_limit
        ])

        dense_x = np.linspace(
            np.min(x),
            np.max(x),
            800
        )

        # -----------------------------------------------------------------------------------------
        # PENALIZED RESIDUAL FUNCTION
        # -----------------------------------------------------------------------------------------

        def residuals(parameters):

            amplitude, mu, sigma, h3, h4 = parameters

            model_at_data = gauss_hermite(
                x,
                amplitude,
                mu,
                sigma,
                h3,
                h4
            )

            model_dense = gauss_hermite(
                dense_x,
                amplitude,
                mu,
                sigma,
                h3,
                h4
            )

            data_residuals = (
                model_at_data - y
            )

            negative_penalty = (
                np.sqrt(400.0)
                * np.minimum(
                    model_dense,
                    0.0
                )
            )

            peak_penalty = (
                np.sqrt(300.0)
                * np.maximum(
                    model_dense
                    - maximum_allowed_peak,
                    0.0
                )
            )

            mu_penalty = np.array([
                np.sqrt(4.0)
                * (
                    mu - direct_mu
                )
                / 0.12
            ])

            sigma_penalty = np.array([
                np.sqrt(5.0)
                * (
                    sigma - direct_sigma
                )
                / 0.10
            ])

            h3_penalty = np.array([
                np.sqrt(2.0)
                * h3
                / 0.08
            ])

            h4_penalty = np.array([
                np.sqrt(2.0)
                * h4
                / 0.08
            ])

            return np.concatenate([
                data_residuals,
                negative_penalty,
                peak_penalty,
                mu_penalty,
                sigma_penalty,
                h3_penalty,
                h4_penalty
            ])

        # -----------------------------------------------------------------------------------------
        # MULTIPLE STARTING POINTS
        # -----------------------------------------------------------------------------------------

        h_start_values = [
            -0.08,
            -0.04,
            0.0,
            0.04,
            0.08
        ]

        sigma_start_values = np.unique(
            np.clip(
                np.array([
                    direct_sigma,
                    0.90 * direct_sigma,
                    1.10 * direct_sigma
                ]),
                sigma_min,
                sigma_max
            )
        )

        best_result = None
        best_score = np.inf

        for sigma_start in sigma_start_values:
            for h3_start in h_start_values:
                for h4_start in h_start_values:

                    initial_parameters = np.array([
                        min(
                            observed_peak,
                            0.95 * amplitude_upper
                        ),
                        np.clip(
                            direct_mu,
                            minimum_allowed_mu,
                            maximum_allowed_mu
                        ),
                        sigma_start,
                        h3_start,
                        h4_start
                    ])

                    try:
                        result = least_squares(
                            residuals,
                            initial_parameters,
                            bounds=(
                                lower_bounds,
                                upper_bounds
                            ),
                            max_nfev=100000,
                            x_scale="jac",
                            ftol=1.0e-10,
                            xtol=1.0e-10,
                            gtol=1.0e-10
                        )

                    except (
                        RuntimeError,
                        ValueError,
                        FloatingPointError
                    ):
                        continue

                    if (
                        not result.success
                        or not np.all(
                            np.isfinite(result.x)
                        )
                    ):
                        continue

                    candidate_model = gauss_hermite(
                        dense_x,
                        *result.x
                    )

                    if not np.all(
                        np.isfinite(candidate_model)
                    ):
                        continue

                    amplitude, mu, sigma, h3, h4 = result.x

                    if np.min(candidate_model) < -0.002:
                        continue

                    if (
                        np.max(candidate_model)
                        > 1.23 * observed_peak
                    ):
                        continue

                    if (
                        sigma
                        <= sigma_min + 0.005
                    ):
                        continue

                    if (
                        abs(h3)
                        >= h_limit - 0.004
                    ):
                        continue

                    if (
                        abs(h4)
                        >= h_limit - 0.004
                    ):
                        continue

                    score = np.sum(
                        residuals(result.x)**2
                    )

                    if score < best_score:
                        best_score = score
                        best_result = result

        # -----------------------------------------------------------------------------------------
        # CONSERVATIVE GAUSSIAN FALLBACK
        # -----------------------------------------------------------------------------------------

        fitting_strategy = (
            "extra-constrained face-on true GH"
        )

        if best_result is None:

            print(
                "No stable purple true-MDF GH solution found; "
                "using conservative Gaussian fallback."
            )

            def gaussian_residuals(parameters):

                amplitude, mu, sigma = parameters

                model_at_data = gauss_hermite(
                    x,
                    amplitude,
                    mu,
                    sigma,
                    0.0,
                    0.0
                )

                model_dense = gauss_hermite(
                    dense_x,
                    amplitude,
                    mu,
                    sigma,
                    0.0,
                    0.0
                )

                data_residuals = (
                    model_at_data - y
                )

                negative_penalty = (
                    np.sqrt(400.0)
                    * np.minimum(
                        model_dense,
                        0.0
                    )
                )

                peak_penalty = (
                    np.sqrt(300.0)
                    * np.maximum(
                        model_dense
                        - maximum_allowed_peak,
                        0.0
                    )
                )

                mu_penalty = np.array([
                    np.sqrt(4.0)
                    * (
                        mu - direct_mu
                    )
                    / 0.12
                ])

                sigma_penalty = np.array([
                    np.sqrt(5.0)
                    * (
                        sigma - direct_sigma
                    )
                    / 0.10
                ])

                return np.concatenate([
                    data_residuals,
                    negative_penalty,
                    peak_penalty,
                    mu_penalty,
                    sigma_penalty
                ])

            gaussian_initial = np.array([
                min(
                    observed_peak,
                    0.95 * amplitude_upper
                ),
                np.clip(
                    direct_mu,
                    minimum_allowed_mu,
                    maximum_allowed_mu
                ),
                direct_sigma
            ])

            try:
                gaussian_result = least_squares(
                    gaussian_residuals,
                    gaussian_initial,
                    bounds=(
                        np.array([
                            0.0,
                            minimum_allowed_mu,
                            sigma_min
                        ]),
                        np.array([
                            amplitude_upper,
                            maximum_allowed_mu,
                            sigma_max
                        ])
                    ),
                    max_nfev=100000,
                    x_scale="jac",
                    ftol=1.0e-10,
                    xtol=1.0e-10,
                    gtol=1.0e-10
                )

            except (
                RuntimeError,
                ValueError,
                FloatingPointError
            ):
                print(
                    "Purple true-MDF Gaussian fallback failed."
                )
                return None

            if (
                not gaussian_result.success
                or not np.all(
                    np.isfinite(
                        gaussian_result.x
                    )
                )
            ):
                print(
                    "Purple true-MDF Gaussian fallback "
                    "returned an invalid solution."
                )
                return None

            amplitude, mu, sigma = (
                gaussian_result.x
            )

            h3 = 0.0
            h4 = 0.0

            fitting_strategy = (
                "conservative Gaussian fallback"
            )

        else:

            amplitude, mu, sigma, h3, h4 = (
                best_result.x
            )

        # -----------------------------------------------------------------------------------------
        # DRAW PURPLE DIAGNOSTIC CURVE
        # -----------------------------------------------------------------------------------------

        plot_x = np.linspace(
            np.min(xgrid),
            np.max(xgrid),
            800
        )

        plot_y = gauss_hermite(
            plot_x,
            amplitude,
            mu,
            sigma,
            h3,
            h4
        )

        ax.plot(
            plot_x,
            plot_y,
            color=color,
            linestyle=":",
            linewidth=2.5,
            alpha=0.95,
            label=(
                label
                if do_label
                else None
            ),
            zorder=5
        )

        model_at_data = gauss_hermite(
            x,
            amplitude,
            mu,
            sigma,
            h3,
            h4
        )

        rmse = np.sqrt(
            np.mean(
                (
                    y - model_at_data
                )**2
            )
        )

        direct_skewness = np.sum(
            y
            * (
                (
                    x - direct_mu
                )
                / direct_sigma
            )**3
        )

        direct_excess_kurtosis = (
            np.sum(
                y
                * (
                    (
                        x - direct_mu
                    )
                    / direct_sigma
                )**4
            )
            - 3.0
        )

        return {
            "A": float(amplitude),
            "mu": float(mu),
            "sigma": float(sigma),
            "h3": float(h3),
            "h4": float(h4),
            "rmse": float(rmse),
            "direct_mu": float(direct_mu),
            "direct_sigma": float(direct_sigma),
            "direct_skewness": float(direct_skewness),
            "direct_excess_kurtosis": float(
                direct_excess_kurtosis
            ),
            "fitting_strategy": fitting_strategy
        }


    # ---------------------------------------------------------------------------------------------
    # DRAW ONE PANEL PER TRUE FACE-ON RADIAL INTERVAL
    # ---------------------------------------------------------------------------------------------

    for col in range(nr):

        ax = axes[0, col]
        ax.tick_params(direction='in')

        r0, r1 = r_range_list[col]

        recovered_distribution = distri_bin_list[col]
        true_distribution = distri_bin_true_list[col]

        # Panel title contains only true face-on radius.
        ax.text(
            0.5,
            0.965,
            rf'$\mathbf{{{r0:g}<R/\mathrm{{kpc}}<{r1:g}}}$',
            ha='center',
            va='top',
            transform=ax.transAxes,
            fontsize=panel_fs,
            fontweight='bold'
        )

        ax.text(
            0.5,
            0.885,
            rf'$\mathbf{{N_{{\mathrm{{bin}}}}={coverage_list[col]}}}$',
            ha='center',
            va='top',
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight='bold'
        )

        if (
            recovered_distribution is None
            or true_distribution is None
        ):
            ax.set_axis_off()
            continue

        if alpha_index >= recovered_distribution.shape[1]:
            raise IndexError(
                f'alpha_index={alpha_index} exceeds the pPXF '
                f'alpha-grid size {recovered_distribution.shape[1]}.'
            )

        if alpha_index >= true_distribution.shape[1]:
            raise IndexError(
                f'alpha_index={alpha_index} exceeds the true '
                f'alpha-grid size {true_distribution.shape[1]}.'
            )

        # Keep the requested alpha slice. For this project alpha_index=0
        # corresponds to [alpha/Fe] = 0.0.
        ppxf_mdf = np.asarray(
            recovered_distribution[:, alpha_index],
            dtype=float
        ).copy()

        true_mdf = np.asarray(
            true_distribution[:, alpha_index],
            dtype=float
        ).copy()

        ppxf_total = np.nansum(ppxf_mdf)
        true_total = np.nansum(true_mdf)

        if (
            not np.isfinite(ppxf_total)
            or not np.isfinite(true_total)
            or ppxf_total <= 0.0
            or true_total <= 0.0
        ):
            ax.set_axis_off()
            continue

        # Normalize each MDF independently within the radial panel.
        ppxf_mdf /= ppxf_total
        true_mdf /= true_total

        do_label = (col == 0)

        # True/input MDF.
        ax.plot(
            metal_grid_true,
            true_mdf,
            color=true_color,
            linewidth=2.0,
            linestyle='-',
            label=(
                r'True MDF ([$\alpha$/Fe]=0.0)'
                if do_label
                else None
            ),
            zorder=2
        )

        # pPXF-recovered MDF.
        ax.plot(
            metal_grid,
            ppxf_mdf,
            color=ppxf_color,
            linewidth=2.0,
            linestyle='-',
            label=(
                r'pPXF MDF ([$\alpha$/Fe]=0.0)'
                if do_label
                else None
            ),
            zorder=3
        )

        # Fit both distributions.
        true_fit = _gh_fit_and_plot(
            ax,
            metal_grid_true,
            true_mdf,
            true_color,
            r'True MDF GH fit',
            do_label
        )

        ppxf_fit = _gh_fit_and_plot(
            ax,
            metal_grid,
            ppxf_mdf,
            ppxf_color,
            r'pPXF MDF GH fit',
            do_label
        )

        true_constrained_fit = _true_constrained_fit_and_plot(
            ax,
            metal_grid_true,
            true_mdf,
            constrained_color,
            r'True MDF constrained GH fit',
            do_label
        )

        fit_results.append({
            'rbin': (
                float(r0),
                float(r1)
            ),
            'number_of_spatial_bins': int(
                coverage_list[col]
            ),
            'bin_ids': binid_list[col].copy(),
            'ppxf': ppxf_fit,
            'true': true_fit,
            'true_constrained': true_constrained_fit
        })

        # -----------------------------------------------------------------------------------------
        # GH PARAMETER TEXT
        # -----------------------------------------------------------------------------------------

        text_y0 = 0.79

        if true_fit is not None:

            true_lines = [
                r'$\mathbf{True\ GH}$',
                rf'$\mu={true_fit["mu"]:+.3f}$',
                rf'$\sigma={true_fit["sigma"]:.3f}$',
                rf'$h_3={true_fit["h3"]:+.3f}$',
                rf'$h_4={true_fit["h4"]:+.3f}$'
            ]

            for line_index, text_line in enumerate(
                true_lines
            ):
                ax.text(
                    0.03,
                    text_y0 - line_index * moment_dy,
                    text_line,
                    transform=ax.transAxes,
                    ha='left',
                    va='top',
                    fontsize=moment_fs,
                    fontweight='bold',
                    color=true_color,
                    bbox={
                        'facecolor': 'white',
                        'edgecolor': 'none',
                        'alpha': 0.72,
                        'pad': 0.4
                    }
                )

        if ppxf_fit is not None:

            ppxf_lines = [
                r'$\mathbf{pPXF\ GH}$',
                rf'$\mu={ppxf_fit["mu"]:+.3f}$',
                rf'$\sigma={ppxf_fit["sigma"]:.3f}$',
                rf'$h_3={ppxf_fit["h3"]:+.3f}$',
                rf'$h_4={ppxf_fit["h4"]:+.3f}$'
            ]

            for line_index, text_line in enumerate(
                ppxf_lines
            ):
                ax.text(
                    0.97,
                    text_y0 - line_index * moment_dy,
                    text_line,
                    transform=ax.transAxes,
                    ha='right',
                    va='top',
                    fontsize=moment_fs,
                    fontweight='bold',
                    color=ppxf_color,
                    bbox={
                        'facecolor': 'white',
                        'edgecolor': 'none',
                        'alpha': 0.72,
                        'pad': 0.4
                    }
                )

        ax.set_xlim(-1.0, 0.5)
        ax.set_ylim(0.0, 0.7)

    # ---------------------------------------------------------------------------------------------
    # GLOBAL FIGURE FORMATTING
    # ---------------------------------------------------------------------------------------------

    global_axis = fig.add_subplot(
        111,
        frame_on=False
    )

    global_axis.tick_params(
        labelcolor='none',
        bottom=False,
        left=False
    )

    global_axis.set_xlabel(
        '[M/H]',
        fontsize=17,
        fontweight='bold',
        labelpad=22
    )

    global_axis.set_ylabel(
        fraction_type,
        fontsize=17,
        fontweight='bold',
        labelpad=22
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.79,
        bottom=0.18,
        wspace=0.18
    )

    for ax in axes.flatten():

        if not ax.axison:
            continue

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=12,
            width=1.5,
            length=6
        )

        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')

        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')

        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

    # ---------------------------------------------------------------------------------------------
    # LEGEND
    # ---------------------------------------------------------------------------------------------

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=true_color,
            linewidth=2.0,
            linestyle='-',
            label=r'True MDF ([$\alpha$/Fe]=0.0)'
        ),
        Line2D(
            [0],
            [0],
            color=true_color,
            linewidth=2.2,
            linestyle='--',
            label=r'True MDF GH fit'
        ),
        Line2D(
            [0],
            [0],
            color=ppxf_color,
            linewidth=2.0,
            linestyle='-',
            label=r'pPXF MDF ([$\alpha$/Fe]=0.0)'
        ),
        Line2D(
            [0],
            [0],
            color=ppxf_color,
            linewidth=2.2,
            linestyle='--',
            label=r'pPXF MDF GH fit'
        )
    ]

    if include_true_constrained_fit:
        legend_handles.insert(
            2,
            Line2D(
                [0],
                [0],
                color=constrained_color,
                linewidth=2.4,
                linestyle=':',
                label=r'True MDF constrained GH fit'
            )
        )

    fig.suptitle(
        figure_title,
        fontsize=18,
        fontweight='bold',
        y=0.985
    )

    legend = fig.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=len(legend_handles),
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, 0.91)
    )

    for legend_text in legend.get_texts():
        legend_text.set_fontweight('bold')

    return fig, fit_results

# ----------------------------------------------------------------------------------------------------------------
def plot_mh_alpha_rz_hist_mode(weights_values, weights_true_values, rbin, zbin, d, reg_dim, reg_dim_true, results_table,
                          metal_grid, metal_grid_true, alpha_grid, alpha_grid_true, fraction_type='mass fraction', mode='metal'):

    fig, axes = plt.subplots(len(zbin) - 1, len(rbin) - 1, figsize=(2 * (len(rbin) - 1), 1.8 * (len(zbin) - 1)), sharey=True)

    if mode == 'metal':
        xlabel = '[M/H]'
    elif mode == 'alpha':
        xlabel = r'[$\alpha$/Fe]'

    r_range_list, z_range_list, distri_bin_list, i_list, j_list = cal_rz_distri_bin(rbin, zbin, d, results_table, weights_values, reg_dim)
    r_range_list, z_range_list, distri_bin_true_list, i_list, j_list = cal_rz_distri_bin(rbin, zbin, d, results_table, weights_true_values, reg_dim_true)

    for k in range(len(distri_bin_list)):

        axes[-j_list[k] - 1, i_list[k]].tick_params(direction="in")
        if -j_list[k] - 1 != -1:
            axes[-j_list[k] - 1, i_list[k]].xaxis.set_ticklabels([])
        axes[-j_list[k] - 1, i_list[k]].text(0.5, 0.98, '%s<|z|/kpc<%s' % (z_range_list[k][0], z_range_list[k][1]), ha='center', va='top',
                             transform=axes[-j_list[k] - 1, i_list[k]].transAxes, fontsize=11.2)
        axes[-j_list[k] - 1, i_list[k]].text(0.5, 0.85, '%s<R/kpc<%s' % (r_range_list[k][0], r_range_list[k][1]), ha='center', va='top',
                             transform=axes[-j_list[k] - 1, i_list[k]].transAxes, fontsize=11.2)

        if mode == 'metal':
            axes[-j_list[k] - 1, i_list[k]].plot(metal_grid, np.sum(distri_bin_list[k], axis=1) / np.sum(distri_bin_list[k]), c='black',
                                 label=r'PPXF')
            axes[-j_list[k] - 1, i_list[k]].plot(metal_grid_true, np.sum(distri_bin_true_list[k], axis=1) / np.sum(distri_bin_true_list[k]), ls='dashed',
                                 c='black', label=r'True')
            axes[-j_list[k] - 1, i_list[k]].set_xlim(-1,0.5)
            axes[-j_list[k] - 1, i_list[k]].set_ylim(0,0.7)
            if len(np.sum(distri_bin_list[k], axis=1)) == len(np.sum(distri_bin_true_list[k], axis=1)):
                axes[-j_list[k] - 1, i_list[k]].text(0.02, 0.30, "Corr=%.3f" % pearsonr(np.sum(distri_bin_list[k], axis=1) / np.sum(distri_bin_list[k]),
                                                                        np.sum(distri_bin_true_list[k], axis=1) / np.sum(distri_bin_true_list[k]))[0],
                                     ha='left', va='bottom', transform=axes[-j_list[k] - 1, i_list[k]].transAxes, c='black', fontsize=11)
        elif mode == 'alpha':
            axes[-j_list[k] - 1, i_list[k]].plot(alpha_grid, np.sum(distri_bin_list[k], axis=0) / np.sum(distri_bin_list[k]), c='black',
                                 label=r'PPXF')
            axes[-j_list[k] - 1, i_list[k]].plot(alpha_grid_true, np.sum(distri_bin_true_list[k], axis=0) / np.sum(distri_bin_true_list[k]), ls='dashed',
                                 c='black', label=r'True')
            axes[-j_list[k] - 1, i_list[k]].set_xlim(-1,0.6)
            axes[-j_list[k] - 1, i_list[k]].set_ylim(0,0.69)
            if len(np.sum(distri_bin_list[k], axis=0)) == len(np.sum(distri_bin_true_list[k], axis=0)):
                axes[-j_list[k] - 1, i_list[k]].text(0.98, 0.30, "Corr=%.3f" % pearsonr(np.sum(distri_bin_list[k], axis=0) / np.sum(distri_bin_list[k]),
                                                                        np.sum(distri_bin_true_list[k], axis=0) / np.sum(distri_bin_true_list[k]))[0],
                                     ha='right', va='bottom', transform=axes[-j_list[k] - 1, i_list[k]].transAxes, c='black', fontsize=11)


    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(fraction_type, fontsize=15)

    fig.subplots_adjust(left=0.05, right=0.995, top=0.96, bottom=0.12, hspace=0.1, wspace=0.0)
    axes[0, 0].legend(bbox_to_anchor=(0, 1.06, 1, 0.102), loc='lower left', ncol=4, fontsize=11)
    
    return fig

# ----------------------------------------------------------------------------------------------------------------
def plot_mh_alpha_rz(weights_values, rbin, zbin, d, results_table, reg_dim, metal_grid, alpha_grid):
    x = metal_grid  # Grid centers
    y = alpha_grid
    xb = (x[1:] + x[:-1])/2  # internal grid borders
    yb = (y[1:] + y[:-1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])  # 1st/last border
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])
    xxb, yyb = np.meshgrid(np.diff(xb), np.diff(yb))
    grid_area_2d = xxb * yyb

    fig, axes = plt.subplots(len(zbin)-1, len(rbin)-1, figsize=(3*(len(rbin)-1), 2*(len(zbin)-1)))

    r_range_list, z_range_list, distri_bin_list, i_list, j_list = cal_rz_distri_bin(rbin, zbin, d, results_table, weights_values, reg_dim)

    vmin, vmax = np.nanpercentile(np.array([distri_bin.T / grid_area_2d for distri_bin in distri_bin_list]), (5, 95))
    for k in range(len(distri_bin_list)):
        pc = axes[-j_list[k]-1, i_list[k]].pcolormesh(xb, yb, distri_bin_list[k].T / grid_area_2d, edgecolors='face', cmap='Oranges')
        axes[-j_list[k]-1, i_list[k]].tick_params(direction="in")
        axes[-j_list[k] - 1, i_list[k]].set_xlim(-1,0.5)
        axes[-j_list[k] - 1, i_list[k]].set_ylim(0,0.7)
        if -j_list[k]-1 != -1:
            axes[-j_list[k]-1, i_list[k]].xaxis.set_ticklabels([])
        if i_list[k] != 0:
            axes[-j_list[k]-1, i_list[k]].yaxis.set_ticklabels([])

        axes[-j_list[k]-1, i_list[k]].text(0.5, 0.98, '%s<|z|/kpc<%s' % (z_range_list[k][0], z_range_list[k][1]),
                                           ha='center', va='top', transform=axes[-j_list[k]-1, i_list[k]].transAxes,
                                           weight='bold', fontsize=10)
        axes[-j_list[k]-1, i_list[k]].text(0.5, 0.02, '%s<R/kpc<%s' % (r_range_list[k][0], r_range_list[k][1]),
                                           ha='center', va='bottom', transform=axes[-j_list[k]-1, i_list[k]].transAxes,
                                           weight='bold', fontsize=10)

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel('[M/H]', fontsize=15)
    plt.ylabel(r'[$\alpha$/Fe]', fontsize=15)

    p0 = axes[0, -1].get_position().get_points().flatten()
    p4 = axes[-1, -1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[2]+0.01, p4[1], 0.01, p0[3]-p4[1]])

    cb = plt.colorbar(pc, cax=ax_cbar, orientation='vertical')
    cb.set_label('mass fraction / bin_area', fontsize=12)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    fig.subplots_adjust(wspace=0)
    return fig

# ----------------------------------------------------------------------------------------------------------------
def mask_mass_arrays(mass_array, mask_age, mask_metal, mask_alpha):
    mass_array = mass_array[mask_age, :, :]
    mass_array = mass_array[:, mask_metal, :]
    mass_array = mass_array[:, :, mask_alpha]
    return mass_array

# ----------------------------------------------------------------------------------------------------------------
def load_mass_fractions(cube_gist_path, cube_gist_run):
    cube_gist_run_path = cube_gist_path + cube_gist_run + '/'
    filename = cube_gist_run_path + cube_gist_run

    if 'rmax' in cube_gist_run:
        weights = fits.open(filename + '_sfh-weights_rmax.fits')
    else:
        weights = fits.open(filename + '_sfh-weights.fits')
    weights_values = np.zeros(weights[1].data.WEIGHTS.shape)
    for i in range(weights_values.shape[0]):
        weights_values[i, :] = weights[1].data.WEIGHTS[i, :] * weights[1].data.WEIGHTS_FACTOR[i]
    logage_grid = weights[2].data.LOGAGE
    age_grid = 10 ** (weights[2].data.LOGAGE)
    metal_grid = weights[2].data.METAL
    alpha_grid = weights[2].data.ALPHA
    unique_logage_grid = np.unique(logage_grid)
    unique_age_grid = np.unique(age_grid)
    unique_metal_grid = np.unique(metal_grid)
    unique_alpha_grid = np.unique(alpha_grid)
    reg_dim = np.array(
        [np.unique(logage_grid).shape, np.unique(metal_grid).shape, np.unique(alpha_grid).shape]).reshape(-1)

    if 'LW' in cube_gist_run:
        weights_true = fits.open(filename + '_sfh-weights_true_lw.fits')
    else:
        weights_true = fits.open(filename + '_sfh-weights_true_mw.fits')
    weights_true_values = weights_true[1].data.WEIGHTS
    logage_grid_true = weights_true[2].data.LOGAGE
    age_grid_true = 10 ** (weights_true[2].data.LOGAGE)
    metal_grid_true = weights_true[2].data.METAL
    alpha_grid_true = weights_true[2].data.ALPHA
    unique_logage_grid_true = np.unique(logage_grid_true)
    unique_age_grid_true = np.unique(age_grid_true)
    unique_metal_grid_true = np.unique(metal_grid_true)
    unique_alpha_grid_true = np.unique(alpha_grid_true)
    reg_dim_true = np.array([np.unique(logage_grid_true).shape, np.unique(metal_grid_true).shape,
                             np.unique(alpha_grid_true).shape]).reshape(-1)

    mask_age = np.array([True] * len(unique_logage_grid))
    if 'pegasehrinterp' in cube_gist_path and 'noyoungest_true' in cube_gist_path:
        mask_age = np.where(10 ** unique_logage_grid>0.22)[0]
    elif 'loginterp' in cube_gist_path and 'noyoungest_true' in cube_gist_path:
        mask_age = np.where(10 ** unique_logage_grid>0.25)[0]
    elif 'interp' in cube_gist_path and 'noyoungest_true' in cube_gist_path:
        mask_age = np.where(10 ** unique_logage_grid>0.25)[0]
    elif 'conroymiles' in cube_gist_path and 'noyoungest_true' in cube_gist_path:
        mask_age = np.where(10 ** unique_logage_grid>0.2)[0]
    elif 'noyoungest_true' in cube_gist_path:
        mask_age = np.where(10 ** unique_logage_grid>0.24)[0]
    mask_metal = np.array([True] * len(unique_metal_grid))
    mask_alpha = np.array([True] * len(unique_alpha_grid))


    mask_age_true = np.array([x.round(6) in unique_logage_grid_true.round(6) for x in unique_logage_grid_true])
    if 'pegasehrinterp' in cube_gist_path and 'noyoungest_true' in cube_gist_path:
        mask_age_true = np.where(10 ** unique_logage_grid_true>0.22)[0]
    elif 'loginterp' in cube_gist_path and 'noyoungest_true' in cube_gist_path:
        mask_age_true = np.where(10 ** unique_logage_grid_true>0.25)[0]
    elif 'interp' in cube_gist_path and 'noyoungest_true' in cube_gist_path:
        mask_age_true = np.where(10 ** unique_logage_grid_true>0.25)[0]
    elif 'conroymiles' in cube_gist_path and 'noyoungest_true' in cube_gist_path:
        mask_age_true = np.where(10 ** unique_logage_grid_true>0.2)[0]
    elif 'noyoungest_true' in cube_gist_path:
        mask_age_true = np.where(10 ** unique_logage_grid_true>0.24)[0]
    mask_metal_true = np.array([x.round(6) in unique_metal_grid_true.round(6) for x in unique_metal_grid_true])
    mask_alpha_true = np.array([x.round(6) in unique_alpha_grid_true.round(6) for x in unique_alpha_grid_true])

    # Version with zb
    # # Divide the mass-fraction by the area of each bin, to be per dex per Gyr
    # x = np.unique(metal_grid)[mask_metal]  # Grid centers
    # y = np.unique(logage_grid)[mask_age]
    # z = np.unique(alpha_grid)[mask_alpha]
    # xb = (x[1:] + x[:-1]) / 2  # internal grid borders
    # yb = (y[1:] + y[:-1]) / 2
    # zb = (z[1:] + z[:-1]) / 2
    # xb = np.hstack([1.5 * x[0] - x[1] / 2, xb, 1.5 * x[-1] - x[-2] / 2])  # 1st/last border
    # yb_log = np.hstack([1.5 * y[0] - y[1] / 2, yb, 1.5 * y[-1] - y[-2] / 2])
    # zb = np.hstack([1.5 * z[0] - z[1] / 2, zb, 1.5 * z[-1] - z[-2] / 2])
    # xxb_log, yyb_log, zzb_log = np.meshgrid(np.diff(xb), np.diff(yb_log), np.diff(zb))
    # grid_area_2d_log = xxb_log * yyb_log
    # yb = 10 ** (np.hstack([1.5 * y[0] - y[1] / 2, yb, 1.5 * y[-1] - y[-2] / 2]))
    # if 'noyoungest_true' in cube_gist_path:
    #     yb[0] = 0.25
    # # else:
    # #     yb[0] = 0
    # xxb, yyb, zzb = np.meshgrid(np.diff(xb), np.diff(yb), np.diff(zb))
    # grid_area_2d = xxb * yyb
    #
    # x_true = np.unique(metal_grid_true)[mask_metal_true]  # Grid centers
    # y_true = np.unique(logage_grid_true)[mask_age_true]
    # z_true = np.unique(alpha_grid_true)[mask_alpha_true]
    # xb_true = (x_true[1:] + x_true[:-1]) / 2  # internal grid borders
    # yb_true = (y_true[1:] + y_true[:-1]) / 2
    # zb_true = (z_true[1:] + z_true[:-1]) / 2
    # xb_true = np.hstack([1.5 * x_true[0] - x_true[1] / 2, xb_true, 1.5 * x_true[-1] - x_true[-2] / 2])  # 1st/last border
    # yb_true_log = np.hstack([1.5 * y[0] - y[1] / 2, yb_true, 1.5 * y[-1] - y[-2] / 2])
    # zb_true = np.hstack([1.5 * z_true[0] - z_true[1] / 2, zb_true, 1.5 * z_true[-1] - z_true[-2] / 2])
    # xxb_true_log, yyb_true_log, zzb_true_log = np.meshgrid(np.diff(xb_true), np.diff(yb_true_log), np.diff(zb_true))
    # grid_area_2d_true_log = xxb_true_log * yyb_true_log
    # yb_true = 10 ** (np.hstack([1.5 * y_true[0] - y_true[1] / 2, yb_true, 1.5 * y_true[-1] - y_true[-2] / 2]))
    # if 'noyoungest_true' in cube_gist_path:
    #     yb_true[0] = 0.25
    # xxb_true, yyb_true, zzb_true = np.meshgrid(np.diff(xb_true), np.diff(yb_true), np.diff(zb_true))
    # grid_area_2d_true = xxb_true * yyb_true

    # Version with no zb
    # # Divide the mass-fraction by the area of each bin, to be per dex per Gyr
    # x = np.unique(metal_grid)[mask_metal]  # Grid centers
    # y = np.unique(logage_grid)[mask_age]
    # z = np.unique(alpha_grid)[mask_alpha]
    # xb = (x[1:] + x[:-1]) / 2  # internal grid borders
    # yb = (y[1:] + y[:-1]) / 2
    # xb = np.hstack([1.5 * x[0] - x[1] / 2, xb, 1.5 * x[-1] - x[-2] / 2])  # 1st/last border
    # yb_log = np.hstack([1.5 * y[0] - y[1] / 2, yb, 1.5 * y[-1] - y[-2] / 2])
    # xxb_log, yyb_log = np.meshgrid(np.diff(xb), np.diff(yb_log))
    # grid_area_2d_log = xxb_log * yyb_log
    # yb = 10 ** (np.hstack([1.5 * y[0] - y[1] / 2, yb, 1.5 * y[-1] - y[-2] / 2]))
    # if 'noyoungest_true' in cube_gist_path:
    #     yb[0] = 0.25
    # # else:
    # #     yb[0] = 0
    # xxb, yyb = np.meshgrid(np.diff(xb), np.diff(yb))
    # grid_area_2d = xxb * yyb
    #
    # x_true = np.unique(metal_grid_true)[mask_metal_true]  # Grid centers
    # y_true = np.unique(logage_grid_true)[mask_age_true]
    # z_true = np.unique(alpha_grid_true)[mask_alpha_true]
    # xb_true = (x_true[1:] + x_true[:-1]) / 2  # internal grid borders
    # yb_true = (y_true[1:] + y_true[:-1]) / 2
    # xb_true = np.hstack([1.5 * x_true[0] - x_true[1] / 2, xb_true, 1.5 * x_true[-1] - x_true[-2] / 2])  # 1st/last border
    # yb_true_log = np.hstack([1.5 * y[0] - y[1] / 2, yb_true, 1.5 * y[-1] - y[-2] / 2])
    # xxb_true_log, yyb_true_log = np.meshgrid(np.diff(xb_true), np.diff(yb_true_log))
    # grid_area_2d_true_log = xxb_true_log * yyb_true_log
    # yb_true = 10 ** (np.hstack([1.5 * y_true[0] - y_true[1] / 2, yb_true, 1.5 * y_true[-1] - y_true[-2] / 2]))
    # if 'noyoungest_true' in cube_gist_path:
    #     yb_true[0] = 0.25
    # xxb_true, yyb_true = np.meshgrid(np.diff(xb_true), np.diff(yb_true))
    # grid_area_2d_true = xxb_true * yyb_true


    # version with modified zb
    # Divide the mass-fraction by the area of each bin, to be per dex per Gyr
    x = np.unique(metal_grid)[mask_metal]  # Grid centers
    y = np.unique(logage_grid)[mask_age]
    z = np.unique(alpha_grid)[mask_alpha]
    xb = (x[1:] + x[:-1]) / 2  # internal grid borders
    yb = (y[1:] + y[:-1]) / 2
    xb = np.hstack([1.5 * x[0] - x[1] / 2, xb, 1.5 * x[-1] - x[-2] / 2])  # 1st/last border
    yb_log = np.hstack([1.5 * y[0] - y[1] / 2, yb, 1.5 * y[-1] - y[-2] / 2])
    yb = 10 ** (np.hstack([1.5 * y[0] - y[1] / 2, yb, 1.5 * y[-1] - y[-2] / 2]))
    if 'noyoungest_true' in cube_gist_path:
        yb[0] = 0.25
    # else:
    #     yb[0] = 0
    if len(z) > 1:
        zb = (z[1:] + z[:-1]) / 2
        zb = np.hstack([1.5 * z[0] - z[1] / 2, zb, 1.5 * z[-1] - z[-2] / 2])
        xxb_log, yyb_log, zzb_log = np.meshgrid(np.diff(xb), np.diff(yb_log), np.diff(zb))
        xxb, yyb, zzb = np.meshgrid(np.diff(xb), np.diff(yb), np.diff(zb))
    else:
        zb = np.array([0])
        xxb_log, yyb_log, zzb_log = np.meshgrid(np.diff(xb), np.diff(yb_log), np.array([0]))
        xxb, yyb, zzb = np.meshgrid(np.diff(xb), np.diff(yb), np.array([0]))
    grid_area_2d_log = xxb_log * yyb_log
    grid_area_2d = xxb * yyb

    x_true = np.unique(metal_grid_true)[mask_metal_true]  # Grid centers
    y_true = np.unique(logage_grid_true)[mask_age_true]
    z_true = np.unique(alpha_grid_true)[mask_alpha_true]
    xb_true = (x_true[1:] + x_true[:-1]) / 2  # internal grid borders
    yb_true = (y_true[1:] + y_true[:-1]) / 2
    xb_true = np.hstack([1.5 * x_true[0] - x_true[1] / 2, xb_true, 1.5 * x_true[-1] - x_true[-2] / 2])  # 1st/last border
    yb_true_log = np.hstack([1.5 * y[0] - y[1] / 2, yb_true, 1.5 * y[-1] - y[-2] / 2])
    yb_true = 10 ** (np.hstack([1.5 * y_true[0] - y_true[1] / 2, yb_true, 1.5 * y_true[-1] - y_true[-2] / 2]))
    if 'noyoungest_true' in cube_gist_path:
        yb_true[0] = 0.25
    if len(z_true) > 1:
        zb_true = (z_true[1:] + z_true[:-1]) / 2
        zb_true = np.hstack([1.5 * z_true[0] - z_true[1] / 2, zb_true, 1.5 * z_true[-1] - z_true[-2] / 2])
        xxb_true_log, yyb_true_log, zzb_true_log = np.meshgrid(np.diff(xb_true), np.diff(yb_true_log), np.diff(zb_true))
        xxb_true, yyb_true, zzb_true = np.meshgrid(np.diff(xb_true), np.diff(yb_true), np.diff(zb_true))
    else:
        zb_true = np.array([0])
        xxb_true_log, yyb_true_log, zzb_true_log = np.meshgrid(np.diff(xb_true), np.diff(yb_true_log), np.array([0]))
        xxb_true, yyb_true, zzb_true = np.meshgrid(np.diff(xb_true), np.diff(yb_true), np.array([0]))
    grid_area_2d_true_log = xxb_true_log * yyb_true_log
    grid_area_2d_true = xxb_true * yyb_true

    weights_values = weights_values / np.sum(weights_values)
    weights_true_values = weights_true_values / np.sum(weights_true_values)

    # Reshape the weights and grids
    cumulative_weights = np.sum(weights_values, axis=0)
    cumulative_weights_2d = mask_mass_arrays(cumulative_weights.reshape(reg_dim), mask_age, mask_metal, mask_alpha)
    cumulative_weights_true = np.sum(weights_true_values, axis=0)
    cumulative_weights_true_2d = mask_mass_arrays(cumulative_weights_true.reshape(reg_dim_true), mask_age_true, mask_metal_true, mask_alpha_true)

    logage_grid_2d = mask_mass_arrays(logage_grid.reshape(reg_dim), mask_age, mask_metal, mask_alpha)
    age_grid_2d = mask_mass_arrays(age_grid.reshape(reg_dim), mask_age, mask_metal, mask_alpha)
    metal_grid_2d = mask_mass_arrays(metal_grid.reshape(reg_dim), mask_age, mask_metal, mask_alpha)
    alpha_grid_2d = mask_mass_arrays(alpha_grid.reshape(reg_dim), mask_age, mask_metal, mask_alpha)

    logage_grid_true_2d = mask_mass_arrays(logage_grid_true.reshape(reg_dim_true), mask_age_true, mask_metal_true, mask_alpha_true)
    age_grid_true_2d = mask_mass_arrays(age_grid_true.reshape(reg_dim_true), mask_age_true, mask_metal_true, mask_alpha_true)
    metal_grid_true_2d = mask_mass_arrays(metal_grid_true.reshape(reg_dim_true), mask_age_true, mask_metal_true, mask_alpha_true)
    alpha_grid_true_2d = mask_mass_arrays(alpha_grid_true.reshape(reg_dim_true), mask_age_true, mask_metal_true, mask_alpha_true)

    xgrid = age_grid_2d[:, :, 0]
    ygrid = metal_grid_2d[:, :, 0]
    zgrid = alpha_grid_2d[:, :, 0]

    xgrid_true = age_grid_true_2d[:, :, 0]
    ygrid_true = metal_grid_true_2d[:, :, 0]
    zgrid_true = alpha_grid_true_2d[:, :, 0]

    mfd = np.log10(np.sum(cumulative_weights_2d, axis=-1) / grid_area_2d[:, :, 0])
    mfd_true = np.log10(np.sum(cumulative_weights_true_2d, axis=-1) / grid_area_2d_true[:, :, 0])
    mfd[mfd == -np.inf] = np.nan
    mfd_true[mfd_true == -np.inf] = np.nan

    results_table = Table.read(filename + '_table.fits')
    pixelsize = fits.open(filename + '_table.fits')[0].header['PIXSIZE']
    results_nuclear = results_table[
        (results_table['XBIN'] > -5) & (results_table['XBIN'] < 5) & (results_table['YBIN'] > -3) & (
                    results_table['YBIN'] < 1) & (results_table['BIN_ID'] >= 0)]
    results_bulge = results_table[
        (results_table['XBIN'] > -10) & (results_table['XBIN'] < 10) & (results_table['YBIN'] > 14) & (
                    results_table['YBIN'] < 25) & (results_table['BIN_ID'] >= 0)]
    results_thin = results_table[
        (results_table['XBIN'] > 30) & (results_table['XBIN'] < 120) & (results_table['YBIN'] > -5) & (
                    results_table['YBIN'] < 12) & (results_table['BIN_ID'] >= 0)]
    results_thick = results_table[
        (results_table['XBIN'] > 30) & (results_table['XBIN'] < 120) & (results_table['YBIN'] > 14) & (
                    results_table['YBIN'] < 25) & (results_table['BIN_ID'] >= 0)]
    results_innerthin = results_table[
        (results_table['XBIN'] > 30) & (results_table['XBIN'] < 65) & (results_table['YBIN'] > -5) & (
                    results_table['YBIN'] < 12) & (results_table['BIN_ID'] >= 0)]
    results_outerthin = results_table[
        (results_table['XBIN'] > 65) & (results_table['XBIN'] < 120) & (results_table['YBIN'] > -5) & (
                    results_table['YBIN'] < 12) & (results_table['BIN_ID'] >= 0)]
    results_innerthick = results_table[
        (results_table['XBIN'] > 30) & (results_table['XBIN'] < 65) & (results_table['YBIN'] > 14) & (
                    results_table['YBIN'] < 25) & (results_table['BIN_ID'] >= 0)]
    results_outerthick = results_table[
        (results_table['XBIN'] > 65) & (results_table['XBIN'] < 120) & (results_table['YBIN'] > 14) & (
                    results_table['YBIN'] < 25) & (results_table['BIN_ID'] >= 0)]

    binid_nuclear = np.unique(results_nuclear['BIN_ID'])
    binid_bulge = np.unique(results_bulge['BIN_ID'])
    binid_thin = np.unique(results_thin['BIN_ID'])
    binid_thick = np.unique(results_thick['BIN_ID'])
    binid_innerthin = np.unique(results_innerthin['BIN_ID'])
    binid_outerthin = np.unique(results_outerthin['BIN_ID'])
    binid_innerthick = np.unique(results_innerthick['BIN_ID'])
    binid_outerthick = np.unique(results_outerthick['BIN_ID'])
    shapex = np.unique(results_table['X']).shape[0]
    shapey = np.unique(results_table['Y']).shape[0]
    flux_reshape = np.array(results_table['FLUX']).reshape([shapey, shapex])

    # Mass Fraction components
    weights_nuclear = mask_mass_arrays(np.sum(weights_values[binid_nuclear, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
    weights_bulge = mask_mass_arrays(np.sum(weights_values[binid_bulge, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
    weights_thin = mask_mass_arrays(np.sum(weights_values[binid_thin, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
    weights_thick = mask_mass_arrays(np.sum(weights_values[binid_thick, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
    weights_innerthin = mask_mass_arrays(np.sum(weights_values[binid_innerthin, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
    weights_outerthin = mask_mass_arrays(np.sum(weights_values[binid_outerthin, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
    weights_innerthick = mask_mass_arrays(np.sum(weights_values[binid_innerthick, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
    weights_outerthick = mask_mass_arrays(np.sum(weights_values[binid_outerthick, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)

    weights_nuclear_true = mask_mass_arrays(np.sum(weights_true_values[binid_nuclear, :], axis=0).reshape(reg_dim_true), mask_age_true, mask_metal_true, mask_alpha_true)
    weights_bulge_true = mask_mass_arrays(np.sum(weights_true_values[binid_bulge, :], axis=0).reshape(reg_dim_true), mask_age_true, mask_metal_true, mask_alpha_true)
    weights_thin_true = mask_mass_arrays(np.sum(weights_true_values[binid_thin, :], axis=0).reshape(reg_dim_true), mask_age_true, mask_metal_true, mask_alpha_true)
    weights_thick_true = mask_mass_arrays(np.sum(weights_true_values[binid_thick, :], axis=0).reshape(reg_dim_true), mask_age_true, mask_metal_true, mask_alpha_true)
    weights_innerthin_true = mask_mass_arrays(np.sum(weights_true_values[binid_innerthin, :], axis=0).reshape(reg_dim_true), mask_age_true, mask_metal_true, mask_alpha_true)
    weights_outerthin_true = mask_mass_arrays(np.sum(weights_true_values[binid_outerthin, :], axis=0).reshape(reg_dim_true), mask_age_true, mask_metal_true, mask_alpha_true)
    weights_innerthick_true = mask_mass_arrays(np.sum(weights_true_values[binid_innerthick, :], axis=0).reshape(reg_dim_true), mask_age_true, mask_metal_true, mask_alpha_true)
    weights_outerthick_true = mask_mass_arrays(np.sum(weights_true_values[binid_outerthick, :], axis=0).reshape(reg_dim_true), mask_age_true, mask_metal_true, mask_alpha_true)

    weights_nuclear = weights_nuclear / np.sum(weights_nuclear)
    weights_bulge = weights_bulge / np.sum(weights_bulge)
    weights_thin = weights_thin / np.sum(weights_thin)
    weights_thick = weights_thick / np.sum(weights_thick)
    weights_innerthin = weights_innerthin / np.sum(weights_innerthin)
    weights_outerthin = weights_outerthin / np.sum(weights_outerthin)
    weights_innerthick = weights_innerthick / np.sum(weights_innerthick)
    weights_outerthick = weights_outerthick / np.sum(weights_outerthick)

    weights_nuclear_true = weights_nuclear_true / np.sum(weights_nuclear_true)
    weights_bulge_true = weights_bulge_true / np.sum(weights_bulge_true)
    weights_thin_true = weights_thin_true / np.sum(weights_thin_true)
    weights_thick_true = weights_thick_true / np.sum(weights_thick_true)
    weights_innerthin_true = weights_innerthin_true / np.sum(weights_innerthin_true)
    weights_outerthin_true = weights_outerthin_true / np.sum(weights_outerthin_true)
    weights_innerthick_true = weights_innerthick_true / np.sum(weights_innerthick_true)
    weights_outerthick_true = weights_outerthick_true / np.sum(weights_outerthick_true)


    mdf_nuclear = np.log10(np.sum(weights_nuclear, axis=-1) / grid_area_2d[:, :, 0])
    mdf_bulge = np.log10(np.sum(weights_bulge, axis=-1) / grid_area_2d[:, :, 0])
    mdf_thin = np.log10(np.sum(weights_thin, axis=-1) / grid_area_2d[:, :, 0])
    mdf_thick = np.log10(np.sum(weights_thick, axis=-1) / grid_area_2d[:, :, 0])
    mdf_innerthin = np.log10(np.sum(weights_innerthin, axis=-1) / grid_area_2d[:, :, 0])
    mdf_outerthin = np.log10(np.sum(weights_outerthin, axis=-1) / grid_area_2d[:, :, 0])
    mdf_innerthick = np.log10(np.sum(weights_innerthick, axis=-1) / grid_area_2d[:, :, 0])
    mdf_outerthick = np.log10(np.sum(weights_outerthick, axis=-1) / grid_area_2d[:, :, 0])

    mdf_nuclear_true = np.log10(np.sum(weights_nuclear_true, axis=-1) / grid_area_2d_true[:, :, 0])
    mdf_bulge_true = np.log10(np.sum(weights_bulge_true, axis=-1) / grid_area_2d_true[:, :, 0])
    mdf_thin_true = np.log10(np.sum(weights_thin_true, axis=-1) / grid_area_2d_true[:, :, 0])
    mdf_thick_true = np.log10(np.sum(weights_thick_true, axis=-1) / grid_area_2d_true[:, :, 0])
    mdf_innerthin_true = np.log10(np.sum(weights_innerthin_true, axis=-1) / grid_area_2d_true[:, :, 0])
    mdf_outerthin_true = np.log10(np.sum(weights_outerthin_true, axis=-1) / grid_area_2d_true[:, :, 0])
    mdf_innerthick_true = np.log10(np.sum(weights_innerthick_true, axis=-1) / grid_area_2d_true[:, :, 0])
    mdf_outerthick_true = np.log10(np.sum(weights_outerthick_true, axis=-1) / grid_area_2d_true[:, :, 0])

    mean_alpha_grids_nuclear = np.sum((weights_nuclear * alpha_grid_2d), axis=-1) / np.sum(weights_nuclear, axis=-1)
    mean_alpha_grids_bulge = np.sum((weights_bulge * alpha_grid_2d), axis=-1) / np.sum(weights_bulge, axis=-1)
    mean_alpha_grids_thin = np.sum((weights_thin * alpha_grid_2d), axis=-1) / np.sum(weights_thin, axis=-1)
    mean_alpha_grids_thick = np.sum((weights_thick * alpha_grid_2d), axis=-1) / np.sum(weights_thick, axis=-1)

    mean_alpha_grids_nuclear_true = np.sum((weights_nuclear_true * alpha_grid_true_2d), axis=-1) / np.sum(
        weights_nuclear_true, axis=-1)
    mean_alpha_grids_bulge_true = np.sum((weights_bulge_true * alpha_grid_true_2d), axis=-1) / np.sum(
        weights_bulge_true, axis=-1)
    mean_alpha_grids_thin_true = np.sum((weights_thin_true * alpha_grid_true_2d), axis=-1) / np.sum(
        weights_thin_true, axis=-1)
    mean_alpha_grids_thick_true = np.sum((weights_thick_true * alpha_grid_true_2d), axis=-1) / np.sum(
        weights_thick_true, axis=-1)



    # Mass Fraction components
    bin_edges = np.arange(0, 14.1, 0.5)
    bin_areas = bin_edges[:-1] + np.diff(bin_edges) / 2
    statistic_sfh_true = spectres(bin_areas, np.concatenate(([0.], age_grid_true_2d[:, 0, 0])),
                                  np.concatenate(([0.], np.sum(cumulative_weights_true_2d, axis=(1, 2)) / yyb_true[:, 0, 0])),
                                  fill=np.nan, verbose=False)
    statistic_sfh_nuclear_true = spectres(bin_areas,
                                          np.concatenate(([0.], age_grid_true_2d[:, 0, 0])),
                                          np.concatenate(([0.], np.sum(weights_nuclear_true, axis=(1, 2)) / yyb_true[:, 0, 0])),
                                          fill=np.nan, verbose=False)
    statistic_sfh_bulge_true = spectres(bin_areas, np.concatenate(([0.], age_grid_true_2d[:, 0, 0])),
                                        np.concatenate(([0.], np.sum(weights_bulge_true, axis=(1, 2)) / yyb_true[:, 0, 0])),
                                        fill=np.nan, verbose=False)
    statistic_sfh_thin_true = spectres(bin_areas, np.concatenate(([0.], age_grid_true_2d[:, 0, 0])),
                                       np.concatenate(([0.], np.sum(weights_thin_true, axis=(1, 2)) / yyb_true[:, 0, 0])),
                                       fill=np.nan, verbose=False)
    statistic_sfh_thick_true = spectres(bin_areas, np.concatenate(([0.], age_grid_true_2d[:, 0, 0])),
                                        np.concatenate(([0.], np.sum(weights_thick_true, axis=(1, 2)) / yyb_true[:, 0, 0])),
                                        fill=np.nan, verbose=False)
    statistic_sfh_innerthin_true = spectres(bin_areas,
                                            np.concatenate(([0.], age_grid_true_2d[:, 0, 0])),
                                            np.concatenate(([0.], np.sum(weights_innerthin_true, axis=(1, 2)) / yyb_true[:, 0, 0])),
                                            fill=np.nan, verbose=False)
    statistic_sfh_outerthin_true = spectres(bin_areas,
                                            np.concatenate(([0.], age_grid_true_2d[:, 0, 0])),
                                            np.concatenate(([0.], np.sum(weights_outerthin_true, axis=(1, 2)) / yyb_true[:, 0, 0])),
                                            fill=np.nan, verbose=False)
    statistic_sfh_innerthick_true = spectres(bin_areas,
                                             np.concatenate(([0.], age_grid_true_2d[:, 0, 0])),
                                             np.concatenate(([0.], np.sum(weights_innerthick_true, axis=(1, 2)) / yyb_true[:, 0, 0])),
                                             fill=np.nan, verbose=False)
    statistic_sfh_outerthick_true = spectres(bin_areas,
                                             np.concatenate(([0.], age_grid_true_2d[:, 0, 0])),
                                             np.concatenate(([0.], np.sum(weights_outerthick_true, axis=(1, 2)) / yyb_true[:, 0, 0])),
                                             fill=np.nan, verbose=False)


    statistic_sfh = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                             np.concatenate(([0.], np.sum(cumulative_weights_2d, axis=(1, 2)) / yyb[:, 0, 0])),
                             fill=np.nan, verbose=False)
    statistic_sfh_nuclear = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                     np.concatenate(([0.], np.sum(weights_nuclear, axis=(1, 2)) / yyb[:, 0, 0])),
                                     fill=np.nan, verbose=False)
    statistic_sfh_bulge = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                   np.concatenate(([0.], np.sum(weights_bulge, axis=(1, 2)) / yyb[:, 0, 0])),
                                   fill=np.nan, verbose=False)
    statistic_sfh_thin = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                  np.concatenate(([0.], np.sum(weights_thin, axis=(1, 2)) / yyb[:, 0, 0])),
                                  fill=np.nan, verbose=False)
    statistic_sfh_thick = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                   np.concatenate(([0.], np.sum(weights_thick, axis=(1, 2)) / yyb[:, 0, 0])),
                                   fill=np.nan, verbose=False)
    statistic_sfh_innerthin = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                       np.concatenate(([0.], np.sum(weights_innerthin, axis=(1, 2)) / yyb[:, 0, 0])),
                                       fill=np.nan, verbose=False)
    statistic_sfh_outerthin = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                       np.concatenate(([0.], np.sum(weights_outerthin, axis=(1, 2)) / yyb[:, 0, 0])),
                                       fill=np.nan, verbose=False)
    statistic_sfh_innerthick = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                        np.concatenate(([0.], np.sum(weights_innerthick, axis=(1, 2)) / yyb[:, 0, 0])),
                                        fill=np.nan, verbose=False)
    statistic_sfh_outerthick = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                        np.concatenate(([0.], np.sum(weights_outerthick, axis=(1, 2)) / yyb[:, 0, 0])),
                                        fill=np.nan, verbose=False)



    metal_true = np.sum(cumulative_weights_true_2d, axis=(0, 2)) / xxb_true[0, :, 0]
    metal_thin_true = np.sum(weights_thin_true, axis=(0, 2)) / xxb_true[0, :, 0]
    metal_thick_true = np.sum(weights_thick_true, axis=(0, 2)) / xxb_true[0, :, 0]
    metal_bulge_true = np.sum(weights_bulge_true, axis=(0, 2)) / xxb_true[0, :, 0]
    metal_nuclear_true = np.sum(weights_nuclear_true, axis=(0, 2)) / xxb_true[0, :, 0]

    metal = np.sum(cumulative_weights_2d, axis=(0, 2)) / xxb[0, :, 0]
    metal_thin = np.sum(weights_thin, axis=(0, 2)) / xxb[0, :, 0]
    metal_thick = np.sum(weights_thick, axis=(0, 2)) / xxb[0, :, 0]
    metal_bulge = np.sum(weights_bulge, axis=(0, 2)) / xxb[0, :, 0]
    metal_nuclear = np.sum(weights_nuclear, axis=(0, 2)) / xxb[0, :, 0]


    return cube_gist_run_path, filename, bin_edges, bin_areas, xgrid, ygrid, zgrid, xgrid_true, ygrid_true, zgrid_true, grid_area_2d, grid_area_2d_log, grid_area_2d_true, grid_area_2d_true_log, \
           age_grid_2d, age_grid_true_2d, metal_grid_2d, metal_grid_true_2d, alpha_grid_2d, alpha_grid_true_2d, xb, xb_true, xxb, xxb_log, xxb_true, xxb_true_log, yb, yb_log, yb_true, yb_true_log, yyb, yyb_log, yyb_true, yyb_true_log, \
           zb, zb_true, zzb, zzb_log, zzb_true, zzb_true_log, reg_dim, reg_dim_true, \
           cumulative_weights_2d, weights_values, weights_nuclear, weights_bulge, weights_thin, weights_thick, weights_innerthin, weights_outerthin, weights_innerthick, weights_outerthick, \
           cumulative_weights_true_2d, weights_true_values, weights_nuclear_true, weights_bulge_true, weights_thin_true, weights_thick_true, weights_innerthin_true, weights_outerthin_true, weights_innerthick_true, weights_outerthick_true, \
           results_table, pixelsize, flux_reshape, results_nuclear, results_bulge, results_thin, results_thick, results_innerthin, results_outerthin, results_innerthick, results_outerthick, \
           mfd, mdf_nuclear, mdf_bulge, mdf_thin, mdf_thick, mdf_innerthin, mdf_outerthin, mdf_innerthick, mdf_outerthick, \
           mfd_true, mdf_nuclear_true, mdf_bulge_true, mdf_thin_true, mdf_thick_true, mdf_innerthin_true, mdf_outerthin_true, mdf_innerthick_true, mdf_outerthick_true, \
           mean_alpha_grids_nuclear, mean_alpha_grids_bulge, mean_alpha_grids_thin, mean_alpha_grids_thick, \
           mean_alpha_grids_nuclear_true, mean_alpha_grids_bulge_true, mean_alpha_grids_thin_true, mean_alpha_grids_thick_true, \
           statistic_sfh, statistic_sfh_nuclear, statistic_sfh_bulge, statistic_sfh_thin, statistic_sfh_thick, \
           statistic_sfh_innerthin, statistic_sfh_outerthin, statistic_sfh_innerthick, statistic_sfh_outerthick, \
           statistic_sfh_true, statistic_sfh_nuclear_true, statistic_sfh_bulge_true, statistic_sfh_thin_true, statistic_sfh_thick_true, \
           statistic_sfh_innerthin_true, statistic_sfh_outerthin_true, statistic_sfh_innerthick_true, statistic_sfh_outerthick_true, \
           metal, metal_nuclear, metal_bulge, metal_thin, metal_thick, \
           metal_true, metal_nuclear_true, metal_bulge_true, metal_thin_true, metal_thick_true

# ----------------------------------------------------------------------------------------------------------------
def load_mass_fraction_single(cube_gist_path, cube_gist_run, truevalue):
    cube_gist_run_path = cube_gist_path + cube_gist_run + '/'
    filename = cube_gist_run_path + cube_gist_run

    if truevalue == False:
        if 'rmax' in cube_gist_run:
            weights = fits.open(filename + '_sfh-weights_rmax.fits')
        else:
            weights = fits.open(filename + '_sfh-weights.fits')
    else:
        if 'LW' in cube_gist_run:
            weights = fits.open(filename + '_sfh-weights_true_lw.fits')
        else:
            weights = fits.open(filename + '_sfh-weights_true_mw.fits')
    weights_values = np.zeros(weights[1].data.WEIGHTS.shape)
    for i in range(weights_values.shape[0]):
        weights_values[i, :] = weights[1].data.WEIGHTS[i, :] * weights[1].data.WEIGHTS_FACTOR[i]
    logage_grid = weights[2].data.LOGAGE
    age_grid = 10 ** (weights[2].data.LOGAGE)
    metal_grid = weights[2].data.METAL
    alpha_grid = weights[2].data.ALPHA
    reg_dim = np.array(
        [np.unique(logage_grid).shape, np.unique(metal_grid).shape, np.unique(alpha_grid).shape]).reshape(-1)

    weights_values = weights_values / np.sum(weights_values)

    # Reshape the weights and grids
    cumulative_weights = np.sum(weights_values, axis=0)
    cumulative_weights_2d = cumulative_weights.reshape(reg_dim)

    logage_grid_2d = logage_grid.reshape(reg_dim)
    age_grid_2d = age_grid.reshape(reg_dim)
    metal_grid_2d = metal_grid.reshape(reg_dim)
    alpha_grid_2d = alpha_grid.reshape(reg_dim)

    # Divide the mass-fraction by the area of each bin, to be per dex per Gyr
    x = np.unique(metal_grid)  # Grid centers
    y = np.unique(logage_grid)
    xb = (x[1:] + x[:-1]) / 2  # internal grid borders
    yb = (y[1:] + y[:-1]) / 2
    xb = np.hstack([1.5 * x[0] - x[1] / 2, xb, 1.5 * x[-1] - x[-2] / 2])  # 1st/last border
    yb_log = np.hstack([1.5 * y[0] - y[1] / 2, yb, 1.5 * y[-1] - y[-2] / 2])
    xxb_log, yyb_log, zzb_log = np.meshgrid(np.diff(xb), np.diff(yb_log), np.array([0.4, 0.4]))
    grid_area_2d_log = xxb_log * yyb_log
    yb = 10 ** (np.hstack([1.5 * y[0] - y[1] / 2, yb, 1.5 * y[-1] - y[-2] / 2]))
    yb[0] = 0
    xxb, yyb, zzb = np.meshgrid(np.diff(xb), np.diff(yb), np.array([0.4, 0.4]))
    grid_area_2d = xxb * yyb

    xgrid = age_grid_2d[:, :, 0]
    ygrid = metal_grid_2d[:, :, 0]


    mfd = np.log10(np.sum(cumulative_weights_2d, axis=-1) / grid_area_2d[:, :, 0])
    mfd[mfd == -np.inf] = np.nan

    results_table = Table.read(filename + '_table.fits')
    pixelsize = fits.open(filename + '_table.fits')[0].header['PIXSIZE']
    results_nuclear = results_table[
        (results_table['XBIN'] > -5) & (results_table['XBIN'] < 5) & (results_table['YBIN'] > -3) & (
                    results_table['YBIN'] < 1) & (results_table['BIN_ID'] >= 0)]
    results_bulge = results_table[
        (results_table['XBIN'] > -10) & (results_table['XBIN'] < 10) & (results_table['YBIN'] > 14) & (
                    results_table['YBIN'] < 25) & (results_table['BIN_ID'] >= 0)]
    results_thin = results_table[
        (results_table['XBIN'] > 30) & (results_table['XBIN'] < 120) & (results_table['YBIN'] > -5) & (
                    results_table['YBIN'] < 12) & (results_table['BIN_ID'] >= 0)]
    results_thick = results_table[
        (results_table['XBIN'] > 30) & (results_table['XBIN'] < 120) & (results_table['YBIN'] > 14) & (
                    results_table['YBIN'] < 25) & (results_table['BIN_ID'] >= 0)]
    results_innerthin = results_table[
        (results_table['XBIN'] > 30) & (results_table['XBIN'] < 65) & (results_table['YBIN'] > -5) & (
                    results_table['YBIN'] < 12) & (results_table['BIN_ID'] >= 0)]
    results_outerthin = results_table[
        (results_table['XBIN'] > 65) & (results_table['XBIN'] < 120) & (results_table['YBIN'] > -5) & (
                    results_table['YBIN'] < 12) & (results_table['BIN_ID'] >= 0)]
    results_innerthick = results_table[
        (results_table['XBIN'] > 30) & (results_table['XBIN'] < 65) & (results_table['YBIN'] > 14) & (
                    results_table['YBIN'] < 25) & (results_table['BIN_ID'] >= 0)]
    results_outerthick = results_table[
        (results_table['XBIN'] > 65) & (results_table['XBIN'] < 120) & (results_table['YBIN'] > 14) & (
                    results_table['YBIN'] < 25) & (results_table['BIN_ID'] >= 0)]

    binid_nuclear = np.unique(results_nuclear['BIN_ID'])
    binid_bulge = np.unique(results_bulge['BIN_ID'])
    binid_thin = np.unique(results_thin['BIN_ID'])
    binid_thick = np.unique(results_thick['BIN_ID'])
    binid_innerthin = np.unique(results_innerthin['BIN_ID'])
    binid_outerthin = np.unique(results_outerthin['BIN_ID'])
    binid_innerthick = np.unique(results_innerthick['BIN_ID'])
    binid_outerthick = np.unique(results_outerthick['BIN_ID'])
    shapex = np.unique(results_table['X']).shape[0]
    shapey = np.unique(results_table['Y']).shape[0]
    flux_reshape = np.array(results_table['FLUX']).reshape([shapey, shapex])

    # Mass Fraction components
    weights_nuclear = np.sum(weights_values[binid_nuclear, :], axis=0).reshape(reg_dim)
    weights_bulge = np.sum(weights_values[binid_bulge, :], axis=0).reshape(reg_dim)
    weights_thin = np.sum(weights_values[binid_thin, :], axis=0).reshape(reg_dim)
    weights_thick = np.sum(weights_values[binid_thick, :], axis=0).reshape(reg_dim)
    weights_innerthin = np.sum(weights_values[binid_innerthin, :], axis=0).reshape(reg_dim)
    weights_outerthin = np.sum(weights_values[binid_outerthin, :], axis=0).reshape(reg_dim)
    weights_innerthick = np.sum(weights_values[binid_innerthick, :], axis=0).reshape(reg_dim)
    weights_outerthick = np.sum(weights_values[binid_outerthick, :], axis=0).reshape(reg_dim)

    weights_nuclear = weights_nuclear / np.sum(weights_nuclear)
    weights_bulge = weights_bulge / np.sum(weights_bulge)
    weights_thin = weights_thin / np.sum(weights_thin)
    weights_thick = weights_thick / np.sum(weights_thick)
    weights_innerthin = weights_innerthin / np.sum(weights_innerthin)
    weights_outerthin = weights_outerthin / np.sum(weights_outerthin)
    weights_innerthick = weights_innerthick / np.sum(weights_innerthick)
    weights_outerthick = weights_outerthick / np.sum(weights_outerthick)

    mdf_nuclear = np.log10(np.sum(weights_nuclear, axis=-1) / grid_area_2d[:, :, 0])
    mdf_bulge = np.log10(np.sum(weights_bulge, axis=-1) / grid_area_2d[:, :, 0])
    mdf_thin = np.log10(np.sum(weights_thin, axis=-1) / grid_area_2d[:, :, 0])
    mdf_thick = np.log10(np.sum(weights_thick, axis=-1) / grid_area_2d[:, :, 0])
    mdf_innerthin = np.log10(np.sum(weights_innerthin, axis=-1) / grid_area_2d[:, :, 0])
    mdf_outerthin = np.log10(np.sum(weights_outerthin, axis=-1) / grid_area_2d[:, :, 0])
    mdf_innerthick = np.log10(np.sum(weights_innerthick, axis=-1) / grid_area_2d[:, :, 0])
    mdf_outerthick = np.log10(np.sum(weights_outerthick, axis=-1) / grid_area_2d[:, :, 0])

    mean_alpha_grids_nuclear = np.sum((weights_nuclear * alpha_grid_2d), axis=-1) / np.sum(weights_nuclear, axis=-1)
    mean_alpha_grids_bulge = np.sum((weights_bulge * alpha_grid_2d), axis=-1) / np.sum(weights_bulge, axis=-1)
    mean_alpha_grids_thin = np.sum((weights_thin * alpha_grid_2d), axis=-1) / np.sum(weights_thin, axis=-1)
    mean_alpha_grids_thick = np.sum((weights_thick * alpha_grid_2d), axis=-1) / np.sum(weights_thick, axis=-1)


    # Mass Fraction components
    bin_edges = np.arange(0, 14.1, 0.5)
    bin_areas = bin_edges[:-1] + np.diff(bin_edges) / 2

    statistic_sfh = spectres(bin_areas, np.concatenate(([0.], age_grid.reshape(reg_dim)[:, 0, 0])),
                            np.concatenate(([0.], np.sum(cumulative_weights_2d, axis=(1, 2)) / yyb[:, 0, 0])),
                            fill=np.nan, verbose=False)
    statistic_sfh_nuclear = spectres(bin_areas, np.concatenate(([0.], age_grid.reshape(reg_dim)[:, 0, 0])),
                                    np.concatenate(([0.], np.sum(weights_nuclear, axis=(1, 2)) / yyb[:, 0, 0])),
                                    fill=np.nan, verbose=False)
    statistic_sfh_bulge = spectres(bin_areas, np.concatenate(([0.], age_grid.reshape(reg_dim)[:, 0, 0])),
                                np.concatenate(([0.], np.sum(weights_bulge, axis=(1, 2)) / yyb[:, 0, 0])),
                                fill=np.nan, verbose=False)
    statistic_sfh_thin = spectres(bin_areas, np.concatenate(([0.], age_grid.reshape(reg_dim)[:, 0, 0])),
                                np.concatenate(([0.], np.sum(weights_thin, axis=(1, 2)) / yyb[:, 0, 0])),
                                fill=np.nan, verbose=False)
    statistic_sfh_thick = spectres(bin_areas, np.concatenate(([0.], age_grid.reshape(reg_dim)[:, 0, 0])),
                                np.concatenate(([0.], np.sum(weights_thick, axis=(1, 2)) / yyb[:, 0, 0])),
                                fill=np.nan, verbose=False)
    statistic_sfh_innerthin = spectres(bin_areas, np.concatenate(([0.], age_grid.reshape(reg_dim)[:, 0, 0])),
                                    np.concatenate(([0.], np.sum(weights_innerthin, axis=(1, 2)) / yyb[:, 0, 0])),
                                    fill=np.nan, verbose=False)
    statistic_sfh_outerthin = spectres(bin_areas, np.concatenate(([0.], age_grid.reshape(reg_dim)[:, 0, 0])),
                                    np.concatenate(([0.], np.sum(weights_outerthin, axis=(1, 2)) / yyb[:, 0, 0])),
                                    fill=np.nan, verbose=False)
    statistic_sfh_innerthick = spectres(bin_areas, np.concatenate(([0.], age_grid.reshape(reg_dim)[:, 0, 0])),
                                        np.concatenate(([0.], np.sum(weights_innerthick, axis=(1, 2)) / yyb[:, 0, 0])),
                                        fill=np.nan, verbose=False)
    statistic_sfh_outerthick = spectres(bin_areas, np.concatenate(([0.], age_grid.reshape(reg_dim)[:, 0, 0])),
                                        np.concatenate(([0.], np.sum(weights_outerthick, axis=(1, 2)) / yyb[:, 0, 0])),
                                        fill=np.nan, verbose=False)



    metal = np.sum(cumulative_weights_2d, axis=(0, 2)) / xxb[0, :, 0]
    metal_thin = np.sum(weights_thin, axis=(0, 2)) / xxb[0, :, 0]
    metal_thick = np.sum(weights_thick, axis=(0, 2)) / xxb[0, :, 0]
    metal_bulge = np.sum(weights_bulge, axis=(0, 2)) / xxb[0, :, 0]
    metal_nuclear = np.sum(weights_nuclear, axis=(0, 2)) / xxb[0, :, 0]


    return cube_gist_run_path, filename, bin_edges, bin_areas, xgrid, ygrid, grid_area_2d, grid_area_2d_log, \
           age_grid_2d, metal_grid_2d, xb, xxb, xxb_log, yb, yb_log, yyb, yyb_log, reg_dim, \
           cumulative_weights_2d, weights_values, weights_nuclear, weights_bulge, weights_thin, weights_thick, weights_innerthin, weights_outerthin, weights_innerthick, weights_outerthick, \
           results_table, pixelsize, flux_reshape, results_nuclear, results_bulge, results_thin, results_thick, results_innerthin, results_outerthin, results_innerthick, results_outerthick, \
           mfd, mdf_nuclear, mdf_bulge, mdf_thin, mdf_thick, mdf_innerthin, mdf_outerthin, mdf_innerthick, mdf_outerthick, \
           mean_alpha_grids_nuclear, mean_alpha_grids_bulge, mean_alpha_grids_thin, mean_alpha_grids_thick, \
           statistic_sfh, statistic_sfh_nuclear, statistic_sfh_bulge, statistic_sfh_thin, statistic_sfh_thick, \
           statistic_sfh_innerthin, statistic_sfh_outerthin, statistic_sfh_innerthick, statistic_sfh_outerthick, \
           metal, metal_nuclear, metal_bulge, metal_thin, metal_thick



# ----------------------------------------------------------------------------------------------------------------
def plot_weights_2d_cases(weights_list, age_grid_list, metal_grid_list, label_list, title_list,
                          vmin, vmax, axes_pad=(0.35, 0.3), cbar_mode='single', cbar_location='top', cbar_pad=0.3,
                          log_age=False, cb_label=r'light fraction ($\rm Gyr^{-1} dex^{-1}$)', **kwargs):

    ncols = len(weights_list)
    nrows = len(weights_list[0])

    if log_age == True:
        for i in range(len(age_grid_list)):
            age_grid_list[i] = np.log10(age_grid_list[i]) + 9
        age_label = "log Age (yr)"
    else:
        age_label = 'Age (Gyr)'

    fig = plt.figure(figsize=[ncols*4, nrows*2.5])
    grid = AxesGrid(fig, rect=111, nrows_ncols=(nrows, ncols), axes_pad=axes_pad, share_all=True, aspect=False,
                    label_mode="all", cbar_location=cbar_location, cbar_mode=cbar_mode, cbar_pad=cbar_pad, cbar_size=0.1)
    for row_i in range(nrows):
        for col_i in range(ncols):
            if row_i == 0:
                title = title_list[col_i]
            else:
                title = None

            pc = plot_weights_2d(grid[col_i + row_i * ncols], age_grid_list[col_i], metal_grid_list[col_i],
                                                         weights_list[col_i][row_i], title=title, threshold=vmin,
                                                         vmin=vmin, vmax=vmax, plot_xlabel=False, plot_ylabel=False,
                                                         colorbar=False, **kwargs)
            t = grid[col_i + row_i * ncols].text(0.03, 0.05, label_list[row_i], ha="left", va="bottom", rotation=0, size=12,
                                                 transform=grid[col_i + row_i * ncols].transAxes,
                                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2))
            if cbar_mode == 'each':
                cb = grid.cbar_axes[col_i + row_i * ncols].colorbar(pc)
                cb.set_label(cb_label, fontsize=10)

    for row_i in range(nrows):
        grid[row_i * ncols].set_ylabel('[M/H]')
    for col_i in range(ncols):
        grid[col_i + (nrows - 1) * ncols].set_xlabel(age_label)

    grid[0].set_ylim(np.min(np.concatenate([x.reshape(-1) for x in metal_grid_list])), np.max(np.concatenate([x.reshape(-1) for x in metal_grid_list])))
    grid[0].set_xlim(np.min(np.concatenate([x.reshape(-1) for x in age_grid_list])), np.max(np.concatenate([x.reshape(-1) for x in age_grid_list])))

    if cbar_mode == 'single':
        cb = grid.cbar_axes[0].colorbar(pc)
        cb.set_label(cb_label, fontsize=10)

    return fig

# ----------------------------------------------------------------------------------------------------------------
def plot_weights_1d_cases(weights_list, age_grid_list, metal_grid_list, age_gridsize_list, metal_gridsize_list, age_gridborder_list, metal_gridborder_list,
                          plot_type_list, perbinsize_list, logmass_list, show_corr_list, color_list, label_list, text_list, direction, fraction_type, **kwargs):
    npanels = len(weights_list[0])
    nweights = len(weights_list)

    if direction == 'row':
        axflattype = 'C'
        ncols = npanels
        nrows = len(plot_type_list)
    elif direction == 'col':
        axflattype = 'F'
        ncols = len(plot_type_list)
        nrows = npanels

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.5*ncols, 2*nrows))
    axesl = axes.flatten(axflattype)
    axes_idx = 0
    for plot_type_idx, plot_type in enumerate(plot_type_list):
        for panel_i in range(npanels):

            # if axes_idx % ncols == 0:
            #     plot_ylabel = True
            # else:
            #     plot_ylabel = False

            if plot_type == 'age_stairs':
                if perbinsize_list[plot_type_idx] != False:
                    panel_weights_list = [np.sum(weights_list[weight_i][panel_i], axis=(1, 2)) / age_gridsize_list[weight_i] for weight_i in range(nweights)]
                else:
                    panel_weights_list = [np.sum(weights_list[weight_i][panel_i], axis=(1, 2)) for weight_i in range(nweights)]
                plot_sfh_stairs_list(axesl[axes_idx], panel_weights_list,
                                    age_grid_list.copy(), age_gridborder_list.copy(), label_list, colors=color_list,
                                    text=text_list[panel_i], plot_xlabel=True, plot_ylabel=True, perbinsize=perbinsize_list[plot_type_idx],
                                                             plot_legend=False, linear_age=True, logmass=logmass_list[plot_type_idx], fraction_type=fraction_type)
            elif plot_type == 'logage_stairs':
                if perbinsize_list[plot_type_idx] != False:
                    panel_weights_list = [np.sum(weights_list[weight_i][panel_i], axis=(1, 2)) / age_gridsize_list[weight_i] for weight_i in range(nweights)]
                else:
                    panel_weights_list = [np.sum(weights_list[weight_i][panel_i], axis=(1, 2)) for weight_i in range(nweights)]
                plot_sfh_stairs_list(axesl[axes_idx], panel_weights_list,
                                    age_grid_list.copy(), age_gridborder_list.copy(), label_list, colors=color_list,
                                    text=text_list[panel_i], plot_xlabel=True, plot_ylabel=True, perbinsize=perbinsize_list[plot_type_idx],
                                                             plot_legend=False, linear_age=False, logmass=logmass_list[plot_type_idx], fraction_type=fraction_type)

            elif plot_type == 'metal_stairs':
                if perbinsize_list[plot_type_idx] != False:
                    panel_weights_list = [np.sum(weights_list[weight_i][panel_i], axis=(0, 2)) / metal_gridsize_list[weight_i] for weight_i in range(nweights)]
                else:
                    panel_weights_list = [np.sum(weights_list[weight_i][panel_i], axis=(0, 2)) for weight_i in range(nweights)]
                plot_metal_stairs_list(axesl[axes_idx], panel_weights_list,
                                    metal_grid_list, metal_gridborder_list, label_list, colors=color_list,
                                    text=text_list[panel_i], plot_xlabel=True, plot_ylabel=True, perbinsize=perbinsize_list[plot_type_idx],
                                                               plot_legend=False, logmass=logmass_list[plot_type_idx], fraction_type=fraction_type)
            elif plot_type == 'age_cum':
                panel_weights_list = [np.sum(weights_list[weight_i][panel_i], axis=(1, 2)) for weight_i in range(nweights)]
                plot_cumulative_sfh_list(axesl[axes_idx], panel_weights_list,
                                        age_grid_list.copy(), age_gridborder_list.copy(), label_list, colors=color_list,
                                        text=None, plot_xlabel=True, plot_ylabel=True, linear_age=True, plot_legend=False)
            elif plot_type == 'logage_cum':
                panel_weights_list = [np.sum(weights_list[weight_i][panel_i], axis=(1, 2)) for weight_i in range(nweights)]
                plot_cumulative_sfh_list(axesl[axes_idx], panel_weights_list,
                                        age_grid_list.copy(), age_gridborder_list.copy(), label_list, colors=color_list,
                                        text=None, plot_xlabel=True, plot_ylabel=True, linear_age=False, plot_legend=False)

            elif plot_type == 'metal_cum':
                panel_weights_list = [np.sum(weights_list[weight_i][panel_i], axis=(0, 2)) for weight_i in range(nweights)]
                plot_cumulative_metal_list(axesl[axes_idx], panel_weights_list,
                                        metal_grid_list, metal_gridborder_list, label_list, colors=color_list,
                                        text=None, plot_xlabel=True, plot_ylabel=True, plot_legend=False)

            if nweights == 2:
                if show_corr_list[plot_type_idx] == True:
                    if 'age' in plot_type:
                        axesl[axes_idx].text(0.05, 0.83, "Corr=%.3f" % cal_pearsonr_corr(age_grid_list[0], panel_weights_list[0], age_grid_list[1], panel_weights_list[1])[0],
                                             ha="left", va="top", rotation=0, size=10, transform=axesl[axes_idx].transAxes)
                    elif 'metal' in plot_type:
                        axesl[axes_idx].text(0.05, 0.83, "Corr=%.3f" % cal_pearsonr_corr(metal_grid_list[0], panel_weights_list[0], metal_grid_list[1], panel_weights_list[1])[0],
                                             ha="left", va="top", rotation=0, size=10, transform=axesl[axes_idx].transAxes)
            axes_idx += 1
    return fig
