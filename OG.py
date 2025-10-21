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





def plot_age_metal_grid_2d(distri1, distri2, title1, title2, age_grid_2d, metal_grid_2d, linear_age=False, threshold=0, **kwargs):
    '''
    plot two series values along the age/metallicity grid with different title
    :param distri1:
    :param distri2:
    :param title1:
    :param title2:
    :param age_bins:
    :param metal_bins:
    :param age_grid_2d:
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


def plot_mh_alpha_rz_hist(weights_values, weights_true_values, rbin, zbin, d, reg_dim, reg_dim_true, results_table,
                          metal_grid, metal_grid_true, fraction_type='mass fraction'):

    fig, axes = plt.subplots(len(zbin) - 1, len(rbin) - 1, figsize=(2 * (len(rbin) - 1), 1.8 * (len(zbin) - 1)), sharey=True)


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

        axes[-j_list[k] - 1, i_list[k]].plot(metal_grid, distri_bin_list[k][:, 0] / np.sum(distri_bin_list[k]), c='deepskyblue',
                             label=r'[$\alpha$/Fe]=0.0 (PPXF)')
        axes[-j_list[k] - 1, i_list[k]].plot(metal_grid, distri_bin_list[k][:, 1] / np.sum(distri_bin_list[k]), c='tomato',
                             label=r'[$\alpha$/Fe]=0.4 (PPXF)')
        axes[-j_list[k] - 1, i_list[k]].plot(metal_grid_true, distri_bin_true_list[k][:, 0] / np.sum(distri_bin_true_list[k]), ':',
                             c='deepskyblue', label=r'[$\alpha$/Fe]=0.0 (True)')
        axes[-j_list[k] - 1, i_list[k]].plot(metal_grid_true, distri_bin_true_list[k][:, 1] / np.sum(distri_bin_true_list[k]), ':',
                             c='tomato', label=r'[$\alpha$/Fe]=0.4 (True)')
        if len(distri_bin_list[k][:, 0]) == len(distri_bin_true_list[k][:, 0]):
            axes[-j_list[k] - 1, i_list[k]].text(0.02, 0.30, "Corr=%.3f" % pearsonr(distri_bin_list[k][:, 0] / np.sum(distri_bin_list[k]),
                                                                    distri_bin_true_list[k][:, 0] / np.sum(distri_bin_true_list[k]))[0],
                                 ha='left', va='bottom', transform=axes[-j_list[k] - 1, i_list[k]].transAxes, c='deepskyblue', fontsize=11)
            axes[-j_list[k] - 1, i_list[k]].text(0.02, 0.18, "Corr=%.3f" % pearsonr(distri_bin_list[k][:, 1] / np.sum(distri_bin_list[k]),
                                                                    distri_bin_true_list[k][:, 1] / np.sum(distri_bin_true_list[k]))[0],
                                 ha='left', va='bottom', transform=axes[-j_list[k] - 1, i_list[k]].transAxes, c='tomato', fontsize=11)

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel('[M/H]', fontsize=15)
    plt.ylabel(fraction_type, fontsize=15)

    fig.subplots_adjust(left=0.042, right=0.995, top=0.92, bottom=0.09, hspace=0.1, wspace=0.0)
    axes[0, 0].legend(bbox_to_anchor=(0, 1.02, 1, 0.102), loc='lower left', ncol=4, fontsize=12)

    return fig

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
            if len(np.sum(distri_bin_list[k], axis=1)) == len(np.sum(distri_bin_true_list[k], axis=1)):
                axes[-j_list[k] - 1, i_list[k]].text(0.02, 0.30, "Corr=%.3f" % pearsonr(np.sum(distri_bin_list[k], axis=1) / np.sum(distri_bin_list[k]),
                                                                        np.sum(distri_bin_true_list[k], axis=1) / np.sum(distri_bin_true_list[k]))[0],
                                     ha='left', va='bottom', transform=axes[-j_list[k] - 1, i_list[k]].transAxes, c='black', fontsize=11)
        elif mode == 'alpha':
            axes[-j_list[k] - 1, i_list[k]].plot(alpha_grid, np.sum(distri_bin_list[k], axis=0) / np.sum(distri_bin_list[k]), c='black',
                                 label=r'PPXF')
            axes[-j_list[k] - 1, i_list[k]].plot(alpha_grid_true, np.sum(distri_bin_true_list[k], axis=0) / np.sum(distri_bin_true_list[k]), ls='dashed',
                                 c='black', label=r'True')
            if len(np.sum(distri_bin_list[k], axis=0)) == len(np.sum(distri_bin_true_list[k], axis=0)):
                axes[-j_list[k] - 1, i_list[k]].text(0.98, 0.30, "Corr=%.3f" % pearsonr(np.sum(distri_bin_list[k], axis=0) / np.sum(distri_bin_list[k]),
                                                                        np.sum(distri_bin_true_list[k], axis=0) / np.sum(distri_bin_true_list[k]))[0],
                                     ha='right', va='bottom', transform=axes[-j_list[k] - 1, i_list[k]].transAxes, c='black', fontsize=11)


    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(fraction_type, fontsize=15)

    fig.subplots_adjust(left=0.042, right=0.995, top=0.92, bottom=0.09, hspace=0.1, wspace=0.0)
    axes[0, 0].legend(bbox_to_anchor=(0, 1.02, 1, 0.102), loc='lower left', ncol=4, fontsize=12)

    return fig

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



def mask_mass_arrays(mass_array, mask_age, mask_metal, mask_alpha):
    mass_array = mass_array[mask_age, :, :]
    mass_array = mass_array[:, mask_metal, :]
    mass_array = mass_array[:, :, mask_alpha]
    return mass_array


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
