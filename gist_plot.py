# -*- coding: utf-8 -*-

"""
@Time        : 8/1/2023
@Author      : purmortal
@File        : gist_plot
@Description : 
"""


from   astropy.io import fits
import numpy as np

from matplotlib.tri import Triangulation, TriAnalyzer

import matplotlib.pyplot       as     plt
from   mpl_toolkits.axes_grid1 import AxesGrid
from   matplotlib.ticker       import MultipleLocator, FuncFormatter

from plotbin.sauron_colormap import register_sauron_colormap
register_sauron_colormap()


def TicklabelFormatter(x, pos):
    return ("${}$".format(int(x)).replace("-", r"\textendash"))


# def setup_plot(usetex=False):
#     # fontsize = 14
#     # dpi = 300
#
#     # plt.rc('font', family='serif')
#     plt.rc('text', usetex=usetex)
#
#     # plt.rcParams['axes.labelsize'] = fontsize
#     # plt.rcParams['legend.fontsize'] = fontsize - 3
#     # plt.rcParams['legend.fancybox'] = True
#     # plt.rcParams['font.size'] = fontsize
#
#     # plt.rcParams['xtick.major.pad'] = '7'
#     # plt.rcParams['ytick.major.pad'] = '7'
#
#     # plt.rcParams['savefig.bbox'] = 'tight'
#     # plt.rcParams['savefig.dpi'] = dpi
#     # plt.rcParams['savefig.pad_inches'] = 0.02
#
#     # plt.rcParams['text.latex.preamble'] = r'\boldmath'


def read_results_kin(filename, suffix):
    # Read bintable
    table_hdu = fits.open(filename + '_table.fits')
    idx_inside = np.where(table_hdu[1].data.BIN_ID >= 0)[0]
    X = np.array(table_hdu[1].data.X[idx_inside]) * -1
    Y = np.array(table_hdu[1].data.Y[idx_inside])
    FLUX = np.array(table_hdu[1].data.FLUX[idx_inside])
    binNum_long = np.array(table_hdu[1].data.BIN_ID[idx_inside])
    ubins = np.unique(binNum_long)
    pixelsize = table_hdu[0].header['PIXSIZE']

    result_l = []

    for suffix_i in suffix:

        # Read results
        # print(filename + '_kin' + suffix_i + '.fits')
        hdu = fits.open(filename + '_kin' + suffix_i + '.fits')
        result = np.zeros((len(ubins), 4))
        result[:, 0] = np.array(hdu[1].data.V)
        result[:, 1] = np.array(hdu[1].data.SIGMA)
        if hasattr(hdu[1].data, 'H3'): result[:, 2] = np.array(hdu[1].data.H3)
        if hasattr(hdu[1].data, 'H4'): result[:, 3] = np.array(hdu[1].data.H4)

        # Convert results to long version
        result_long = np.zeros((len(binNum_long), result.shape[1]));
        result_long[:, :] = np.nan
        for i in range(len(ubins)):
            idx = np.where(ubins[i] == np.abs(binNum_long))[0]
            result_long[idx, :] = result[i, :]
        result = result_long
        result_l.append(result)

    return result_l


def plot_kin(filename, rootname, result_l, plot_contour=False, contour_offset_saved=0.2, vminmax=None, use_vminmax=None,
             residual=None, cmap_l=None):
    

    if np.all(vminmax != None):
        if len(vminmax.shape) == 2:
            vminmax = np.tile(vminmax, (len(result_l), 1, 1))
        elif len(vminmax.shape) == 1:
            vminmax = np.tile(vminmax, (len(result_l), result_l[0].shape[1], 1))
    if np.all(use_vminmax != None):
        if type(use_vminmax) == bool and len(result_l) > 1:
            use_vminmax = [use_vminmax] * len(result_l)
    if np.all(residual == None):
        residual = [False] * len(result_l)
    if np.all(cmap_l == None):
        cmap_l = ['sauron'] * len(result_l)

    labellist = ['V', 'SIGMA', 'H3', 'H4']

    # Read bintable
    table_hdu = fits.open(filename + '_table.fits')
    idx_inside = np.where(table_hdu[1].data.BIN_ID >= 0)[0]
    X = np.array(table_hdu[1].data.X[idx_inside]) * -1
    Y = np.array(table_hdu[1].data.Y[idx_inside])
    FLUX = np.array(table_hdu[1].data.FLUX[idx_inside])
    binNum_long = np.array(table_hdu[1].data.BIN_ID[idx_inside])
    ubins = np.unique(np.abs(np.array(table_hdu[1].data.BIN_ID)))
    pixelsize = table_hdu[0].header['PIXSIZE']

    # Check spatial coordinates
    if len(np.where(np.logical_or(X == 0.0, np.isnan(X) == True))[0]) == len(X):
        print(
            'All X-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!')
    if len(np.where(np.logical_or(Y == 0.0, np.isnan(Y) == True))[0]) == len(Y):
        print(
            'All Y-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!\n')

    # Setup main figure
    nrows = len(result_l)
    nplots = len(labellist)
    fig = plt.figure(figsize=(4.5 * nplots, 4 * nrows))
    grid = AxesGrid(fig, 111, nrows_ncols=(nrows, nplots), axes_pad=0.0, share_all=True, \
                    label_mode="L", cbar_location="right", cbar_mode="single", cbar_size='2%')
    contour_offset = contour_offset_saved

    for row_i in range(nrows):
        result = result_l[row_i]

        for iterate in range(0, nplots):
            subplot_index = row_i * nplots + iterate

            # Prepare main plot
            val = result[:, iterate]

            if np.all(vminmax != None) and use_vminmax[row_i] == True:
                vmin = vminmax[row_i, iterate, 0]
                vmax = vminmax[row_i, iterate, 1]
            elif residual[row_i] == False:
                # Determine vmin/vmax automatically, if called from within the pipeline
                vmin = np.nanpercentile(val, 0.5)
                vmax = np.nanpercentile(val, 99.5)
            elif residual[row_i] == True:
                vmin_0 = np.nanpercentile(val, 0.5)
                vmax_0 = np.nanpercentile(val, 99.5)
                vmin = -np.max(np.abs([vmin_0, vmax_0]))
                vmax = np.max(np.abs([vmin_0, vmax_0]))

            # Create image in pixels
            xmin = np.nanmin(X) - 5;
            xmax = np.nanmax(X) + 5
            ymin = np.nanmin(Y) - 5;
            ymax = np.nanmax(Y) + 5
            npixels_x = int(np.round((xmax - xmin) / pixelsize) + 1)
            npixels_y = int(np.round((ymax - ymin) / pixelsize) + 1)
            i = np.array(np.round((X - xmin) / pixelsize), dtype=int)
            j = np.array(np.round((Y - ymin) / pixelsize), dtype=int)
            image = np.full((npixels_x, npixels_y), np.nan)
            image[i, j] = val

            # Plot map and colorbar
            image = grid[subplot_index].imshow(np.rot90(image), cmap=cmap_l[row_i], interpolation=None, vmin=vmin,
                                               vmax=vmax, \
                                               extent=[-(xmin - pixelsize / 2), -(xmax + pixelsize / 2),
                                                       ymin - pixelsize / 2, ymax + pixelsize / 2])
            grid.cbar_axes[subplot_index].colorbar(image)

            if plot_contour:
                # Plot contours
                XY_Triangulation = Triangulation(X - pixelsize / 2,
                                                 Y - pixelsize / 2)  # Create a mesh from a Delaunay triangulation
                XY_Triangulation.set_mask(TriAnalyzer(XY_Triangulation).get_flat_tri_mask(
                    0.01))  # Remove bad triangles at the border of the field-of-view
                levels = np.arange(np.min(np.log10(FLUX)) + contour_offset, np.max(np.log10(FLUX)), 0.2)
                grid[subplot_index].tricontour(XY_Triangulation, np.log10(FLUX), levels=levels, linewidths=1,
                                               colors='k')

            # Label vmin and vmax
            if iterate in [0, 1]:
                grid[subplot_index].text(0.02, 0.85, r'[%.0f, %.0f]' % (vmin, vmax),
                                         horizontalalignment='left', verticalalignment='top',
                                         transform=grid[subplot_index].transAxes, fontsize=14)
            elif iterate in [2, 3]:
                grid[subplot_index].text(0.02, 0.85, r'[%.2f, %.2f]' % (vmin, vmax),
                                         horizontalalignment='left', verticalalignment='top',
                                         transform=grid[subplot_index].transAxes, fontsize=14)

        # Set V, SIGMA, H3, H4 labels
        grid[row_i * nplots].text(0.02, 0.975, r'$V_{\mathrm{LOS}}$ [km/s]', horizontalalignment='left',
                                  verticalalignment='top', transform=grid[row_i * nplots].transAxes, fontsize=16)
        grid[row_i * nplots].text(0.975, 0.965, rootname[row_i], horizontalalignment='right', fontweight='semibold',
                                  verticalalignment='top', transform=grid[row_i * nplots].transAxes, fontsize=16)
        grid[row_i * nplots + 1].text(0.02, 0.975, r'$\sigma$ [km/s]',
                                      horizontalalignment='left', verticalalignment='top',
                                      transform=grid[row_i * nplots + 1].transAxes, fontsize=16)
        grid[row_i * nplots + 2].text(0.02, 0.975, r'$h_3$', horizontalalignment='left', verticalalignment='top',
                                      transform=grid[row_i * nplots + 2].transAxes, fontsize=16)
        grid[row_i * nplots + 3].text(0.02, 0.975, r'$h_4$', horizontalalignment='left', verticalalignment='top',
                                      transform=grid[row_i * nplots + 3].transAxes, fontsize=16)

        # Invert x-axis
        grid[0].invert_xaxis()

        # Set xlabel and ylabel
        grid[row_i * nplots].set_ylabel(r'$y$ [arcsec]', fontsize=15)
        grid[row_i * nplots].yaxis.set_major_locator(MultipleLocator(10))  # Major tick every 10 units
        grid[row_i * nplots].yaxis.set_minor_locator(MultipleLocator(2))   # Minor tick every 1 units


        # Set tick frequency and parameters
        for plot_i in range(nplots):
            grid[row_i * nplots + plot_i].set_xlabel(r'$x$ [arcsec]', fontsize=15)
            grid[row_i * nplots + plot_i].xaxis.set_major_locator(MultipleLocator(20))
            grid[row_i * nplots + plot_i].xaxis.set_minor_locator(MultipleLocator(2))
            grid[row_i * nplots + plot_i].tick_params(direction="in", which='both', bottom=True, top=True, left=True, right=True, labelsize=13) # Ticks inside of plot


        # Remove ticks and labels from colorbar
        for cax in grid.cbar_axes:
            cax.toggle_label(False)
            cax.yaxis.set_ticks([])

    return fig





def plot_dkin_sig(filename):

    table_kin = fits.open(filename + '_kin.fits')[1].data
    try:
        table_kin_true = fits.open(filename + '_kin_true_lw.fits')[1].data
    except:
        table_kin_true = fits.open(filename + '_kin_true.fits')[1].data
    velscale = fits.open(filename + "_BinSpectra.fits")[0].header['VELSCALE']

    mask = table_kin.H4 > -999

    fig = plt.figure(figsize=[12, 2])
    plt.clf()
    plt.subplot(141)
    plt.plot(table_kin.SIGMA[mask], table_kin.V[mask] - table_kin_true.V[mask], '+k', alpha=0.6)
    plt.axhline(0, color='r')
    plt.axvline(velscale, linestyle='dashed')
    # plt.axvline(2 * velscale, linestyle='dashed')
    plt.ylim(-25, 25)
    plt.xlabel(r'$\sigma_{\rm true}\ (\rm{km\ s^{-1}})$')
    plt.ylabel(r'$V_{\rm LOS} - V_{\rm true}\ (\rm{km\ s^{-1}})$')
    # plt.text(1.025 * velscale, 15, r'1$\times$velscale')

    plt.subplot(142)
    plt.plot(table_kin.SIGMA[mask], table_kin.SIGMA[mask] - table_kin_true.SIGMA[mask], '+k', alpha=0.6)
    plt.axhline(0, color='r')
    plt.axvline(velscale, linestyle='dashed')
    # plt.axvline(2 * velscale, linestyle='dashed')
    plt.ylim(-25, 25)
    plt.xlabel(r'$\sigma_{\rm true}\ (\rm{km\ s^{-1}})$')
    plt.ylabel(r'$\sigma - \sigma_{\rm true}\ (\rm{km\ s^{-1}})$')
    # plt.text(1.025 * velscale, 15, r'1$\times$velscale')

    plt.subplot(143)
    plt.plot(table_kin.SIGMA[mask], table_kin.H3[mask] - table_kin_true.H3[mask], '+k', alpha=0.6)
    plt.axhline(0, color='r')
    # plt.axhline(0, linestyle='dotted', color='limegreen')
    plt.axvline(velscale, linestyle='dashed')
    # plt.axvline(2 * velscale, linestyle='dashed')
    plt.ylim(-0.2, 0.2)
    plt.xlabel(r'$\sigma_{\rm true}\ (\rm{km\ s^{-1}})$')
    plt.ylabel(r'$h3 - h3_{\rm true}$')
    # plt.text(1.025 * velscale, 0.15, r'1$\times$velscale')

    plt.subplot(144)
    plt.plot(table_kin.SIGMA[mask], table_kin.H4[mask] - table_kin_true.H4[mask], '+k', alpha=0.6)
    plt.axhline(0, color='r')
    # plt.axhline(0, linestyle='dotted', color='limegreen')
    plt.axvline(velscale, linestyle='dashed')
    # plt.axvline(2 * velscale, linestyle='dashed')
    plt.ylim(-0.2, 0.2)
    plt.xlabel(r'$\sigma_{\rm true}\ (\rm{km\ s^{-1}})$')
    plt.ylabel(r'$h4 - h4_{\rm true}$')
    # plt.text(1.025 * velscale, 0.15, r'1$\times$velscale')
    plt.tight_layout(pad=0.05, w_pad=0.6)

    return fig










def read_results_sfh(filename, flag, suffix):
    # Read bintable
    table_hdu = fits.open(filename + '_table.fits')
    idx_inside = np.where(table_hdu[1].data.BIN_ID >= 0)[0]
    X = np.array(table_hdu[1].data.X[idx_inside]) * -1
    Y = np.array(table_hdu[1].data.Y[idx_inside])
    FLUX = np.array(table_hdu[1].data.FLUX[idx_inside])
    binNum_long = np.array(table_hdu[1].data.BIN_ID[idx_inside])
    ubins = np.unique(binNum_long)
    pixelsize = table_hdu[0].header['PIXSIZE']

    result_l = []

    for flag_i, suffix_i in zip(flag, suffix):

        # Read results
        if flag_i == 'SFH':
            # print(filename + '_sfh' + suffix_i + '.fits')
            sfh_hdu = fits.open(filename + '_sfh' + suffix_i + '.fits')
            result = np.zeros((len(ubins), 3))
            result[:, 0] = np.array(sfh_hdu[1].data.AGE)
            result[:, 1] = np.array(sfh_hdu[1].data.METAL)
            result[:, 2] = np.array(sfh_hdu[1].data.ALPHA)

        # Read results
        elif flag_i == 'SFH_ERR':
            # print(filename + '_sfh' + suffix_i + '.fits')
            sfh_hdu = fits.open(filename + '_sfh' + suffix_i + '.fits')
            result = np.zeros((len(ubins), 3))
            result[:, 0] = np.array(sfh_hdu[1].data.ERR_AGE)
            result[:, 1] = np.array(sfh_hdu[1].data.ERR_METAL)
            result[:, 2] = np.array(sfh_hdu[1].data.ERR_ALPHA)

        # Read results
        elif flag_i == 'SFH_BT':
            # print(filename + '_sfh' + suffix_i + '.fits')
            sfh_hdu = fits.open(filename + '_sfh' + suffix_i + '.fits')
            result = np.zeros((len(ubins), 3))
            result[:, 0] = np.array(sfh_hdu[1].data.AGE_BT)
            result[:, 1] = np.array(sfh_hdu[1].data.METAL_BT)
            result[:, 2] = np.array(sfh_hdu[1].data.ALPHA_BT)


        elif flag_i == 'LS':
            # print(filename + '_ls_AdapRes' + suffix_i + '.fits')
            ls_hdu = fits.open(filename + '_ls_AdapRes' + suffix_i + '.fits')
            result = np.zeros((len(ubins), 3))
            result[:, 0] = np.array(ls_hdu[1].data.AGE)
            result[:, 1] = np.array(ls_hdu[1].data.METAL)
            result[:, 2] = np.array(ls_hdu[1].data.ALPHA)

        # Convert results to long version
        result_long = np.zeros((len(binNum_long), result.shape[1]));
        result_long[:, :] = np.nan
        for i in range(len(ubins)):
            idx = np.where(ubins[i] == np.abs(binNum_long))[0]
            result_long[idx, :] = result[i, :]
        result = result_long
        result_l.append(result)

    return result_l


def plot_sfh(filename, rootname, result_l, plot_contour=False, contour_offset_saved=0.2, vminmax=None, use_vminmax=None,
             residual=None, cmap_l=None):

    if np.all(vminmax != None):
        if len(vminmax.shape) == 2:
            vminmax = np.tile(vminmax, (len(result_l), 1, 1))
        elif len(vminmax.shape) == 1:
            vminmax = np.tile(vminmax, (len(result_l), result_l[0].shape[1], 1))
    if np.all(use_vminmax != None):
        if type(use_vminmax) == bool and len(result_l) > 1:
            use_vminmax = [use_vminmax] * len(result_l)
    if np.all(residual == None):
        residual = [False] * len(result_l)
    if np.all(cmap_l == None):
        cmap_l = ['sauron'] * len(result_l)


    # Read bintable
    table_hdu = fits.open(filename + '_table.fits')
    idx_inside = np.where(table_hdu[1].data.BIN_ID >= 0)[0]
    X = np.array(table_hdu[1].data.X[idx_inside]) * -1
    Y = np.array(table_hdu[1].data.Y[idx_inside])
    FLUX = np.array(table_hdu[1].data.FLUX[idx_inside])
    binNum_long = np.array(table_hdu[1].data.BIN_ID[idx_inside])
    ubins = np.unique(binNum_long)
    pixelsize = table_hdu[0].header['PIXSIZE']

    # Check spatial coordinates
    if len(np.where(np.logical_or(X == 0.0, np.isnan(X) == True))[0]) == len(X):
        print(
            'All X-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!')
    if len(np.where(np.logical_or(Y == 0.0, np.isnan(Y) == True))[0]) == len(Y):
        print(
            'All Y-coordinates are 0.0 or np.nan. Plotting maps will not work without reasonable spatial information!\n')

    # result = result_l[0]
    # if len(np.unique(result[:, 2])) == 1:
    #     labellist = ['AGE', 'METAL']
    # else:
    labellist = ['AGE', 'METAL', 'ALPHA']

    # Setup main figure
    nrows = len(result_l)
    nplots = len(labellist)
    fig = plt.figure(figsize=(4.5 * nplots, 4 * nrows))
    grid = AxesGrid(fig, 111, nrows_ncols=(nrows, nplots), axes_pad=0.0, share_all=True, \
                    label_mode="L", cbar_location="right", cbar_mode="single", cbar_size='1.5%')
    contour_offset = contour_offset_saved

    for row_i in range(nrows):
        result = result_l[row_i]

        for iterate in range(0, nplots):
            subplot_index = row_i * nplots + iterate

            # Prepare main plot
            val = result[:, iterate]

            if np.all(vminmax != None) and use_vminmax[row_i] == True:
                vmin = vminmax[row_i, iterate, 0]
                vmax = vminmax[row_i, iterate, 1]
            elif residual[row_i] == False:
                # Determine vmin/vmax automatically, if called from within the pipeline
                vmin = np.nanpercentile(val, 0.5)
                vmax = np.nanpercentile(val, 99.5)
            elif residual[row_i] == True:
                vmin_0 = np.nanpercentile(val, 0.5)
                vmax_0 = np.nanpercentile(val, 99.5)
                vmin = -np.max(np.abs([vmin_0, vmax_0]))
                vmax = np.max(np.abs([vmin_0, vmax_0]))

            # Create image in pixels
            xmin = np.nanmin(X) - 5;
            xmax = np.nanmax(X) + 5
            ymin = np.nanmin(Y) - 5;
            ymax = np.nanmax(Y) + 5
            npixels_x = int(np.round((xmax - xmin) / pixelsize) + 1)
            npixels_y = int(np.round((ymax - ymin) / pixelsize) + 1)
            i = np.array(np.round((X - xmin) / pixelsize), dtype=int)
            j = np.array(np.round((Y - ymin) / pixelsize), dtype=int)
            image = np.full((npixels_x, npixels_y), np.nan)
            image[i, j] = val

            # Plot map and colorbar
            image = grid[subplot_index].imshow(np.rot90(image), cmap=cmap_l[row_i], interpolation=None, vmin=vmin, vmax=vmax,
                                               extent=[-(xmin - pixelsize / 2), -(xmax + pixelsize / 2), ymin - pixelsize / 2, ymax + pixelsize / 2])
            grid.cbar_axes[subplot_index].colorbar(image)

            if plot_contour:
                # Plot contours
                XY_Triangulation = Triangulation(X - pixelsize / 2,
                                                 Y - pixelsize / 2)  # Create a mesh from a Delaunay triangulation
                XY_Triangulation.set_mask(TriAnalyzer(XY_Triangulation).get_flat_tri_mask(
                    0.01))  # Remove bad triangles at the border of the field-of-view
                levels = np.arange(np.min(np.log10(FLUX)) + contour_offset, np.max(np.log10(FLUX)), 0.2)
                grid[subplot_index].tricontour(XY_Triangulation, np.log10(FLUX), levels=levels, linewidths=1,
                                               colors='k')

            # Label vmin and vmax
            if iterate in [0, 1, 2]:
                grid[subplot_index].text(0.02, 0.85, r'[%.2f, %.2f]' % (vmin, vmax),
                                         horizontalalignment='left', verticalalignment='top',
                                         transform=grid[subplot_index].transAxes, fontsize=14)

        # Set age, metallicity labels
        grid[row_i * nplots].text(0.02, 0.975, r'Age [Gyr]', horizontalalignment='left',
                                  verticalalignment='top', transform=grid[row_i * nplots].transAxes, fontsize=16)
        grid[row_i * nplots].text(0.975, 0.965, rootname[row_i], horizontalalignment='right', fontweight='semibold',
                                  verticalalignment='top', transform=grid[row_i * nplots].transAxes, fontsize=16)
        grid[row_i * nplots + 1].text(0.02, 0.975, r'[M/H]', horizontalalignment='left',
                                      verticalalignment='top', transform=grid[row_i * nplots + 1].transAxes,
                                      fontsize=16)
        if nplots == 3:
            grid[row_i * nplots + 2].text(0.02, 0.975, r'[$\alpha$/Fe]', horizontalalignment='left',
                                          verticalalignment='top', transform=grid[row_i * nplots + 2].transAxes,
                                          fontsize=16)

        # Set xlabel and ylabel
        grid[row_i * nplots].set_ylabel(r'$y$ [arcsec]', fontsize=15)
        grid[row_i * nplots].yaxis.set_major_locator(MultipleLocator(10))  # Major tick every 10 units
        grid[row_i * nplots].yaxis.set_minor_locator(MultipleLocator(2))   # Minor tick every 1 units


        # Set tick frequency and parameters
        for plot_i in range(nplots):
            grid[row_i * nplots + plot_i].set_xlabel(r'$x$ [arcsec]', fontsize=15)
            grid[row_i * nplots + plot_i].xaxis.set_major_locator(MultipleLocator(20))
            grid[row_i * nplots + plot_i].xaxis.set_minor_locator(MultipleLocator(2))
            grid[row_i * nplots + plot_i].tick_params(direction="in", which='both', bottom=True, top=True, left=True, right=True, labelsize=13) # Ticks inside of plot

    # Invert x-axis
    grid[0].invert_xaxis()

    # Remove ticks and labels from colorbar
    for cax in grid.cbar_axes:
        cax.toggle_label(False)
        cax.yaxis.set_ticks([])

    return fig


def plot_kin_scatter(table_kin, table_kin_true):

        fig, axes = plt.subplots(4, 1, figsize=[5, 11])

        axes[0].scatter(table_kin.V, table_kin_true.V, s=8, marker='o', alpha=0.2)
        axes[0].plot([-500, 500], [-500, 500], 'k-')
        axes[0].set_xlim(np.min([table_kin.V, table_kin_true.V]) - 10, np.max([table_kin.V, table_kin_true.V]) + 10)
        axes[0].set_ylim(np.min([table_kin.V, table_kin_true.V]) - 10, np.max([table_kin.V, table_kin_true.V]) + 10)
        axes[0].set_xlabel(r'PPXF $V_{\mathrm{stellar}} \mathrm{[km/s]}$')
        axes[0].set_ylabel(r'True $V_{\mathrm{stellar}} \mathrm{[km/s]}$')

        axes[1].scatter(table_kin.SIGMA, table_kin_true.SIGMA, s=8, marker='o', alpha=0.2)
        axes[1].plot([0, 300], [0, 300], 'k-')
        axes[1].set_xlim(np.min([table_kin.SIGMA, table_kin_true.SIGMA]) - 5,
                        np.max([table_kin.SIGMA, table_kin_true.SIGMA]) + 5)
        axes[1].set_ylim(np.min([table_kin.SIGMA, table_kin_true.SIGMA]) - 5,
                        np.max([table_kin.SIGMA, table_kin_true.SIGMA]) + 5)
        axes[1].set_xlabel(r'PPXF $\sigma_{\mathrm{stellar}} \mathrm{[km/s]}$')
        axes[1].set_ylabel(r'True $\sigma_{\mathrm{stellar}} \mathrm{[km/s]}$')

        axes[2].scatter(table_kin.H3, table_kin_true.H3, s=8, marker='o', alpha=0.2)
        axes[2].plot([-2, 2], [-2, 2], 'k-')
        axes[2].set_xlim(np.min([table_kin.H3, table_kin_true.H3]) - 0.03, np.max([table_kin.H3, table_kin_true.H3]) + 0.03)
        axes[2].set_ylim(np.min([table_kin.H3, table_kin_true.H3]) - 0.03, np.max([table_kin.H3, table_kin_true.H3]) + 0.03)
        axes[2].set_xlabel(r'PPXF $h3$')
        axes[2].set_ylabel(r'True $h3$')

        axes[3].scatter(table_kin.H4, table_kin_true.H4, s=8, marker='o', alpha=0.2)
        axes[3].plot([-2, 2], [-2, 2], 'k-')
        axes[3].set_xlim(np.min([table_kin.H4, table_kin_true.H4]) - 0.03, np.max([table_kin.H4, table_kin_true.H4]) + 0.03)
        axes[3].set_ylim(np.min([table_kin.H4, table_kin_true.H4]) - 0.03, np.max([table_kin.H4, table_kin_true.H4]) + 0.03)
        axes[3].set_xlabel(r'PPXF $h4$')
        axes[3].set_ylabel(r'True $h4$')

        fig.tight_layout(pad=0.2, h_pad=0.50)

        return fig


def plot_sfh_scatter(table_sfh, table_sfh_true):

        if np.all(table_sfh.ALPHA==0.): ncol = 2
        else: ncol = 3

        fig, axes = plt.subplots(ncol, 1, figsize=[5, 9])

        axes[0].scatter(table_sfh.AGE, table_sfh_true.AGE, s=8, marker='o', alpha=0.2)
        axes[0].plot([0, 14], [0, 14], 'k-')
        axes[0].set_xlim(np.min([table_sfh.AGE, table_sfh_true.AGE]) - 0.5,
                         np.max([table_sfh.AGE, table_sfh_true.AGE]) + 0.5)
        axes[0].set_ylim(np.min([table_sfh.AGE, table_sfh_true.AGE]) - 0.5,
                         np.max([table_sfh.AGE, table_sfh_true.AGE]) + 0.5)
        axes[0].set_xlabel('PPXF Age [Gyr]')
        axes[0].set_ylabel('True Age [Gyr]')

        axes[1].scatter(table_sfh.METAL, table_sfh_true.METAL, s=8, marker='o', alpha=0.2)
        axes[1].plot([-2, 2], [-2, 2], 'k-')
        axes[1].set_xlim(np.min([table_sfh.METAL, table_sfh_true.METAL]) - 0.02,
                         np.max([table_sfh.METAL, table_sfh_true.METAL]) + 0.02)
        axes[1].set_ylim(np.min([table_sfh.METAL, table_sfh_true.METAL]) - 0.02,
                         np.max([table_sfh.METAL, table_sfh_true.METAL]) + 0.02)
        axes[1].set_xlabel('PPXF [M/H]')
        axes[1].set_ylabel('True [M/H]')

        if ncol == 3:
            axes[2].scatter(table_sfh.ALPHA, table_sfh_true.ALPHA, s=8, marker='o', alpha=0.2)
            axes[2].plot([-2, 2], [-2, 2], 'k-')
            axes[2].set_xlim(np.min([table_sfh.ALPHA, table_sfh_true.ALPHA]) - 0.02,
                             np.max([table_sfh.ALPHA, table_sfh_true.ALPHA]) + 0.02)
            axes[2].set_ylim(np.min([table_sfh.ALPHA, table_sfh_true.ALPHA]) - 0.02,
                             np.max([table_sfh.ALPHA, table_sfh_true.ALPHA]) + 0.02)
            axes[2].set_xlabel(r'PPXF [$\alpha$/Fe]')
            axes[2].set_ylabel(r'True [$\alpha$/Fe]')

        fig.tight_layout(pad=0.2, h_pad=0.50)

        return fig
