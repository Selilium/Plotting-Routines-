import numpy as np
from spectres import spectres
from astropy.io import fits
from astropy.table import Table


def mask_mass_arrays(mass_array, mask_age, mask_metal, mask_alpha):
    mass_array = mass_array[mask_age, :, :]
    mass_array = mass_array[:, mask_metal, :]
    mass_array = mass_array[:, :, mask_alpha]
    return mass_array


class weights:

    def __init__(self, cube_gist_path, cube_gist_run, mode='ppxf', fraction_type='light fraction'):

        cube_gist_run_path = cube_gist_path + cube_gist_run + '/'
        filename = cube_gist_run_path + cube_gist_run

        if mode == 'ppxf':
            if 'rmax' in cube_gist_run:
                weights = fits.open(filename + '_sfh-weights_rmax.fits')
            else:
                weights = fits.open(filename + '_sfh-weights.fits')
            weights_values = np.zeros(weights[1].data.WEIGHTS.shape)
            for i in range(weights_values.shape[0]):
                weights_values[i, :] = weights[1].data.WEIGHTS[i, :] * weights[1].data.WEIGHTS_FACTOR[i]
            original_weights_values = weights_values.copy()
            weights_values = weights_values / np.nansum(weights_values)
            weights_values_forsum = weights_values.copy()

        if mode == 'true':
            if fraction_type == 'light fraction':
                weights = fits.open(filename + '_sfh-weights_true_lw.fits')
            else:
                weights = fits.open(filename + '_sfh-weights_true_mw.fits')
            weights_values = weights[1].data.WEIGHTS
            original_weights_values = weights_values.copy()
            weights_values = weights_values / np.nansum(weights_values)
            weights_values_forsum = weights_values.copy()

        if 'ppxf_bt' in mode:
            if 'rmax' in cube_gist_run:
                weights = fits.open(filename + '_sfh-weights_rmax.fits')
            else:
                weights = fits.open(filename + '_sfh-weights.fits')
            weights_values_BT = np.zeros(weights[1].data.WEIGHTS.shape)
            weights_std_BT  = np.zeros(weights[1].data.WEIGHTS.shape)
            weights_sig1_BT = np.zeros(weights[1].data.WEIGHTS.shape)
            weights_sig2_BT = np.zeros(weights[1].data.WEIGHTS.shape)
            for i in range(weights_values_BT.shape[0]):
                weights_values_BT[i, :] = weights[1].data.WEIGHTS_BT[i, :] * weights[1].data.WEIGHTS_FACTOR_BT[i]
                weights_std_BT[i, :]    = weights[1].data.ERR_WEIGHTS[i, :] * weights[1].data.WEIGHTS_FACTOR_BT[i]
                weights_sig1_BT[i, :]   = weights[1].data.ERR1_WEIGHTS[i, :] * weights[1].data.WEIGHTS_FACTOR_BT[i]
                weights_sig2_BT[i, :]   = weights[1].data.ERR2_WEIGHTS[i, :] * weights[1].data.WEIGHTS_FACTOR_BT[i]
            weights_factor_BT = np.nansum(weights_values_BT)
            if 'low' in mode:
                original_weights_values = weights_sig1_BT.copy()
                weights_sig1_BT = weights_sig1_BT / weights_factor_BT
                weights_values = weights_sig1_BT.copy()
            elif 'high' in mode:
                original_weights_values = weights_sig2_BT.copy()
                weights_sig2_BT = weights_sig2_BT / weights_factor_BT
                weights_values = weights_sig2_BT.copy()
            elif 'mid' in mode:
                original_weights_values = weights_values_BT.copy()
                weights_values_BT = weights_values_BT / weights_factor_BT
                weights_values = weights_values_BT.copy()
            elif 'std' in mode:
                original_weights_values = weights_std_BT.copy()
                weights_std_BT = weights_std_BT / weights_factor_BT
                weights_values = weights_std_BT.copy()
            weights_values_forsum = weights_values_BT.copy() / weights_factor_BT

        # weights_colnames = [weights[1].data.__dict__['_coldefs'].columns[i].name for i in range(len(weights[1].data.__dict__['_coldefs'].columns))]
        # if 'WEIGHTS_BT' in weights_colnames:
        #     weights_values_BT = np.zeros(weights[1].data.WEIGHTS.shape)
        #     weights_sig1_BT = np.zeros(weights[1].data.WEIGHTS.shape)
        #     weights_sig2_BT = np.zeros(weights[1].data.WEIGHTS.shape)
        #     for i in range(weights_values_BT.shape[0]):
        #         weights_values_BT[i, :] = weights[1].data.WEIGHTS_BT[i, :] * weights[1].data.WEIGHTS_FACTOR_BT[i]
        #         weights_sig1_BT[i, :] = weights[1].data.ERR1_WEIGHTS[i, :] * weights[1].data.WEIGHTS_FACTOR_BT[i]
        #         weights_sig2_BT[i, :] = weights[1].data.ERR2_WEIGHTS[i, :] * weights[1].data.WEIGHTS_FACTOR_BT[i]
        #     weights_factor_BT = np.nansum(weights_values_BT)
        #     weights_values_BT = weights_values_BT / weights_factor_BT
        #     weights_sig1_BT = weights_sig1_BT / weights_factor_BT
        #     weights_sig2_BT = weights_sig2_BT / weights_factor_BT
        #
        #     # Reshape the weights and grids
        #     cumulative_weights_BT = np.nansum(weights_values_BT, axis=0)
        #     cumulative_weights_BT_2d = mask_mass_arrays(cumulative_weights_BT.reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        # else:
        #     weights_values_BT = np.nan
        #     weights_sig1_BT = np.nan
        #     weights_sig2_BT = np.nan



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


        # Reshape the weights and grids
        cumulative_weights = np.nansum(weights_values, axis=0)
        cumulative_weights_2d = mask_mass_arrays(cumulative_weights.reshape(reg_dim), mask_age, mask_metal, mask_alpha)

        logage_grid_2d = mask_mass_arrays(logage_grid.reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        age_grid_2d = mask_mass_arrays(age_grid.reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        metal_grid_2d = mask_mass_arrays(metal_grid.reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        alpha_grid_2d = mask_mass_arrays(alpha_grid.reshape(reg_dim), mask_age, mask_metal, mask_alpha)

        xgrid = age_grid_2d[:, :, 0]
        ygrid = metal_grid_2d[:, :, 0]
        zgrid = alpha_grid_2d[:, :, 0]

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
        weights_nuclear = mask_mass_arrays(np.nansum(weights_values[binid_nuclear, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_bulge = mask_mass_arrays(np.nansum(weights_values[binid_bulge, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_thin = mask_mass_arrays(np.nansum(weights_values[binid_thin, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_thick = mask_mass_arrays(np.nansum(weights_values[binid_thick, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_innerthin = mask_mass_arrays(np.nansum(weights_values[binid_innerthin, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_outerthin = mask_mass_arrays(np.nansum(weights_values[binid_outerthin, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_innerthick = mask_mass_arrays(np.nansum(weights_values[binid_innerthick, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_outerthick = mask_mass_arrays(np.nansum(weights_values[binid_outerthick, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)

        # Mass Fraction components for sum
        weights_forsum_nuclear = mask_mass_arrays(np.nansum(weights_values_forsum[binid_nuclear, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_forsum_bulge = mask_mass_arrays(np.nansum(weights_values_forsum[binid_bulge, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_forsum_thin = mask_mass_arrays(np.nansum(weights_values_forsum[binid_thin, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_forsum_thick = mask_mass_arrays(np.nansum(weights_values_forsum[binid_thick, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_forsum_innerthin = mask_mass_arrays(np.nansum(weights_values_forsum[binid_innerthin, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_forsum_outerthin = mask_mass_arrays(np.nansum(weights_values_forsum[binid_outerthin, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_forsum_innerthick = mask_mass_arrays(np.nansum(weights_values_forsum[binid_innerthick, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)
        weights_forsum_outerthick = mask_mass_arrays(np.nansum(weights_values_forsum[binid_outerthick, :], axis=0).reshape(reg_dim), mask_age, mask_metal, mask_alpha)


        weights_nuclear = weights_nuclear / np.nansum(weights_forsum_nuclear)
        weights_bulge = weights_bulge / np.nansum(weights_forsum_bulge)
        weights_thin = weights_thin / np.nansum(weights_forsum_thin)
        weights_thick = weights_thick / np.nansum(weights_forsum_thick)
        weights_innerthin = weights_innerthin / np.nansum(weights_forsum_innerthin)
        weights_outerthin = weights_outerthin / np.nansum(weights_forsum_outerthin)
        weights_innerthick = weights_innerthick / np.nansum(weights_forsum_innerthick)
        weights_outerthick = weights_outerthick / np.nansum(weights_forsum_outerthick)

        log_wdf = np.log10(np.nansum(cumulative_weights_2d, axis=-1) / grid_area_2d[:, :, 0])
        log_wdf_nuclear = np.log10(np.nansum(weights_nuclear, axis=-1) / grid_area_2d[:, :, 0])
        log_wdf_bulge = np.log10(np.nansum(weights_bulge, axis=-1) / grid_area_2d[:, :, 0])
        log_wdf_thin = np.log10(np.nansum(weights_thin, axis=-1) / grid_area_2d[:, :, 0])
        log_wdf_thick = np.log10(np.nansum(weights_thick, axis=-1) / grid_area_2d[:, :, 0])
        log_wdf_innerthin = np.log10(np.nansum(weights_innerthin, axis=-1) / grid_area_2d[:, :, 0])
        log_wdf_outerthin = np.log10(np.nansum(weights_outerthin, axis=-1) / grid_area_2d[:, :, 0])
        log_wdf_innerthick = np.log10(np.nansum(weights_innerthick, axis=-1) / grid_area_2d[:, :, 0])
        log_wdf_outerthick = np.log10(np.nansum(weights_outerthick, axis=-1) / grid_area_2d[:, :, 0])
        log_wdf[log_wdf == -np.inf] = np.nan
        log_wdf_nuclear[log_wdf_nuclear == -np.inf] = np.nan
        log_wdf_bulge[log_wdf_bulge == -np.inf] = np.nan
        log_wdf_thin[log_wdf_thin == -np.inf] = np.nan
        log_wdf_thick[log_wdf_thick == -np.inf] = np.nan
        log_wdf_innerthin[log_wdf_innerthin == -np.inf] = np.nan
        log_wdf_outerthin[log_wdf_outerthin == -np.inf] = np.nan
        log_wdf_innerthick[log_wdf_innerthick == -np.inf] = np.nan
        log_wdf_outerthick[log_wdf_outerthick == -np.inf] = np.nan

        wdf = np.nansum(cumulative_weights_2d, axis=-1) / grid_area_2d[:, :, 0]
        wdf_nuclear = np.nansum(weights_nuclear, axis=-1) / grid_area_2d[:, :, 0]
        wdf_bulge = np.nansum(weights_bulge, axis=-1) / grid_area_2d[:, :, 0]
        wdf_thin = np.nansum(weights_thin, axis=-1) / grid_area_2d[:, :, 0]
        wdf_thick = np.nansum(weights_thick, axis=-1) / grid_area_2d[:, :, 0]
        wdf_innerthin = np.nansum(weights_innerthin, axis=-1) / grid_area_2d[:, :, 0]
        wdf_outerthin = np.nansum(weights_outerthin, axis=-1) / grid_area_2d[:, :, 0]
        wdf_innerthick = np.nansum(weights_innerthick, axis=-1) / grid_area_2d[:, :, 0]
        wdf_outerthick = np.nansum(weights_outerthick, axis=-1) / grid_area_2d[:, :, 0]

        mean_alpha_grids_nuclear = np.nansum((weights_nuclear * alpha_grid_2d), axis=-1) / np.nansum(weights_forsum_nuclear, axis=-1)
        mean_alpha_grids_bulge = np.nansum((weights_bulge * alpha_grid_2d), axis=-1) / np.nansum(weights_forsum_bulge, axis=-1)
        mean_alpha_grids_thin = np.nansum((weights_thin * alpha_grid_2d), axis=-1) / np.nansum(weights_forsum_thin, axis=-1)
        mean_alpha_grids_thick = np.nansum((weights_thick * alpha_grid_2d), axis=-1) / np.nansum(weights_forsum_thick, axis=-1)


        # Weight Fraction components
        bin_edges = np.arange(0, 14.1, 0.5)
        bin_areas = bin_edges[:-1] + np.diff(bin_edges) / 2
        statistic_sfh = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                 np.concatenate(([0.], np.nansum(cumulative_weights_2d, axis=(1, 2)) / yyb[:, 0, 0])),
                                 fill=np.nan, verbose=False)
        statistic_sfh_nuclear = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                         np.concatenate(([0.], np.nansum(weights_nuclear, axis=(1, 2)) / yyb[:, 0, 0])),
                                         fill=np.nan, verbose=False)
        statistic_sfh_bulge = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                       np.concatenate(([0.], np.nansum(weights_bulge, axis=(1, 2)) / yyb[:, 0, 0])),
                                       fill=np.nan, verbose=False)
        statistic_sfh_thin = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                      np.concatenate(([0.], np.nansum(weights_thin, axis=(1, 2)) / yyb[:, 0, 0])),
                                      fill=np.nan, verbose=False)
        statistic_sfh_thick = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                       np.concatenate(([0.], np.nansum(weights_thick, axis=(1, 2)) / yyb[:, 0, 0])),
                                       fill=np.nan, verbose=False)
        statistic_sfh_innerthin = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                           np.concatenate(([0.], np.nansum(weights_innerthin, axis=(1, 2)) / yyb[:, 0, 0])),
                                           fill=np.nan, verbose=False)
        statistic_sfh_outerthin = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                           np.concatenate(([0.], np.nansum(weights_outerthin, axis=(1, 2)) / yyb[:, 0, 0])),
                                           fill=np.nan, verbose=False)
        statistic_sfh_innerthick = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                            np.concatenate(([0.], np.nansum(weights_innerthick, axis=(1, 2)) / yyb[:, 0, 0])),
                                            fill=np.nan, verbose=False)
        statistic_sfh_outerthick = spectres(bin_areas, np.concatenate(([0.], age_grid_2d[:, 0, 0])),
                                            np.concatenate(([0.], np.nansum(weights_outerthick, axis=(1, 2)) / yyb[:, 0, 0])),
                                            fill=np.nan, verbose=False)



        metal = np.nansum(cumulative_weights_2d, axis=(0, 2)) / xxb[0, :, 0]
        metal_thin = np.nansum(weights_thin, axis=(0, 2)) / xxb[0, :, 0]
        metal_thick = np.nansum(weights_thick, axis=(0, 2)) / xxb[0, :, 0]
        metal_bulge = np.nansum(weights_bulge, axis=(0, 2)) / xxb[0, :, 0]
        metal_nuclear = np.nansum(weights_nuclear, axis=(0, 2)) / xxb[0, :, 0]






        self.cube_gist_run_path  =  cube_gist_run_path
        self.filename  =  filename
        self.original_weights_values = original_weights_values
        self.bin_edges = bin_edges
        self.bin_areas = bin_areas
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.zgrid = zgrid
        self.grid_area_2d = grid_area_2d
        self.grid_area_2d_log = grid_area_2d_log
        self.age_grid_2d = age_grid_2d
        self.metal_grid_2d = metal_grid_2d
        self.alpha_grid_2d = alpha_grid_2d
        self.xb = xb
        self.xxb = xxb
        self.xxb_log = xxb_log
        self.yb = yb
        self.yb_log = yb_log
        self.yyb = yyb
        self.yyb_log = yyb_log
        self.zb = zb
        self.zzb = zzb
        self.zzb_log = zzb_log
        self.reg_dim = reg_dim
        self.cumulative_weights_2d = cumulative_weights_2d
        self.weights_values = weights_values
        self.weights_nuclear = weights_nuclear
        self.weights_bulge = weights_bulge
        self.weights_thin = weights_thin
        self.weights_thick = weights_thick
        self.weights_innerthin = weights_innerthin
        self.weights_outerthin = weights_outerthin
        self.weights_innerthick = weights_innerthick
        self.weights_outerthick = weights_outerthick
        self.results_table = results_table
        self.pixelsize = pixelsize
        self.flux_reshape = flux_reshape
        self.results_nuclear = results_nuclear
        self.results_bulge = results_bulge
        self.results_thin = results_thin
        self.results_thick = results_thick
        self.results_innerthin = results_innerthin
        self.results_outerthin = results_outerthin
        self.results_innerthick = results_innerthick
        self.results_outerthick = results_outerthick
        self.wdf = wdf
        self.wdf_nuclear = wdf_nuclear
        self.wdf_bulge = wdf_bulge
        self.wdf_thin = wdf_thin
        self.wdf_thick = wdf_thick
        self.wdf_innerthin = wdf_innerthin
        self.wdf_outerthin = wdf_outerthin
        self.wdf_innerthick = wdf_innerthick
        self.wdf_outerthick = wdf_outerthick
        self.log_wdf = log_wdf
        self.log_wdf_nuclear = log_wdf_nuclear
        self.log_wdf_bulge = log_wdf_bulge
        self.log_wdf_thin = log_wdf_thin
        self.log_wdf_thick = log_wdf_thick
        self.log_wdf_innerthin = log_wdf_innerthin
        self.log_wdf_outerthin = log_wdf_outerthin
        self.log_wdf_innerthick = log_wdf_innerthick
        self.log_wdf_outerthick = log_wdf_outerthick
        self.mean_alpha_grids_nuclear = mean_alpha_grids_nuclear
        self.mean_alpha_grids_bulge = mean_alpha_grids_bulge
        self.mean_alpha_grids_thin = mean_alpha_grids_thin
        self.mean_alpha_grids_thick = mean_alpha_grids_thick
        self.statistic_sfh = statistic_sfh
        self.statistic_sfh_nuclear = statistic_sfh_nuclear
        self.statistic_sfh_bulge = statistic_sfh_bulge
        self.statistic_sfh_thin = statistic_sfh_thin
        self.statistic_sfh_thick = statistic_sfh_thick
        self.statistic_sfh_innerthin = statistic_sfh_innerthin
        self.statistic_sfh_outerthin = statistic_sfh_outerthin
        self.statistic_sfh_innerthick = statistic_sfh_innerthick
        self.statistic_sfh_outerthick = statistic_sfh_outerthick
        self.metal = metal
        self.metal_nuclear = metal_nuclear
        self.metal_bulge = metal_bulge
        self.metal_thin = metal_thin
        self.metal_thick = metal_thick
