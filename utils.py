import pickle
import numpy as np
from multiprocessing import Pool
import ppxf.ppxf_util as ppxf_util
from time import perf_counter as clock

from scipy import interpolate
from scipy.stats import binned_statistic_2d
from scipy.stats import pearsonr

import matplotlib.colors as colors
from matplotlib import pyplot as plt


##################################################################################
# Doppler shift functions

def doppler_shift_payne(wavelength, flux, dv):
    '''
    This is the function from The Payne
    dv is in km/s
    We use the convention where a positive dv means the object is moving away.

    This linear interpolation is actually not that accurate, but is fine if you
    only care about accuracy to the level of a few tenths of a km/s. If you care
    about better accuracy, you can do better with spline interpolation.
    '''
    c = 2.99792458e5 # km/s
    doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c))
    new_wavelength = wavelength * doppler_factor
    new_flux = np.interp(new_wavelength, wavelength, flux)
    return new_flux



def doppler_shift(wavelength, flux, dv):
    '''spec
    This is the function developed by myself
    dv is in km/s
    We use the convention where a positive dv means the object is moving away.

    This linear interpolation is actually not that accurate, but is fine if you
    only care about accuracy to the level of a few tenths of a km/s. If you care
    about better accuracy, you can do better with spline interpolation.
    '''
    c = 2.99792458e5 # km/s
    doppler_factor = np.sqrt((1 + dv/c)/(1 - dv/c))
    new_wavelength = wavelength * doppler_factor
    new_flux = np.interp(wavelength, new_wavelength, flux)
    # Try to change the up line, use their rebinning function
    return new_flux


def doppler_shift_inarray(wavelength, flux, dv):
    '''
    This is the function developed by myself
    dv is in km/s
    We use the convention where a positive dv means the object is moving away.

    This linear interpolation is actually not that accurate, but is fine if you
    only care about accuracy to the level of a few tenths of a km/s. If you care
    about better accuracy, you can do better with spline interpolation.
    '''
    c = 2.99792458e5 # km/s
    doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c))
    new_wavelength = wavelength.reshape(-1, 1) @ doppler_factor.reshape(1, -1)
    new_flux = np.interp(new_wavelength, wavelength, flux)
    return new_flux


def doppler_shift_inarray_scipy(wavelength, flux, dv):
    '''
    This is the function developed by myself
    dv is in km/s
    We use the convention where a positive dv means the object is moving away.

    This linear interpolation is actually not that accurate, but is fine if you
    only care about accuracy to the level of a few tenths of a km/s. If you care
    about better accuracy, you can do better with spline interpolation.
    '''
    c = 2.99792458e5 # km/s
    doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c))
    new_wavelength = wavelength.reshape(-1, 1) @ doppler_factor.reshape(1, -1)
    f = interpolate.interp1d(wavelength, flux, fill_value="extrapolate", axis=0)
    new_flux = f(new_wavelength)
    return new_flux

##################################################################################




##################################################################################
# Reddening function
def reddening_cal00(lam, ebv):
    """
    Reddening curve of `Calzetti et al. (2000)
    <http://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C>`_
    This is reliable between 0.12 and 2.2 micrometres.
    - LAMBDA is the restframe wavelength in Angstrom of each pixel in the
      input galaxy spectrum (1 Angstrom = 1e-4 micrometres)
    - EBV is the assumed E(B-V) colour excess to redden the spectrum.
      In output the vector FRAC gives the fraction by which the flux at each
      wavelength has to be multiplied, to model the dust reddening effect.

    """
    ilam = 1e4/lam  # Convert Angstrom to micrometres and take 1/lambda
    rv = 4.05  # C+00 equation (5)

    # C+00 equation (3) but extrapolate for lam > 2.2
    # C+00 equation (4) (into Horner form) but extrapolate for lam < 0.12
    k1 = rv + np.where(lam >= 6300, 2.76536*ilam - 4.93776,
                       ilam*((0.029249*ilam - 0.526482)*ilam + 4.01243) - 5.7328)
    fact = 10**(-0.4*ebv*k1.clip(0))  # Calzetti+00 equation (2) with opposite sign

    return fact # The model spectrum has to be multiplied by this vector




##################################################################################




##################################################################################
# Read and write CLASS file

def dict2obj(d):
    '''
    Internal function to transfer dict to object for PPXF results
    :param d:
    :return:
    '''
    # checking whether object d is a
    # instance of class list
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]

        # if d is not a instance of dict then
    # directly object is returned
    if not isinstance(d, dict):
        return d

    # declaring a class
    class C:
        pass

    # constructor of the class passed to obj
    obj = C()

    for k in d:
        obj.__dict__[k] = dict2obj(d[k])

    return obj


def save_object(obj, filename):
    '''
    This function is for saving the PPXF results in the type of "CLASS"
    :param obj:
    :param filename:
    :return:
    '''
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)



def load_object(filename):
    '''
    This functino is for loading the PPXF results in the type of "CLASS"
    :param filename:
    :return:
    '''
    file = open(filename, 'rb')
    object_file = pickle.load(file)
    obj_ = dict2obj(object_file)
    return obj_


##################################################################################





##################################################################################
# Functions to make plots

def plot_binned_grids_color(x, y, values, statistic, x_edges, y_edges, xlabel, ylabel,
                            plot_cb=True, cmap='hot', cblabel=None, color_Lognorm=False,
                            invert_x=True, **kwargs):
    '''

    :param x:
    :param y:
    :param values:
    :param statistic:
    :param x_edges:
    :param y_edges:
    :param cmap:
    :param xlabel:
    :param ylabel:
    :param cblabel:
    :param percentile:
    :param norm:
    :return:
    '''

    binned_statistic = binned_statistic_2d(x, y, values, statistic, bins=[x_edges, y_edges]).statistic

    if statistic == 'count':
        binned_statistic[binned_statistic==0] = np.nan

    vmin = np.nanpercentile(binned_statistic[binned_statistic!=0], 0.5)
    vmax = np.nanpercentile(binned_statistic[binned_statistic!=0], 99.5)

    if color_Lognorm == True:
        color_norm = colors.LogNorm(vmin, vmax)
    else:
        color_norm = colors.Normalize(vmin, vmax)

    ax = plt.gca()
    im = ax.pcolormesh(x_edges, y_edges, binned_statistic.T, cmap=cmap, norm=color_norm, **kwargs)
    if plot_cb == True:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cblabel)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if invert_x:
        ax.invert_xaxis()
    # fig.tight_layout()

    return im, ax, binned_statistic, x_edges, y_edges




def plot_massfraction_2d(mass_grid_alpha00, mass_grid_alpha04, age_grid_2d, metal_grid_2d, linear_age=False, **kwargs):
    '''
    plot mass fractions for alpha=0 and alpha=0.4 populations
    :param mass_grid_alpha00:
    :param mass_grid_alpha04:
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
        xlabel = "log Age (Gyr)"
    ygrid = metal_grid_2d

    plt.figure(figsize=[12, 4])
    plt.subplot(121)
    ppxf_util.plot_weights_2d(xgrid, ygrid, mass_grid_alpha00, nodots=False, colorbar=True,
                              title=r'Mass Fraction [$\alpha$/Fe]=0.0', xlabel=xlabel, **kwargs)
    plt.subplot(122)
    ppxf_util.plot_weights_2d(xgrid, ygrid, mass_grid_alpha04, nodots=False, colorbar=True,
                              title=r'Mass Fraction [$\alpha$/Fe]=0.4', xlabel=xlabel, **kwargs)
    plt.tight_layout()




def plot_losvd_distribution_2d(particles, age_bins, metal_bins, age_grid_2d, metal_grid_2d, threshold, linear_age=False,
                               **kwargs):
    '''
    plot the median and std velocities of all stars in each spectra template.
    :param particles:
    :param age_bins:
    :param metal_bins:
    :param age_grid_2d:
    :param metal_grid_2d:
    :param threshold:
    :param linear_age:
    :param kwargs:
    :return:
    '''

    statistic_v = binned_statistic_2d(x=np.log10((particles['cube_age']) * 1e9), y=particles['cube_m_h'], values=particles['vr'],
                                      statistic='median', bins=[age_bins, metal_bins], expand_binnumbers=True).statistic
    statistic_vsig = binned_statistic_2d(x=np.log10((particles['cube_age']) * 1e9), y=particles['cube_m_h'],
                                         values=particles['vr'], statistic='std', bins=[age_bins, metal_bins],
                                         expand_binnumbers=True).statistic
    statistic_number = binned_statistic_2d(x=np.log10((particles['cube_age']) * 1e9), y=particles['cube_m_h'],
                                           values=particles['vr'], statistic='count', bins=[age_bins, metal_bins],
                                           expand_binnumbers=True).statistic

    statistic_v[statistic_number < threshold] = np.nan
    statistic_vsig[statistic_number < threshold] = np.nan

    if linear_age:
        xgrid = age_grid_2d
        xlabel = "Age (Gyr)"
    else:
        xgrid = np.log10(age_grid_2d) + 9
        xlabel = "log Age (Gyr)"
    ygrid = metal_grid_2d

    plt.figure(figsize=[12, 4])
    plt.subplot(121)
    ppxf_util.plot_weights_2d(xgrid, ygrid, statistic_v, nodots=False, colorbar=True,
                              title=r'Median LOS velovity', xlabel=xlabel, **kwargs)
    plt.subplot(122)
    ppxf_util.plot_weights_2d(xgrid, ygrid, statistic_vsig, nodots=False, colorbar=True,
                              title=r'Dispersion', xlabel=xlabel, **kwargs)
    plt.tight_layout()


def plot_age_metal_grid_2d(distri1, distri2, title1, title2, age_grid_2d, metal_grid_2d, linear_age=False, **kwargs):
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
        xlabel = "log Age (Gyr)"
    ygrid = metal_grid_2d

    plt.figure(figsize=[12, 4])
    plt.subplot(121)
    ppxf_util.plot_weights_2d(xgrid, ygrid, distri1, nodots=False, colorbar=True, title=title1, xlabel=xlabel, **kwargs)
    plt.subplot(122)
    ppxf_util.plot_weights_2d(xgrid, ygrid, distri2, nodots=False, colorbar=True, title=title2, xlabel=xlabel, **kwargs)
    plt.tight_layout()





def plot_massfraction_1d(mass_alpha00_list, mass_alpha04_list, age_grid, metal_grid, labels, linear_age=True):
    '''
    Plot the mass fraction in 1D, age/metallicity/alpha
    :param mass_alpha00_list:
    :param mass_alpha04_list:
    :param labels:
    :return:
    '''

    if linear_age:
        xlabel = "Age (Gyr)"
    else:
        age_grid = np.log10(age_grid) + 9
        xlabel = "log Age (Gyr)"

    if type(mass_alpha00_list) != list:
        mass_alpha00_list = [mass_alpha00_list]
        mass_alpha04_list = [mass_alpha04_list]
        labels = [labels]

    plt.figure(figsize=[12, 4])
    # age
    ax1 = plt.subplot(131)
    for mass_alpha00, mass_alpha04, label in zip(mass_alpha00_list, mass_alpha04_list, labels):
        ax1.plot(age_grid, np.sum(mass_alpha00, axis=1) + np.sum(mass_alpha04, axis=1), 'o-', label=label)
    ax1.set_ylabel('Mass Fraction')
    ax1.set_xlabel(xlabel)
    if linear_age: ax1.set_xticks(np.linspace(0, 12, 7))
    # metallicity
    ax2 = plt.subplot(132)
    for mass_alpha00, mass_alpha04, label in zip(mass_alpha00_list, mass_alpha04_list, labels):
        ax2.plot(metal_grid, np.sum(mass_alpha00, axis=0) + np.sum(mass_alpha04, axis=0), 'o-', label=label)
    ax2.set_xlabel('[M/H]')
    # alpha
    ax3 = plt.subplot(133)
    for mass_alpha00, mass_alpha04, label in zip(mass_alpha00_list, mass_alpha04_list, labels):
        ax3.plot([0, 0.4], [np.sum(mass_alpha00), np.sum(mass_alpha04)], 'o-', label=label)
    ax3.set_xlabel(r'[$\alpha$/Fe]')
    ax3.legend()




def plot_parameter_maps(mass_fraction_pixel_bin, age_grid_2d, metal_grid_2d, alpha_grid, reg_dim,
                        x_edges, y_edges, xlabel, ylabel, cmap, **kwargs):
    '''

    :param mass_fraction_pixel_bin:
    :param age_grid_2d:
    :param metal_grid_2d:
    :param reg_dim:
    :param x_edges:
    :param y_edges:
    :param xlabel:
    :param ylabel:
    :param cmap:
    :param kwargs:
    :return:
    '''

    x_edges = np.flip(x_edges)

    mass_weighted_age = np.zeros(mass_fraction_pixel_bin.shape[3:])
    mass_weighted_metal = np.zeros(mass_fraction_pixel_bin.shape[3:])
    mass_weighted_alpha = np.zeros(mass_fraction_pixel_bin.shape[3:])

    for i in range(mass_fraction_pixel_bin.shape[-2]):
        for j in range(mass_fraction_pixel_bin.shape[-1]):
            mass_fraction_pixel = mass_fraction_pixel_bin[:, :, :, i, j]
            if np.sum(mass_fraction_pixel) == 0:
                mass_weighted_age[i, j] = np.nan
                mass_weighted_metal[i, j] = np.nan
                mass_weighted_alpha[i, j] = np.nan
            else:
                age, metal, alpha = mean_age_metal_alpha(age_grid_2d, metal_grid_2d, alpha_grid, mass_fraction_pixel, reg_dim, True)
                mass_weighted_age[i, j] = age
                mass_weighted_metal[i, j] = metal
                mass_weighted_alpha[i, j] = alpha

    plt.figure(figsize=[18, 4])

    plt.subplot(131)
    ax1 = plt.gca()
    vmin = np.nanpercentile(mass_weighted_age, 0.5)
    vmax = np.nanpercentile(mass_weighted_age, 99.5)
    im1 = ax1.pcolormesh(x_edges, y_edges, mass_weighted_age, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('log(Age)')
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)

    plt.subplot(132)
    ax2 = plt.gca()
    vmin = np.nanpercentile(mass_weighted_metal, 0.5)
    vmax = np.nanpercentile(mass_weighted_metal, 99.5)
    im2 = ax2.pcolormesh(x_edges, y_edges, mass_weighted_metal, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('[M/H]')
    ax2.set_ylabel(ylabel)
    ax2.set_xlabel(xlabel)

    plt.subplot(133)
    ax3 = plt.gca()
    vmin = np.nanpercentile(mass_weighted_alpha, 0.5)
    vmax = np.nanpercentile(mass_weighted_alpha, 99.5)
    im3 = ax3.pcolormesh(x_edges, y_edges, mass_weighted_alpha, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('[Alpha/Fe]')
    ax3.set_ylabel(ylabel)
    ax3.set_xlabel(xlabel)

    ax1.invert_xaxis()
    ax2.invert_xaxis()
    ax3.invert_xaxis()

    plt.tight_layout()

##################################################################################





##################################################################################
# Degrading the datacube

# Modified PPXF method, make it can calculate flux_err

def cal_degrade_sig(FWHM_gal, FWHM_tem, dlam):
    FWHM_dif = np.sqrt(FWHM_gal ** 2 - FWHM_tem ** 2)
    sigma = FWHM_dif / 2.355 / dlam

    # if np.all((sigma==0)):
    #     sigma = sigma - 1 # this is the case when FWHM_gal == FWHM_tem

    return sigma



def degrade_spec_ppxf(spec, spec_err=None, sig=0, gau_npix=None):
    '''
    Modified from PPXF v8.1.0
    :param spec:
    :param spec_err:
    :param sig:
    :param gau_npix:
    :return:
    '''
    # This function can now skip the err if doesn't have
    if np.all(sig == 0):
        return spec, spec_err
    elif np.isscalar(sig):
        sig = np.zeros(spec.shape) + sig
    sig = sig.clip(1e-10)  # forces zero sigmas to have 1e-10 pixels

    if gau_npix == None:
        p = int(np.ceil(np.max(3*sig)))
    else:
        p = gau_npix
    m = 2 * p + 1  # kernel sizes
    x2 = np.linspace(-p, p, m) ** 2

    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n-m+j+1]

    gau = np.exp(-x2[:, None] / (2 * sig ** 2))
    gau /= np.sum(gau, 0)[None, :] # Normalize kernel

    conv_spectrum = np.einsum('ij,ij->j', a, gau)

    if np.all(spec_err) != None:
        a_e = np.zeros((m, n))
        for j in range(m):
            a_e[j, p:-p] = spec_err[j:n - m + j + 1]
        conv_spectrum_err = np.sqrt((a_e**2 * gau**2).sum(0))
    else:
        conv_spectrum_err = None

    return conv_spectrum, conv_spectrum_err



def process_degrade_cube(spec, spec_err, i, j, sigma, gau_npix=None):
    spec_ij_degraded, spec_err_ij_degraded = degrade_spec_ppxf(spec, spec_err, sigma, gau_npix)
    return spec_ij_degraded, spec_err_ij_degraded, i, j



def degrade_spec_cube(cube_flux, cube_err, FWHM_gal, FWHM_tem, ncpu, dlam, gau_npix=None):
    # If not using cube_err, set it tobe np.zeros(cube_flux.shape)
    t = clock()

    sigma = cal_degrade_sig(FWHM_gal, FWHM_tem, dlam)

    cube_flux_degraded = np.zeros(cube_flux.shape)
    cube_err_degraded = np.zeros(cube_err.shape)

    pool = Pool(processes=ncpu)
    results = []
    for i in range(cube_flux.shape[1]):
        for j in range(cube_flux.shape[2]):
            results.append(pool.apply_async(process_degrade_cube, (cube_flux[:, i, j], cube_err[:, i, j], i, j, sigma, gau_npix)))
    pool.close()
    pool.join()

    for result in results:
        spec_ij_degraded, spec_err_ij_degraded, i, j = result.get()
        cube_flux_degraded[:, i, j] = spec_ij_degraded
        cube_err_degraded[:, i, j] = spec_err_ij_degraded

    print('Elapsed time in generating spectra: %.2f s' % (clock() - t))

    return cube_flux_degraded, cube_err_degraded


##################################################################################


##################################################################################
# Calculate correlation coefficients (also works for arrays in different shapes)
def cal_pearsonr_corr(x1, values1, x2, values2):
    x1 = np.round(x1, 5)
    x2 = np.round(x2, 5)
    common_values = np.intersect1d(x1, x2)
    # print(common_values)
    mask_x1 = np.isin(x1, common_values)
    mask_x2 = np.isin(x2, common_values)
    corr = pearsonr(values1[mask_x1], values2[mask_x2])
    return corr




##################################################################################


##################################################################################
# Functions from other sources

def mean_age_metal_alpha(age_grid_2d, metal_grid_2d, alpha_grid, weights, reg_dim, quiet=False):
    '''
    This is modified from PPXF, compatible with single alpha case
    :param age_grid_2d:
    :param metal_grid_2d:
    :param alpha_grid:
    :param weights:
    :param reg_dim:
    :param quiet:
    :return:
    '''

    log_age_grids = np.log10(age_grid_2d) + 9
    metal_grids = metal_grid_2d

    weights_sum = np.sum(weights, axis=2)
    mean_log_age = np.sum(weights_sum * log_age_grids) / np.sum(weights_sum)
    mean_metal = np.sum(weights_sum * metal_grids) / np.sum(weights_sum)
    mean_alpha = np.sum(np.sum(weights * alpha_grid, axis=2)) / np.sum(weights_sum)

    if not quiet:
        print('Weighted <logAge> [yr]: %#.3g' % mean_log_age)
        print('Weighted <[M/H]>: %#.3g' % mean_metal)
        print('Weighted <[Alpha/Fe]>: %#.3g' % mean_alpha)

    return mean_log_age, mean_metal, mean_alpha




# def plot_pp(pp, reg_dim, age_grid_2d, metal_grid_2d):
#     '''
#     Plot the function of plot_massfraction_2d, plot_massfraction_1d, mean_age_metal_alpha together
#     :param pp:
#     :param reg_dim:
#     :param age_grid_2d:
#     :param metal_grid_2d:
#     :return:
#     '''
#     plt.figure(figsize=[10, 4])
#     ppxf.plot(pp)
#
#     ppweights = pp.weights.reshape(reg_dim)
#     ppweights = ppweights / np.sum(ppweights)
#
#     plot_massfraction_2d(ppweights_nov_r_noise_randnoise_1000[:, :, 0], ppweights[:, :, 1], age_grid_2d,
#                                metal_grid_2d, linear_age=True)
#
#     plot_massfraction_1d([mass_fraction_pixel_alpha00, ppweights[:, :, 0]],
#                                [mass_fraction_pixel_alpha04, ppweights[:, :, 1]],
#                                labels=['input', 'output'])
#
#     mean_age_metal_alpha(age_grid_2d, metal_grid_2d, ppweights, reg_dim, False)

##################################################################################
