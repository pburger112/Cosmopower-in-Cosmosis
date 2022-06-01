from builtins import str
import os
from cosmosis.datablock import names, option_section
import sys
import traceback
from scipy.interpolate import CubicSpline
from scipy.interpolate import RectBivariateSpline 
import cosmopower as cp


import cosmopower
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to use CPU if GPU is avalible otherwise the GPU memory runs out of memory. It also does not slower the predtiction.

# These are pre-defined strings we use as datablock
# section names
cosmo = names.cosmological_parameters
distances = names.distances


def get_optional_params(block, section, names):
    """Get values from a datablock from a list of names.

    If the entries of names are tuples or lists of length 2, they are assumed
    to correspond to (cosmosis_name, output_name), where cosmosis_name is the 
    datablock key and output_name the params dict key."""
    params = {}
    for name in names:
        cosmosis_name = name
        output_name = name
        if isinstance(name, (list, tuple)):
            if len(name) == 2 and isinstance(name[1], str):
                # Output name specified
                output_name = name[1]
                cosmosis_name = name[0]
        if block.has_value(section, cosmosis_name):
            params[output_name] = block[section, cosmosis_name]
    return params


def setup(options):

    config = {
        'kmax': options.get_double(option_section, 'kmax', default=10.0),
        'kmin': options.get_double(option_section, 'kmin', default=1e-5),
        'nk': options.get_int(option_section, 'nk', default=200),
        'zmax': options.get_double(option_section, 'zmax', default=3.0),
        'zmin': options.get_double(option_section, 'zmin', default=0.0),
        'nz': options.get_int(option_section, 'nz', default=300),
        'use_specific_k_modes': options.get_bool(option_section, 'use_specific_k_modes', default=False),
        'feedback_model': options.get_string(option_section, 'feedback_model', default='mead2016'),
    }

    assert config['feedback_model'] in ['mead2016', 'mead2020'], "Choose feedback_model from mead2016, mead2020"

    for _, key in options.keys(option_section):
        if key.startswith('cosmopower_'):
            config[key] = block[option_section, key]

    # Create the object that connects to cosmopower
    # load pre-trained NN model: maps cosmological parameters to linear log-P(k)
    path_2_trained_emulator = options.get_string(option_section, 'path_2_trained_emulator')
    config['lin_matter_power_cp'] = cp.cosmopower_NN(restore=True, 
                            restore_filename=os.path.join(path_2_trained_emulator+'/trained_models/CP_paper/PK/PKLIN_NN'))

    config['nl_matter_power_boost_cp'] = cp.cosmopower_NN(restore=True, 
                            restore_filename=os.path.join(path_2_trained_emulator+'/trained_models/CP_paper/PK/PKNLBOOST_NN'))

    # Return all this config information
    return config


def get_cosmopower_inputs(block, z, nz, model):

    # Get parameters from block and give them the
    # names and form that class expects

    if block.has_value(cosmo, 'A_s'):
        A_s = block[cosmo, 'A_s']
        ln_a_s_e10 = np.log(A_s*1e10)
        block[cosmo, 'ln_a_s_e10'] = ln_a_s_e10
    elif block.has_value(cosmo, 'ln_a_s_e10'):
        ln_a_s_e10 = block[cosmo, 'ln_a_s_e10']
        A_s = np.exp(ln_a_s_e10) * 1e-10
        block[cosmo, 'A_s'] = A_s
    else:
        raise ValueError("Can't find A_s or ln(10^10 * A_s) in datablock?")

    params_lin = {
        'ln10^{10}A_s':[ln_a_s_e10]*nz,
        'n_s':       [block[cosmo, 'n_s']]*nz,
        'h':         [block[cosmo, 'h0']]*nz,
        'omega_b':   [block[cosmo, 'ombh2']]*nz,
        'omega_cdm': [block[cosmo, 'omch2']]*nz,
        'z':         z
    }

    if model == 'mead2016':
        if block.has_value(names.halo_model_parameters, 'eta0') or \
           block.has_value(names.halo_model_parameters, 'eta') or \
           block.has_value(names.halo_model_parameters, 'eta_bary'):
            key = [k for s, k in block.keys() if s == names.halo_model_parameters and k.startswith('eta')]
            assert len(key) == 1, f"More than 1 value for eta_baryon in datablock? {key}"
            eta_bary = block[names.halo_model_parameters, key[0]]
        else:
            eta_bary = None
        if block.has_value(names.halo_model_parameters, 'A') or \
           block.has_value(names.halo_model_parameters, 'A_bary'):
            key = [k for s, k in block.keys() if s == names.halo_model_parameters and (k.startswith('A') or k.startswith('a'))]
            assert len(key) == 1, f"More than 1 value for A_baryon in datablock? {key}"
            A_bary = block[names.halo_model_parameters, key[0]]
            if eta_bary is None:
                eta_bary = 0.98 - 0.12 * A_bary # 1-parameter HMCode 2016 coefficients a la K1000
        else:
            A_bary = 2.32 # set A to Pierre's default from testing
            eta_bary = 0.76 # set eta to Pierre's default from testing
            print(f"Halo model {model} parameters not given; setting A_bary={A_bary}, eta_bary={eta_bary}")

    elif model == 'mead2020':
        if block.has_value(names.halo_model_parameters, 'logT_AGN'):
            logT_AGN = block[names.halo_model_parameters, 'logt_agn']
        else:
            logT_AGN = 7.8
            print(f"Halo model {model} parameter not given; setting logT_AGN={logT_AGN}")

    params_boost = {
        'ln10^{10}A_s':  [ln_a_s_e10]*nz,
        'n_s':           [block[cosmo, 'n_s']]*nz,
        'h':             [block[cosmo, 'h0']]*nz,
        'omega_b':       [block[cosmo, 'ombh2']]*nz,
        'omega_cdm':     [block[cosmo, 'omch2']]*nz,
        'z':             z,
#        'c_min':         [A_bary]*nz,
#        'eta_0':         [eta]*nz
    }

    if model == 'mead2016':
        params_boost['c_min'] = [A_bary]*nz
        params_boost['eta_0'] = [eta_bary]*nz
    elif model == 'mead2020':
        params_boost['logt_agn'] = [logT_AGN]*nz

    return params_lin, params_boost


def execute(block, config):

    h0 = block[cosmo, 'h0']

    nz = config['nz']
    z = np.linspace(config['zmin'], config['zmax'], nz)

    block[distances, 'z'] = z
    block[distances, 'nz'] = nz

    #use k modes for cosmopower
    k = config['lin_matter_power_cp'].modes
    nk = len(k)

    params_lin,params_boost = get_cosmopower_inputs(block, z, nz, config['feedback_model'])
    P_lin = config['lin_matter_power_cp'].ten_to_predictions_np(params_lin).T
    P_nl = P_lin*config['nl_matter_power_boost_cp'].ten_to_predictions_np(params_boost).T

    if(config['use_specific_k_modes']):
        k_new = np.logspace(np.log10(config['kmin']), np.log10(config['kmax']),num=config['nk'])

        # P_lin_spline = RectBivariateSpline(k,z,P_lin)
        # P_nl_spline = RectBivariateSpline(k,z,P_nl)
        # P_lin = P_lin_spline(k_new,z)
        # P_nl = P_nl_spline(k_new,z)

        P_lin_new = np.zeros(shape=(len(k_new),nz))
        P_nl_new = np.zeros(shape=(len(k_new),nz))
        for i in range(nz):
            P_lin_spline = CubicSpline(k,P_lin[:,i])
            P_nl_spline = CubicSpline(k,P_nl[:,i])
            P_lin_new[:,i] = P_lin_spline(k_new)
            P_nl_new[:,i] = P_nl_spline(k_new)
        P_lin = P_lin_new
        P_nl = P_nl_new

        k = k_new

    # Save matter power as a grid
    block.put_grid("matter_power_lin", "k_h", k / h0, "z", z, "p_k", P_lin * h0**3)
    block.put_grid("matter_power_nl", "k_h", k / h0, "z", z, "p_k", P_nl * h0**3)

    return 0

