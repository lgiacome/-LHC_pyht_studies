
from tqdm import tqdm
import numpy as np
np.random.seed(42)

from scipy.constants import m_p, c, e

import matplotlib.pyplot as plt

from LHC import LHC
from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.particles.slicing import UniformChargeSlicer

def py_ht_wake_sim(n_macroparticles, n_slices, fig_path, use_damper=False):
    
    n_turns = 1000
    Q_x = 64.28
    Q_y = 59.31
    Q_s = 0.0020443

    C = 26658.883
    R = C / (2.*np.pi)

    alpha_x = 0.
    alpha_y = 0.
    beta_x = 66.0064
    beta_y = 71.5376
    alpha_0 = [0.0003225]

    machine_configuration = 'LHC_6.5TeV_collision_2016'

    chroma = 0
    i_oct = 0

    def get_nonlinear_params(chroma, i_oct, p0=6.5e12*e/c):
        '''Arguments:
            - chroma: first-order chromaticity Q'_{x,y}, identical
              for both transverse planes
            - i_oct: octupole current in A (positive i_oct means
              LOF = i_oct > 0 and LOD = -i_oct < 0)
        '''
        # factor 2p0 is PyHEADTAIL's convention for d/dJx instead of
        # MAD-X's convention of d/d(2Jx)
        app_x = 2 * p0 * 27380.10941 * i_oct / 100.
        app_y = 2 * p0 * 28875.03442 * i_oct / 100.
        app_xy = 2 * p0 * -21766.48714 * i_oct / 100.
        Qpp_x = 4889.00298 * i_oct / 100.
        Qpp_y = -2323.147896 * i_oct / 100.
        return {
            'app_x': app_x,
            'app_y': app_y,
            'app_xy': app_xy,
            'Qp_x': [chroma,],# Qpp_x],
            'Qp_y': [chroma,],# Qpp_y],
            # second-order chroma commented out above!
        }
    
    machine = LHC(n_segments=1,
                  machine_configuration=machine_configuration,
                  **get_nonlinear_params(chroma=chroma, i_oct=i_oct))

    
    epsn_x = 3.e-6 # normalised horizontal emittance
    epsn_y = 3.e-6 # normalised vertical emittance
    sigma_z = 1.2e-9 * machine.beta*c/4. # RMS bunch length in meters
    intensity = 1.1e11

    bunch = machine.generate_6D_Gaussian_bunch_matched(
            n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z)

    slicer_for_wakefields = UniformChargeSlicer(
            n_slices, z_cuts=(-8*sigma_z, 8*sigma_z))

    wakefile = 'wakeforhdtl_PyZbase_Allthemachine_6800GeV_B1_2021_TeleIndex1_updatedMOs_updatedMo_on_MoC_wake'

    data_wake = np.genfromtxt(wakefile + '.dat')
    data_wake_noquad = data_wake[:,[0,1,2]]

    np.savetxt(wakefile+"_no_quad.dat", data_wake_noquad, delimiter='\t', fmt='%15.10e' ,newline='\n')

    wake_table = WakeTable(wakefile+"_no_quad.dat",
                            ['time', 'dipole_x', 'dipole_y',
                              # 'quadrupole_x', 'quadrupole_y',
                             #'dipole_xy', 'dipole_yx',
                            ])

    wake_field = WakeField(slicer_for_wakefields, wake_table)

    machine.one_turn_map.append(wake_field)

    if use_damper:
        damping_rate = 100 # in turns
        # create transverse feedback instance
        damper = TransverseDamper(damping_rate, damping_rate)

    #machine.one_turn_map.append(damper)
    # prepare empty arrays to record transverse moments
    x = np.empty(n_turns, dtype=float)
    xp = np.empty_like(x)
    y = np.empty_like(x)
    yp = np.empty_like(x)

    # actual tracking
    t = np.arange(n_turnsfig_path)
    for i in tqdm(t):
        for m in machine.one_turn_map:
            m.track(bunch)
            x[i] = bunch.mean_x()
            xp[i] = bunch.mean_xp()
            y[i] = bunch.mean_y()
            yp[i] = bunch.mean_yp()

    # evaluation of dipolar bunch moments
    #j_x = np.sqrt(x**2 + (beta_x * xp)**2)
    #exponent_x, amplitude_x = np.polyfit(t, np.log(2 * j_x), 1)

    #j_y = np.sqrt(y**2 + (beta_y * yp)**2)
    #exponent_y, amplitude_y = np.polyfit(t, np.log(2 * j_y), 1)

    #print ('Horizontal reconstructed damping time: {:.3f} turns'.format(1/exponent_x))
    #print ('Vertical reconstructed damping time: {:.3f} turns'.format(1/exponent_y))

    plt.figure()
    plt.plot(t, x, label='horizontal dipolar moment', alpha=0.8)
    plt.plot(t, y, label='vertical dipolar moment', alpha=0.8)
    ylim = plt.ylim()
    plt.legend(loc=0);
    plt.xlabel('Turns')
    plt.ylabel('Centroid motion [m]')
    plt.twinx()
    #plt.plot(t, np.exp(amplitude_x + (exponent_x*t)) / 2., color='turquoise', label='horizontal fit')
    #plt.plot(t, -np.exp(amplitude_x + (exponent_x*t)) / 2., color='red', label='vertical fit')
    plt.ylim(ylim)
    plt.legend(loc=4)
    plt.yticks([]);
    plt.title('N_mp: ' + str(n_mpacroparticles) + 'n_sl: ' + str(n_slices))
    plt.savefig(fig_path)
