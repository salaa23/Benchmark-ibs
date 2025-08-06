'''
Author: Salah Eddine Feddaoui Dellalou
Date: 23-04-2025
Description: This file is the benchmark notebook for Xsuite and mbtrack2 tracking codes
Date de dernière modification: 23-04-2025
'''

import json, sys, os, warnings, random, string, shutil
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import h5py
import time
import scipy.constants as cons
from tqdm import tqdm
import scipy.constants as cons
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from tqdm.notebook import trange
import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo
import xcoll as xc
import xwakes as xw

from mbtrack2 import Synchrotron, Electron
from mbtrack2.utilities import Optics
from mbtrack2.impedance.wakefield import WakeField
from mbtrack2.tracking import LongitudinalMap, SynchrotronRadiation, TransverseMap
from mbtrack2.tracking import IntrabeamScattering
from mbtrack2.tracking import Beam, Bunch, WakePotential
from mbtrack2.tracking import RFCavity, SynchrotronRadiation
from mbtrack2.tracking.monitors import BunchMonitor, WakePotentialMonitor
from mbtrack2.tracking.feedback import FIRDamper #ExponentialDamper
from mbtrack2.tracking import TransverseMapSector, transverse_map_sector_generator

import at

class beam_param:
    """
    Class to define the beam self.parameters from input file
    """
    def __init__(self, mode, **kw):
        if mode == None:
            print('No mode entered -> default = z')
            mode = 'z'
        self.filename()
        self.read_param(mode)
        self.set_param()

    def filename(self):
        # parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
        current_path = os.getcwd()
        input_dir = current_path + '/_inputs/'
        self.optics_file = input_dir + 'Booster_parameter_table.json'
        self.filename = self.optics_file
        
    def read_param(self, mode):
        inputs_tab = json.load(open(self.filename))
        print(inputs_tab.keys())

        # define variables
        self.C = inputs_tab['C']['value'] # circumference [m]
        self.Np = inputs_tab['Np'][mode] # number of particles per bunch
        self.Nb = inputs_tab['Nb'][mode] # number of bunches
        self.Etot = inputs_tab['E']['injection'] # energy at injection [eV]
        self.epsnx = inputs_tab['bunch']['epsnx']['value'] # normalised horizontal emittance [m]
        self.epsny = inputs_tab['bunch']['epsny']['value'] # normalised horizontal emittance [m]
        self.sigmaz = inputs_tab['bunch']['sigmaz']['value'] # bunch length at injection [m]
        self.sigmae = inputs_tab['bunch']['sigmae']['value'] # energy spread at injection
        self.Qx = inputs_tab['optics']['Qx'][mode] # horizontal tune
        self.Qy = inputs_tab['optics']['Qy'][mode] # vertical tune
        self.chix = inputs_tab['optics']['chix'][mode] # horizontal chromaticity
        self.chiy = inputs_tab['optics']['chiy'][mode] # horizontal chromaticity
        self.alpha = inputs_tab['optics']['alpha'][mode] # momentum compaction
        self.I2 = inputs_tab['optics']['I2'][mode] # 2nd synchrotron integral
        self.I3 = inputs_tab['optics']['I3'][mode] # 3rd synchrotron integral
        self.I5 = inputs_tab['optics']['I5'][mode] # 5th synchrotron integral
        self.I4 = 5.920335834e-09 # for testing, calculated using MAD-X
        self.I6 = 0 # 6th synchrotron integral
        self.dpt = inputs_tab['optics']['dpt'][mode] # maximum energy acceptance at injection
        self.damp_xy = inputs_tab['optics']['damp_xy'][mode] # transverse damping time at injection energy
        self.damp_s = inputs_tab['optics']['damp_s'][mode] # longitudinal damping time at injection energy
        self.coupling = inputs_tab['optics']['coupling'][mode] # horizontal vertical coupling
        self.Cq = 3.8319e-13
        self.Cgamma = 8.846e-05
        self.Erest = 510998.9499961642 # rest energy [eV]
        self.Egain = 0 # energy gain [eV]
        self.freq = inputs_tab['RF']['RF_freq'][mode] # RF frequency [Hz]
        self.Vtot = inputs_tab['RF']['Vtot'][mode] # total cavities voltage [eV]
        self.Qs = inputs_tab['RF']['Qs_inj'][mode] # synchronous tune at injection
        self.phi_s = inputs_tab['RF']['phis_inj'][mode] # synchronous phase at injection
        
    def set_param(self):
        self.lambdaRF = cons.c/self.freq # RF wavelength
        self.frev = cons.c/self.C # revolution frequency
        self.h = self.freq/self.frev # Schottky number
        self.U0 = self.Cgamma*(self.Etot*1e-9)**4/2/np.pi*self.I2*1e9 # Synchrotron energy loss per turn
        self.gamma = self.Etot / self.Erest
        self.sig_e_eq = np.sqrt(self.Cq*self.gamma**2*self.I3/(2*self.I2))#*(self.Etot*1e-9)**2)
        self.emit_eq = self.Cq * self.gamma**2 * self.I5 / self.I2 # geometrical equilibrium emittance
        self.epsnx_eq = self.emit_eq * self.gamma # normalized equilibrium emittance X
        self.epsny_eq = self.epsnx_eq * 2e-3 # normalized equilibrium emittance Y
        self.eta = 1/self.gamma**2-self.alpha # momentum compaction

class _collective_:
    """
    """

    def __init__(self,_inputs=None, **kw):
        print ('== loading inputs ==')
        self._set_inputs()
        print(_inputs)
        if _inputs is not None:
            for key, value in _inputs.items():
                setattr(self, key, value)
                print(key, ":", value)
        else:
            print('Some input parameters are missing')

        self._inputs = _inputs
        print("setting defaults ...")
        self._set_defaults()
        print("setting line ...")
        self._set_line()
        print("setting map ...")
        self._set_map()
        print("setting lattice ...")
        self.v24()
        print("setting bunch ...")
        self._set_bunch()
        print("setting profile ...")
        self._profile_dict()
        print("saving ring file")
        self._save_init_ring()
        print("setting cavity ...")
        self._set_cavity()
        print("setting ibs ...")
        self._set_ibs()
        print("setting radiation ...")
        self._set_radiation()
        print("setting monitors ...")
        self._set_monitors()
        if self._codes == "mbtrack2":
            print("Tracking with mbtrack2 ...")
            self._run_mbtrack2()
            # self._get_Tibs()
        elif self._codes == "xsuite":
            print("tracking with xsuite ...")
            self._run_xsuite()
        else:
            print("Tracking with mbtrack2 ...")
            self._run_mbtrack2()
            print("tracking with xsuite ...")
            self._run_xsuite()

    def _set_inputs(self):
        """Set the default inputs for the benchmark"""
        self.n_macroparticles = 10
        self.n_turns = 8
        self.comment = f"_init_variable"
        self.modelname = 'CIMP'
        self.n_ibs_slices = 100
        self.ibs_formalism = "analytical" #or kinetic, anything else and ibs will be kinetic
        self.bunch_style = "unmatched"
        self.ibs_toggle = False
        self.load_lattice = False
        self.location = 0.0
        self._codes = "mbtrack2"
        self.slicing = 1000
        self.distributed = False
        self.sectormap = False
        self.sectors = 1

    def _set_defaults(self):
        """Set the default parameters for the benchmark"""
        self.parameter = beam_param(mode=None)
        self.temps = time.strftime("%y%m%d_%H%M%S", time.localtime())
        self.job_id = os.environ.get("SLURM_JOB_ID")
        if self.job_id is None:
            self.job_id = "_"
        self.Np = 0.0
        self.current_path = os.getcwd()
        self.input_dir = self.current_path + '/_inputs/'
        self.output_dir = self.current_path + '/_outputs/'
        self.optics_file = self.input_dir + 'heb_ring_withcav.json' # optics file
        self.lattice =  self.input_dir + 'V24_nocav.mat'
        self.mass =xp.ELECTRON_MASS_EV # mass of the particle
        self.q0 = -1 # charge of the particle
        self.mbtrack2_output = self.output_dir + self.modelname+"_"+self.temps+"_"+self.job_id
        self.xsuite_output = self.output_dir + "xsuite_" + str(self.temps) + "_" +  self.job_id + self.comment + ".json"
        self.chunk_size = 10
        self.n_chunk = int(np.ceil(self.n_turns/self.chunk_size))
        self.file_name_m = self.output_dir + self.modelname + "_" + str(self.temps) +"_"+ self.job_id + self.comment
        self.input_file_name_m = self.output_dir +"input_"+ self.modelname + "_" + str(self.temps) +"_"+ self.job_id + self.comment
        self.bunch_current = self.parameter.Np * cons.elementary_charge * cons.c / self.parameter.C
        self.context = xo.ContextCpu(omp_num_threads="auto")

    def v24(self):
        """
        read lattice .mat format (loaded with AT) and returns ring object
        Takes self.parameters from beam_param class

        Returns
        -------
        ring : Synchrotron object

        """    
        
        h = self.parameter.h
        particle = Electron()
        tau = np.array([self.parameter.damp_xy, self.parameter.damp_xy, self.parameter.damp_s])
        sigma_0 = self.parameter.sigmaz/cons.c
        sigma_delta = self.parameter.sigmae
        emit = np.array([self.parameter.epsnx /self.parameter.gamma, self.parameter.epsny /self.parameter.gamma])
        f0 = self.parameter.frev
        f1 = self.parameter.freq
        U0 = self.parameter.U0
        ac = self.parameter.alpha
        chro = np.array([self.parameter.chix, self.parameter.chiy])
        gamma = self.parameter.gamma
        tune = np.array([self.parameter.Qx, self.parameter.Qy])
        local_beta = np.array([self.betax, self.betay])
        local_alpha = np.array([self.alphax, self.alphay])
        local_dispersion = np.array([self.dx, self.dpx, self.dy, self.dpy])
        L = self.parameter.C
        E0 = self.parameter.Etot
        if self.load_lattice == True:
            optics = Optics(lattice_file=self.lattice, n_points=len(self.tw.betx))
            self.ring = Synchrotron(h, optics, particle, tau=tau,emit=emit, 
                    sigma_0=sigma_0, sigma_delta=sigma_delta, U0=U0, ac=ac, chro=chro, gamma=gamma, f1=f1, 
                    f0=f0)#, dx=dx, dy=dy, betax=betax, betay=betay, alphax=alphax, alphay=alphay)
        else:
            optics = Optics(local_alpha=local_alpha,local_beta=local_beta,local_dispersion=local_dispersion, n_points=1e4)
            self.ring = Synchrotron(h, optics, particle, tau=tau,emit=emit, 
               sigma_0=sigma_0, sigma_delta=sigma_delta, f0=f0, f1=f1, U0=U0, ac=ac, chro=chro, gamma=gamma, L=L, E0=E0,tune=tune)
        # print(self.ring.L)

    def _set_line(self):
        """Set the line for the benchmark"""
        #for xsuite
        self.line = xt.Line.from_json(self.optics_file)
        self.particle_ref = xp.Particles(mass0=self.mass, q0=self.q0, gamma0=self.parameter.gamma)
        self.line.particle_ref = self.particle_ref

        self.line.slice_thick_elements( slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(2)), 
        xt.Strategy(slicing=xt.Teapot(3), element_type=xt.Bend), 
        xt.Strategy(slicing=xt.Teapot(5), element_type=xt.Quadrupole), 
        xt.Strategy(slicing=xt.Teapot(3), element_type=xt.Sextupole)])

        self.line.build_tracker()
        self.line.configure_radiation(model='mean')
        self.environment = self.line.env
        self.tw = self.line.twiss(method="6d",particle_ref=self.particle_ref ,eneloss_and_damping=True)
        self.C =  self.tw.s[-1] # circumference
        self.qx =  self.tw.qx # horizontal tune
        self.qy =  self.tw.qy # vertical tune
        self.dqx =  self.tw.dqx # horizontal chromaticity
        self.dqy =  self.tw.dqy # vertical chromaticity
        self.eneloss_turn = self.tw.eneloss_turn
        self.df = self.tw.to_pandas()
        self.betax_0 = self.parameter.C / (2 * np.pi * self.tw.qx)
        # self.betax_0 = self.location #np.average(self.tw.betx) #to be defined to optimize kick position
        self.df["distance"] = abs(self.df.betx - self.betax_0)
        smallest = self.df.nsmallest(10, 'distance')
        smallest = smallest.drop('distance', axis=1)
        self.index = smallest.dx.idxmin()
        self.alphax = smallest.alfx[self.index]
        self.alphay = smallest.alfy[self.index]
        self.dx = smallest.dx[self.index]
        self.dpx = smallest.dpx[self.index]
        self.dy = smallest.dy[self.index]
        self.dpy = smallest.dpy[self.index]
        self.betax = smallest.betx[self.index]
        self.betay = smallest.bety[self.index]
        self.chosen_params = {'betax_0': float(self.betax_0)  ,'alphax': float(self.alphax), 'alphay': float(self.alphay), 'dx': float(self.dx), 'dy': float(self.dy), 'dpx': float(self.dpx), 'dpy':float(self.dpy), 'betx': float(self.betax), 'bety':float(self.betay), 'index': float(self.index)}
        print('Extracting detuning coefficients')
        det_= self.line.get_amplitude_detuning_coefficients(
            nemitt_x=self.parameter.epsnx, 
            nemitt_y=self.parameter.epsny, 
            num_turns=500, 
            a0_sigmas=0.01, 
            a1_sigmas=0.1, 
            a2_sigmas=0.2)
        self.det_xx = det_['det_xx']
        self.det_yy = det_['det_yy']
        self.det_xy = det_['det_xy']
        self.det_yx = det_['det_yx']
        self.damping_rate_emit_h = 2 * self.tw.damping_constants_turns[0] # horizontal damping rate
        self.damping_rate_emit_v = 2 * self.tw.damping_constants_turns[1] # horizontal damping rate
        self.damping_rate_emit_zeta = 2 * self.tw.damping_constants_turns[2] # longitudinal damping rate
        self.gauss_noise_ampl_px = 2 * np.sqrt(self.tw.eq_gemitt_x  * self.damping_rate_emit_h / self.tw.betx)#replaced tw.betx[self.index]
        self.gauss_noise_ampl_x = 0
        self.gauss_noise_ampl_py = 2 * np.sqrt(self.tw.eq_gemitt_y  * self.damping_rate_emit_h / self.tw.bety)#replaced tw.bety[self.index]
        self.gauss_noise_ampl_y = 0.
        self.gauss_noise_ampl_delta = 2 * np.sqrt(self.tw.eq_gemitt_zeta * (1/2) * self.damping_rate_emit_zeta / self.tw.bets0)

        self.gauss_noise_ampl_delta = np.average(self.gauss_noise_ampl_delta)
        self.gauss_noise_ampl_x = np.average(self.gauss_noise_ampl_x)
        self.gauss_noise_ampl_px = np.average(self.gauss_noise_ampl_px)
        self.gauss_noise_ampl_y = np.average(self.gauss_noise_ampl_y)
        self.gauss_noise_ampl_py = np.average(self.gauss_noise_ampl_py)
    
    def _set_map(self):
        """Set the map for the benchmark"""
        self.map =  xt.LineSegmentMap(
                    length=self.C,
                    qx=self.qx,
                    qy=self.qy,
                    dqx=self.dqx,
                    dqy=self.dqy,
                    momentum_compaction_factor=self.parameter.alpha,
                    betx=self.betax,
                    bety=self.betay,
                    alfx=self.alphax,
                    alfy=self.alphay,
                    dx = self.dx,
                    dpx = self.dpx,
                    dy = self.dy,
                    dpy = self.dpy,
                    det_xx=self.det_xx,
                    det_xy=self.det_xy,
                    det_yx=self.det_yx,
                    det_yy=self.det_yy,  
                    damping_rate_x=  self.damping_rate_emit_h,
                    damping_rate_y=  self.damping_rate_emit_v,
                    # In longitudinal all damping goes on the momentum
                    damping_rate_zeta=0,
                    damping_rate_pzeta=self.damping_rate_emit_zeta,
                    gauss_noise_ampl_px=self.gauss_noise_ampl_px,
                    gauss_noise_ampl_py=self.gauss_noise_ampl_py,
                    gauss_noise_ampl_pzeta=self.gauss_noise_ampl_delta,
                    energy_increment           = -1 * self.eneloss_turn,#M.U0,
                    longitudinal_mode          = 'nonlinear', # needs to be commented for 4D tracking + uncomment betas/qs
                    voltage_rf                 = [self.parameter.Vtot], # needs to be commented for 4D tracking + uncomment betas/qs
                    frequency_rf               = [self.parameter.freq], # needs to be commented for 4D tracking + uncomment betas/qs
                    lag_rf                     = [180 - np.rad2deg(np.arcsin(self.eneloss_turn/self.parameter.Vtot))], # needs to be commented for 4D tracking + uncomment betas/qs
                )#180 - np.rad2deg(np.arcsin(self.eneloss_turn/self.parameter.Vtot))
        ring_map_no_excit = self.map.copy()
        self.environment.elements['ring_map'] = self.map
        ring_map_no_excit.gauss_noise_matrix = 0
        self.ring_map = xt.Line(elements=[ring_map_no_excit])
        self.ring_map._needs_rng = True
        self.ring_map.particle_ref = self.particle_ref.copy()
        tw_check = self.ring_map.twiss()
        self.ring_map.correct_trajectory(twiss_table=self.ring_map.twiss4d())

    def _set_bunch(self):
        """Set the bunch for the benchmark"""
        #for mbtrack2
        particle = Electron()

        self.mybunch = Bunch(
            self.ring, mp_number=self.n_macroparticles, current=self.bunch_current, track_alive=True)
        np.random.seed(42)
        self.rng = np.random.RandomState(42)
        self.mybunch.init_gaussian()
        #for xsuite
        self.x_norm = self.rng.randn(self.n_macroparticles)
        self.px_norm = self.rng.randn(self.n_macroparticles)
        self.y_norm = self.rng.randn(self.n_macroparticles)
        self.py_norm = self.rng.randn(self.n_macroparticles)
        self.zeta = self.parameter.sigmaz * (self.rng.randn(self.n_macroparticles))
        self.delta = self.parameter.sigmae * (self.rng.randn(self.n_macroparticles))
        if self.load_lattice is True:
            if self.bunch_style == "matched":
                self.particles = xp.generate_matched_gaussian_bunch(
                    num_particles=self.n_macroparticles,
                    nemitt_x=self.parameter.epsnx,
                                nemitt_y=self.parameter.epsny,
                                sigma_z=self.parameter.sigmaz,
                                total_intensity_particles=self.parameter.Np,
                                line=self.line, _context = self.context)
            else:
                self.particles = self.line.build_particles(
                    _context=self.context, 
                    _buffer=None, 
                    _offset=None,
                    particle_ref=self.particle_ref,
                    zeta=self.zeta, 
                    delta=self.delta,
                    x_norm=self.x_norm, 
                    px_norm=self.px_norm,
                    y_norm=self.y_norm, 
                    py_norm=self.py_norm,
                    nemitt_x=self.parameter.epsnx, 
                    nemitt_y=self.parameter.epsny,
                    weight=self.parameter.Np/self.n_macroparticles)
        else:
            if self.bunch_style == "matched":
                self.particles = xp.generate_matched_gaussian_bunch(
                    num_particles=self.n_macroparticles,
                    nemitt_x=self.parameter.epsnx,
                                nemitt_y=self.parameter.epsny,
                                sigma_z=self.parameter.sigmaz,
                                total_intensity_particles=self.parameter.Np,
                                line=self.ring_map, _context = self.context)
            else:
                self.particles = self.ring_map.build_particles(
                    _context=self.context, 
                    _buffer=None, 
                    _offset=None,
                    particle_ref=self.particle_ref,
                    zeta=self.zeta, 
                    delta=self.delta,
                    x_norm=self.x_norm, 
                    px_norm=self.px_norm,
                    y_norm=self.y_norm, 
                    py_norm=self.py_norm,
                    nemitt_x=self.parameter.epsnx, 
                    nemitt_y=self.parameter.epsny,
                    weight=self.parameter.Np/self.n_macroparticles)

    def _profile_dict(self):
        """Create a profile dictionary for the benchmark"""
        x = np.array(self.particles.x)
        px = np.array(self.particles.px)
        y = np.array(self.particles.y)
        py = np.array(self.particles.py)
        z = np.array(self.particles.zeta)
        delta = np.array(self.particles.delta)
        x_m = np.array(self.mybunch["x"])
        px_m = np.array(self.mybunch["xp"])
        y_m = np.array(self.mybunch["y"])
        py_m = np.array(self.mybunch["yp"])
        z_m = np.array(self.mybunch["tau"]) * cons.c
        delta_m = np.array(self.mybunch["delta"])
        dims = {"xsuit_x":x, "mbtrack_x":x_m, "xsuit_y":y, "mbtrack_y":y_m, "xsuit_z":z, "mbtrack_z":z_m, "xsuit_px":px,
         "mbtrack_px":px_m, "xsuit_py":py, "mbtrack_py":py_m, "xsuit_delta":delta, "mbtrack_delta":delta_m }
        def get_profile(value):
            n_bin = 75
            bin_min = value.min()
            bin_min = min(bin_min * 0.99, bin_min * 1.01)
            bin_max = value.max()
            bin_max = max(bin_max * 0.99, bin_max * 1.01)
            bins = np.linspace(bin_min, bin_max, n_bin + 1)
            center = (bins[1:] + bins[:-1]) / 2
            sorted_index = np.searchsorted(bins, value, side="left")
            sorted_index -= 1
            profile = np.bincount(sorted_index, minlength=n_bin)
            return center, profile, sorted_index, bins
        profile_dict = {}
        # dim_keys = ["x","y","px","py","z","delta"]
        for key,val in dims.items():
            center, profile, sorted_index, bins = get_profile(val)
            profile = list(profile)
            center = list(center)
            profile = [int(i) if isinstance(i, np.integer)
                    else float(i) if isinstance(i, np.floating)
                    else i
                    for i in profile]
            center = [int(i) if isinstance(i, np.integer)
                else float(i) if isinstance(i, np.floating)
                else i
                for i in center]
            profile_dict[key] = {"center": center, "profile": profile}
        self.profile_json = json.dumps(profile_dict)
        self.vars_parameter = vars(self.parameter)
        for key ,val in self.vars_parameter.items():
            if isinstance(val, np.floating):
                val = float(val)
        

    def _save_init_ring(self):
        """
        save the initial parameters for the ring and the initial profile into an hdf5 file
        filename is _input followed by the same name as the tracker
        """
        dict_ring = vars(self.ring)
        dict_optics = vars(self.ring.optics)
        dict_particle = vars(self.ring.particle)
        if self.ring.optics.use_local_values is True:
            with h5py.File(self.input_file_name_m+".hdf5", "w") as f:
                group_ring = f.create_group("ring")
                for k,v in dict_ring.items():
                    if isinstance(v, (float, list, int, bool, np.ndarray)):
                        group_ring.create_dataset(k, data=v)
                group_optics = f.create_group("optics")
                for k,v in dict_optics.items():
                    if isinstance(v, (float, list, int, bool, np.ndarray)):
                        group_optics.create_dataset(k, data=v) 
                group_particle = f.create_group("particle")
                for k,v in dict_particle.items():
                    if isinstance(v, (float, list, int, bool, np.ndarray)):
                        group_particle.create_dataset(k, data=v)
                group_profile = f.create_group("profile")
                group_profile.create_dataset('profiles', data=self.profile_json)
            print("inputs file created")
        else:
            with h5py.File(self.input_file_name_m+".hdf5", "w") as f:
                group_profile = f.create_group("profile")
                group_profile.create_dataset('profiles', data=self.profile_json)
            print("Profiles file created!")
        with h5py.File(self.input_file_name_m+".hdf5", "w") as f:
            group_params = f.create_group("parameters")
            self.chosen_params_json = json.dumps(self.chosen_params)
            group_params.create_dataset('chosen_params', data=self.chosen_params_json)
            # group_params.create_dataset('init_params', data=self.vars_parameter)
            # group_params.create_dataset('inputs', data=self._inputs)
            group_inputs = f.create_group("inputs")
            for key, value in self._inputs.items():
                if isinstance(value, str):
                    dt = h5py.string_dtype(encoding='utf-8')
                    group_inputs.create_dataset(key, data=value, dtype=dt)
                elif isinstance(value, bool):
                    group_inputs.create_dataset(key, data=int(value))  
                else:
                    group_inputs.create_dataset(key, data=value)

    def _set_cavity(self):
        """Set the cavity for the benchmark"""
        #for mbtrack2 ring parameters should be modified before map functions as transverse map uses local dispersion
        #as bunch was initiated with default parameters to plot it accordingly we introduce the equilibrium parameters for the tracking
        self.sig_p0_2 = self.parameter.Cq * self.parameter.gamma**2 * self.parameter.I3 / ((self.parameter.I2 * 2)+ self.parameter.I4) #* (self.parameter.Etot * 1e-9)**2
        factor =  self.parameter.C * np.sqrt((self.parameter.alpha * self.parameter.Etot)/(2*np.pi * self.parameter.h*((self.parameter.Vtot**2 - self.parameter.U0**2)**0.5)))
        self.sig_p0 = np.sqrt(self.sig_p0_2)
        self.ring.emit = np.array([self.parameter.epsnx_eq / self.parameter.gamma, self.parameter.epsny_eq / self.parameter.gamma])
        self.ring.sigma_delta = self.sig_p0
        self.ring.sigma_0 = factor * self.sig_p0/cons.c
        
        # self.theta = (np.pi/2) - np.arccos(self.eneloss_turn/self.parameter.Vtot)
        self.theta = np.arccos(self.eneloss_turn/self.parameter.Vtot)
        # theta = np.pi/4
        self.rf = RFCavity(self.ring, m=1, Vc=self.parameter.Vtot, theta=self.theta)
    
    def _set_ibs(self):
        """Set the intrabeam scattering for the benchmark"""
        #for mbtrack2
        if self.load_lattice is True:
            self.ibs = IntrabeamScattering(self.ring, model=self.modelname, n_points=self.slicing, n_bin=self.n_ibs_slices,
                                           distributed=self.distributed ,sectorMap=self.sectormap, sectors=self.sectors)
        else:
            self.ibs = IntrabeamScattering(self.ring, model=self.modelname, n_points=10, n_bin=self.n_ibs_slices)
            #Using Xsuite map optics values to calculate rates
            self.ibs.dispX = self.dx
            self.ibs.dispY = self.dy
            self.ibs.disppX = self.dpx
            self.ibs.disppY = self.dpy
            self.ibs.beta_x = self.betax
            self.ibs.beta_y = self.betay 
            self.ibs.alphaX = self.alphax
            self.ibs.alphaY = self.alphay
        #for xsuite
        if self.ibs_toggle == True:
            ibs_kick = xf.IBSKineticKick(num_slices=self.n_ibs_slices)
            self.line.configure_intrabeam_scattering(element=ibs_kick, 
                                                     name="ibskick", 
                                                     index=self.index, 
                                                     update_every=1)
            if self.ibs_formalism == 'analytical':
                ibs_kick = xf.IBSAnalyticalKick(formalism="B&M", num_slices=self.n_ibs_slices)
            elif self.ibs_formalism == 'kinetic':
                ibs_kick = xf.IBSKineticKick(num_slices=self.n_ibs_slices)
            self.line_map = self.environment.new_line(name='line_map', components=['ring_map', 'ibskick'])
        else:
            self.line_map = self.environment.new_line(name='line_map', components=['ring_map'])
        
        self.line_map.particle_ref = self.particle_ref.copy()
        self.line_map._needs_rng = True

    def _set_radiation(self):
        """Set the radiation for the benchmark"""
        # Synchrotron radiation damping for mbtrack2
        self.sr = SynchrotronRadiation(self.ring, switch=[1, 1, 1])
        #Making the map and tracking elements and sectors (if there are any)
        self.long_map = LongitudinalMap(self.ring)
        self.trans_map = TransverseMap(self.ring)
        # slicing the map
        if self.ibs_toggle is True:
            if self.sectors > 1 and self.distributed is False and self.sectormap is True:
                positions = np.zeros(self.sectors)
                for i in range(self.sectors):
                    positions[i] = i * self.ring.L / self.sectors
                # creating sectors
                sect = transverse_map_sector_generator(self.ring, positions=positions)
                # creating array of tracking elements
                self.tracking_elements = [self.long_map, self.rf, self.sr]
                for i in range(self.sectors):
                    self.tracking_elements.append(sect[i])
                    self.tracking_elements.append(self.ibs)
            else:
                self.tracking_elements=[self.long_map, self.trans_map, self.rf, self.sr, self.ibs]
        else:
            self.tracking_elements=[self.long_map, self.trans_map, self.rf, self.sr]
        #for xsuite
        self.line_map.build_tracker()
        self.line_map.configure_radiation(model='quantum')

    def _set_monitors(self):
        """Set the monitors for the benchmark"""
        #for mbtrack2
        self.mbtrack_monitor = BunchMonitor(1, 1,buffer_size=10, total_size=self.n_turns, file_name=self.file_name_m)
        #for xsuite
        if self.load_lattice is True:
            self.emit_mon = xc.EmittanceMonitor.install(line=self.line, name="EmittanceMonitor", at=0, stop_at_turn=self.n_turns)
        else:
            self.emit_mon = xc.EmittanceMonitor.install(line=self.line_map, name="EmittanceMonitor", at=0, stop_at_turn=self.n_turns)

    def _run_mbtrack2(self):
        """Run the mbtrack2 tracking"""
        for i in tqdm(range(self.n_turns)):      
            for el in self.tracking_elements:
                el.track(self.mybunch)
            self.mbtrack_monitor.track(self.mybunch)
    
    def _get_Tibs(self):
        if self.modelname == 'CIMP':
            raise ValueError("Cannot track without CIMP use other tracking function")
        if self.ibs_toggle is True:
            self.tracking_elements = [self.trans_map, self.long_map, self.rf, self.sr]
        else:
            self.tracking_elements = [self.trans_map, self.long_map, self.rf, self.sr]
        print(self.tracking_elements)

        T_ibs = []
        for i in tqdm(range(self.n_turns)):      
            for el in self.tracking_elements:
                el.track(self.mybunch)
                self.ibs.initialize(self.mybunch)
                a, b = self.ibs.scatter()
                T_x, T_y, T_s = self.ibs.get_scatter_T(g_ab=a, g_ba=b)
                self.ibs.kick(self.mybunch, T_x=T_x, T_y=T_y, T_p=T_s)
                T_ibs.append([np.average(T_x), np.average(T_y), np.average(T_s)])
            self.mbtrack_monitor.track(self.mybunch)

        filename = self.file_name_m+'_Tibs_.json'
        with open(filename, 'w') as json_file:
            json.dump(T_ibs, json_file)
        print("Done!")

    def _run_xsuite(self):
        """Run the xsuite tracking"""
        epsx= []
        mean_x  = []
        mean_y  = []
        mean_z  = []
        mean_e = []
        sigma_x = []
        sigma_y = []
        sigma_z = []
        sigma_e = []
        if self.load_lattice is True:
            for i_chunk in tqdm(range(self.n_chunk)):
                self.monitor = xt.ParticlesMonitor(_context= self.context,start_at_turn=i_chunk*self.chunk_size, stop_at_turn=(i_chunk+1)*self.chunk_size,
                num_particles=self.n_macroparticles)   
                self.line.track(self.particles, num_turns=self.chunk_size, turn_by_turn_monitor=self.monitor, with_progress=False)
                mean_x[i_chunk*self.chunk_size : (i_chunk+1)*self.chunk_size] = np.average(self.monitor.x, axis=0)
                mean_y[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]   = np.average(self.monitor.y,axis=0)
                mean_z[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]   = np.average(self.monitor.zeta,axis=0)
                mean_e[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]   = np.average(self.monitor.delta,axis=0)
                sigma_x[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]  = np.std(self.monitor.x,axis=0)
                sigma_y[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]  = np.std(self.monitor.y,axis=0)
                sigma_z[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]  = np.std(self.monitor.zeta,axis=0)
                sigma_e[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]  = np.std(self.monitor.delta,axis=0)
                epsx[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]     = (np.std(self.monitor.x, axis=0)**2 - (self.dx * np.std(self.monitor.delta, axis=0) * self.tw.beta0**2)**2) / self.betax

            output = {
                'epsx' : {
                    'value' : list(epsx),
                    'label' : r'$\epsilon_x$'
                },
                'epsy' : {
                    'value': self.emit_mon.gemitt_y.tolist(),
                    'label': r'$\epsilon_y$'
                },
                'epsz' : {
                    'value': self.emit_mon.gemitt_zeta.tolist(),
                    'label': r'$\epsilon_z$'
                },
                'meanx' : {
                    'value': list(mean_x),
                    'label' : r'$\bar{x}$'
                },
                'meany' : {
                    'value': list(mean_y),
                    'label' : r'$\bar{y}$'
                },
                'meanz' : {
                    'value': list(mean_z),
                    'label' : r'$\bar{z}$'
                },
                'meane' : {
                    'value': list(mean_e),
                    'label' : r'$\bar{e}$'
                },
                'sigmax' : {
                    'value': list(sigma_x),
                    'label' : r'$\sigma_x$'
                },
                'sigmay' : {
                    'value': list(sigma_y),
                    'label' : r'$\sigma_y$'
                },
                'sigmaz' : {
                    'value': list(sigma_z),
                    'label' : r'$\sigma_z$'
                },
                'sigmae' : {
                    'value' : list(sigma_e),
                    'label' : r'$\sigma_e$'
                },
                'inputs' : self._inputs,
                'parameters': vars(self.parameter),
                'chosen parameters': self.chosen_params,
                # 'comment' : self.comments
            }
        else:
            for i_chunk in tqdm(range(self.n_chunk)):
                self.monitor = xt.ParticlesMonitor(_context= self.context,start_at_turn=i_chunk*self.chunk_size, stop_at_turn=(i_chunk+1)*self.chunk_size,num_particles=self.n_macroparticles)   
                self.line_map.track(self.particles, num_turns=self.chunk_size, turn_by_turn_monitor=self.monitor, with_progress=False)
                mean_x[i_chunk*self.chunk_size : (i_chunk+1)*self.chunk_size] = np.average(self.monitor.x, axis=0)
                mean_y[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]   = np.average(self.monitor.y,axis=0)
                mean_z[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]   = np.average(self.monitor.zeta,axis=0)
                mean_e[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]   = np.average(self.monitor.delta,axis=0)
                sigma_x[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]  = np.std(self.monitor.x,axis=0)
                sigma_y[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]  = np.std(self.monitor.y,axis=0)
                sigma_z[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]  = np.std(self.monitor.zeta,axis=0)
                sigma_e[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]  = np.std(self.monitor.delta,axis=0)
                epsx[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]     = (np.std(self.monitor.x, axis=0)**2 - (self.dx * np.std(self.monitor.delta, axis=0) * self.tw.beta0**2)**2) / self.betax

            output = {
                'epsx' : {
                    'value' : list(epsx),
                    'label' : r'$\epsilon_x$'
                },
                'epsy' : {
                    'value': self.emit_mon.gemitt_y.tolist(),
                    'label': r'$\epsilon_y$'
                },
                'epsz' : {
                    'value': self.emit_mon.gemitt_zeta.tolist(),
                    'label': r'$\epsilon_z$'
                },
                'meanx' : {
                    'value': list(mean_x),
                    'label' : r'$\bar{x}$'
                },
                'meany' : {
                    'value': list(mean_y),
                    'label' : r'$\bar{y}$'
                },
                'meanz' : {
                    'value': list(mean_z),
                    'label' : r'$\bar{z}$'
                },
                'meane' : {
                    'value': list(mean_e),
                    'label' : r'$\bar{e}$'
                },
                'sigmax' : {
                    'value': list(sigma_x),
                    'label' : r'$\sigma_x$'
                },
                'sigmay' : {
                    'value': list(sigma_y),
                    'label' : r'$\sigma_y$'
                },
                'sigmaz' : {
                    'value': list(sigma_z),
                    'label' : r'$\sigma_z$'
                },
                'sigmae' : {
                    'value' : list(sigma_e),
                    'label' : r'$\sigma_e$'
                },
                'inputs' : self._inputs,
                'parameters': vars(self.parameter),
                'chosen parameters': self.chosen_params,
                # 'comment' : self.comments
            }

        filename = self.xsuite_output
        with open(filename, 'w') as json_file:
            json.dump(output, json_file)
        print("Done!")












