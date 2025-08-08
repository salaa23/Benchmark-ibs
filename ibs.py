# -*- coding: utf-8 -*-
"""
Module for intrabeam scattering computations.
"""
import warnings

import numpy as np
import scipy.integrate as quad
from scipy.constants import c, elementary_charge, epsilon_0
from scipy.special import hyp2f1
from scipy.signal import find_peaks

from mbtrack2.tracking.element import Element


class IntrabeamScattering(Element):
    """
    Intrabeam scattering class computes the IBS growth rate analytically each 
    turn and apply corresponding kicks to each particle according to the 
    following models:
        
    Piwinski Standard (PS):
        Uses classical model that computes the sum of scattering events based 
        on Rutherford formalism [1].
    Piwinski Modified (PM):
        Modification of the PS model in order to take into account the 
        dispersion and the variation of the optic functions around the ring [2].
    Bane (Bane):
        Approximation for high-energy beams which allows computing a single
        scattering integral instead of two [3].
    Completely integrated modified Piwinski model (CIMP):
        Approximating elliptic integral as a special Legendre function [4].

    Parameters:
    -----------
    ring: Synchrotron object
        Ring to consider.
    bunch: Bunch or Beam object
        Bunch or Beam object which will be tracked.
    model: {'CIMP','Bane','PM','PS'} 
        Implemented computational models for computing the growth rate T_(x,y,p):
            CIMP: the CIMP model [4].        
            Bane: the Bane model [3].
            PM: the modified model of Piwinski [2].
            PS: the standard model of Piwinski [1].
    n_points: int, optional
        Number of partitions (sampling) of the optics functions (depends on the 
        lattice complexity), can use a lower value if the beta function is 
        simpler, thus gaining computation speed. 
        The Default is 1000
    n_bin: int, optional
        Number of bins for the bunch.binning profile function. Will determine 
        the number of slices for the profile function.
        If the value of n_bin is greater than 1 the kick method will compute 
        the kick[5], applying momentum change according to the positions 
        of each macroparticle with respect to the density of macroparticles in 
        that position.
        If n_bin is set to 1, the kick will be computed assuming Rho(z) as a 
        uniform distribution. 
        The Default is 100. 
    Distributed: bool, optional
        If True, the kick will be applied to each point that is preselected 
        according to the area with high bending, since that is where the IBS
        is more likely to occur. (it is not usable with other kick methods)
        The Default is False.
    sectorMap: bool, optional
        If True, the kick will be applied by slicing the ring into sectors,
        applying the kick to each sector separately, which is useful for large 
        accelerators. When `sectorMap` is enabled, You should call 
        `transverse_sector_map_generator()` and call ibs calss after each sector, 
        in this case sectors should be set the the number of sections to ensure
        the beta functions are computed correctly for each sector.
         sector-Map method takes precedence over repeated kick and sector
        parameter will only serve to generate an optic function at the proper size. 
        If `sectorMap` is False and `distributed` is False but `sectors > 1`, 
        repeated kicks are applied to the entire ring, which may underestimate the IBS effect.
        to avoide overkicking the maximum value of sectors should be set to (N-1)*P, N is the
        number of bends per achromat, P is the periodicity.
        The Default is False.
    sectors: int, optional
        The default is False.
        the entire ring. using the given map (Should be transverse map)
        If greater than 1 and using the ring transverse map, the kick is 
        applied to each sector (slice) separately. 
        The Default is 1.
        
    Methods:
    --------
    initialize(bunch):
        Initializes the dynamic parameters at each turn, modifies the class 
        variables according to the selected model.
    scatter():
        Computes the scattering integrals according to the selected model.
    get_scatter_T():
        Computes the growth rate from the scattering integrals.
    kick(bunch, T_x, T_y, T_p):
        Tracking method of IntrabeamScattering takes T_(x,y,p) and apply 
        momentum change to the coordinates of macroparticles inside the bunch.

    References:
    -----------
    [1] A. Piwinski, Intra-Beam-Scattering, (1974). 
    http://dx.doi.org/10.5170/CERN-1992-001.226
    [2] K. L. F. Bane, A Simplified Model of Intrabeam Scattering, in 8th 
    European Particle Accelerator Conference (Paris, France, 2002), p. 1443.
    https://doi.org/10.48550/arXiv.physics/0206002
    [3] K. L. F. Bane, H. Hayano, K. Kubo, T. Naito, T. Okugi, and J. Urakawa, 
    Intrabeam Scattering Analysis of Measurements at KEK’s Accelerator Test 
    Facility Damping Ring, Phys. Rev. ST Accel. Beams 5, 084403 (2002).
    https://doi.org/10.1103/PhysRevSTAB.5.084403
    [4] K. Kubo, S. K. Mtingwa, and A. Wolski, Intrabeam Scattering Formulas 
    for High Energy Beams, Phys. Rev. ST Accel. Beams 8, 081001 (2005).
    https://doi.org/10.1103/PhysRevSTAB.8.081001
    [5] R. Bruce, J. M. Jowett, M. Blaskiewicz, and W. Fischer, Time 
    Evolution of the Luminosity of Colliding Heavy-Ion Beams in BNL 
    Relativistic Heavy Ion Collider and CERN Large Hadron Collider, 
    Phys. Rev. ST Accel. Beams 13, 091001 (2010).
    https://doi.org/10.1103/PhysRevSTAB.13.091001

    """

    def __init__(self, ring, model, n_points=1000, n_bin=100, distributed = False, sectorMap = False, sectors = 1):
        self.distrbuted = distributed
        self.sectormap = sectorMap
        self.sectors = int(sectors)
        self.ring = ring
        self.n_points = int(n_points)
        self.n_bin = int(n_bin)
        #This is so to avoid setting sectors since distributed takes into account the entire ring
        if self.distrbuted is True:
            self.s = np.linspace(0, self.ring.L, self.n_points)
        else:
            self.s = np.linspace(0, self.ring.L/self.sectors, self.n_points)
        self.model = str(model)
        #try block is for testing in case we want to import optics array from another code
        try: 
            self.beta_x, self.beta_y = self.ring.optics.beta(self.s)
            self.dispX, self.disppX, self.dispY, self.disppY = self.ring.optics.dispersion(
                self.s)
            self.alphaX, self.alphaY = self.ring.optics.alpha(self.s)
        except:
            self.beta_x = self.ring.optics.betaX 
            self.beta_y = self.ring.optics.betaY 
            self.dispX  = self.ring.optics.dispX 
            self.disppX = self.ring.optics.disppX
            self.dispY  = self.ring.optics.dispY
            self.disppY = self.ring.optics.disppY
            self.alphaX = self.ring.optics.alphaX
            self.alphaY = self.ring.optics.alphaY

        self.H_x = (1 / self.beta_x) * (self.dispX**2 +
                                        ((self.beta_x * self.disppX) +
                                        (self.alphaX * self.dispX))**2)
        self.H_y = (1 / self.beta_y) * (self.dispY**2 +
                                        ((self.beta_y * self.disppY) +
                                        (self.alphaY * self.dispY))**2)

        if self.ring.optics.use_local_values:
            warnings.warn(
                "Lattice file not loaded, intiating optics with approximated values",
                UserWarning)
            self.n_points = 1

        if self.model not in ["Bane", "CIMP", "PM", "PS"]:
            raise ValueError(
                "Incorrect model name. Allowed model names are: PM, PS, Bane, CIMP."
            )

        self.r_0 = (ring.particle.charge**
                    2) / (4 * np.pi * epsilon_0 * c**2 * ring.particle.mass)
        #Normalization parameters
        self.dpx = np.average(self.disppX)
        self.dpy = np.average(self.disppY)
        #Seeking positions with value of beta_x at peak
        betax0 = np.average(self.beta_x) + np.std(self.beta_x)
        self.indexes = find_peaks(self.beta_x, height=betax0)
        if self.sectormap is True and self.distrbuted is True:
            warnings.warn(
                "sectorMap and distributed kick are both set to True, distributed kick will be used.",
                UserWarning)
        if self.sectormap is True and self.sectors == 1:
            warnings.warn("sectorMap kick is set to True, sectors should be set to the number of sectors in the ring. ")
        if self.sectormap is False and self.distrbuted is False and self.sectors > 1:
            warnings.warn(
                "No distributed kick and no sector kick is set, " \
                "repeated kicks will be applied to the entire ring. This may underestimate the IBS effect.",
                UserWarning)
        # Creating the time step list
        s1 = np.array([.0])
        s_tot = np.concatenate((s1, self.s[self.indexes[0]]))
        dt = []
        for i in range(len(s_tot)):
            dt.append(s_tot[i] - s_tot[i-1])
        dt.pop(0)
        self.dt = np.array(dt, dtype=np.float64)/c    

    def initialize(self, bunch):
        """
        Initializes the dynamic parameters at each turn, modifies the class 
        variables according to the selected model.

        Parameters
        ----------
        bunch: Bunch or Beam object
            Bunch or Beam object which will be tracked.
        """
        self.N = bunch.current * self.ring.T0 / elementary_charge
        self.d = 4 * np.std(bunch['y'])
        self.sigma_s = np.std(bunch['tau'])
        self.sigma_p = np.std(bunch['delta'])
        self.sigma_px = np.std(bunch['xp'])
        self.sigma_py = np.std(bunch['yp'])
        self.emit_x, self.emit_y, _ = bunch.emit
        #Computing the normalized energy spread in horizontal and vertical planes
        #to compare with xsuite kick that uses normalized energy spread, will be removed or kept after testing
        self.sigma_px_normalized = np.std(bunch['xp'] - (self.dpx * bunch['delta']))
        self.sigma_py_normalized = np.std(bunch['yp'] - (self.dpy * bunch['delta']))
        if self.model == "PS":
            self.h = (1 / self.sigma_p**2) + \
                     (self.dispX**2 / (self.beta_x * self.emit_x)) + \
                     (self.dispY**2 / (self.beta_y * self.emit_y))
            sigma_h = np.sqrt(1 / self.h)
            self.sigma_H = sigma_h   
        elif self.model in ["CIMP", "PM", "Bane"]:
            self.H = (1 / self.sigma_p**2) + (self.H_x / self.emit_x) + \
                     (self.H_y / self.emit_y)
            sigma_H = np.sqrt(1 / self.H)
            self.sigma_H = sigma_H

        self.a = (self.sigma_H / self.ring.gamma) * np.sqrt(
            self.beta_x / self.emit_x)
        self.b = (self.sigma_H / self.ring.gamma) * np.sqrt(
            self.beta_y / self.emit_y)
        self.q = self.sigma_H * self.ring.beta * np.sqrt(2 * self.d / self.r_0)
        if self.model == "Bane":
            self.C_log = np.log(self.q**2 / self.a**2)
            self.C_a = self.a / self.b

        elif self.model in ["PM", "PS", "CIMP"]:
            self.A = (self.r_0**2 * self.N) / (
                64 * np.pi**2 * self.ring.beta**3 * self.ring.gamma**4 *
                self.emit_x * self.emit_y * self.sigma_s * self.sigma_p)

    def scatter(self):
        """
        Computes the scattering integrals according to the selected model.
        
        Returns
        -------
        If self.model == "PM" or "PS":
            vabq, v1aq, v1bq: arrays
                Array of scattering integral values at each point around the 
                ring.
        If self.model == "Bane":
            gval: array
                Array of scattering integral values at each point around the 
                ring.
        If self.model == "CIMP":
            g_ab, g_ba: arrays
                Array of analytical function g(a/b) and g(b/a) respectively, 
                that simulate the integrals at each point around the ring.

        """
        if self.model in ["PS", "PM"]:
            vabq = np.zeros(self.n_points, dtype=np.float64)
            v1aq = np.zeros(self.n_points, dtype=np.float64)
            v1bq = np.zeros(self.n_points, dtype=np.float64)

            def scattering(u, x, y, z):
                """
                Eq. (17) in:
                L. R. Evans and B. W. Zotter, Intrabeam Scattering in the SPS.
                https://cds.cern.ch/record/126036
                """
                P2 = x**2 + ((1 - x**2) * u**2)
                Q2 = y**2 + ((1 - y**2) * u**2)
                P = np.sqrt(P2)
                Q = np.sqrt(Q2)
                f_abq = 8 * np.pi * ((1 - (3 * u**2)) / (P * Q)) * \
                        (2 * np.log((z) * ((1/P) + (1/Q)) / 2) - 0.5777777777)
                return f_abq

            for i in range(self.n_points):
                el_1aq, err = quad.quad(scattering,
                                        0,
                                        1,
                                        args=(1 / self.b[i],
                                              self.a[i] / self.b[i],
                                              self.q[i] / self.b[i]))
                el_1bq, err = quad.quad(scattering,
                                        0,
                                        1,
                                        args=(1 / self.a[i],
                                              self.b[i] / self.a[i],
                                              self.q[i] / self.a[i]))
                el_abq = -(el_1aq * (1 / self.b[i]**2)) - (el_1bq *
                                                           (1 / self.a[i]**2))
                vabq[i] = el_abq
                v1aq[i] = el_1aq
                v1bq[i] = el_1bq
            return vabq, v1aq, v1bq

        elif self.model == "Bane":
            gval = np.zeros(self.n_points, dtype=np.float64)

            def g_func(u, j, C_a):
                """
                Eq. (12) in [2].

                Parameters
                ----------
                u : float
                    integration variable.
                j : int
                    index.
                C_a : float
                    result of a/b

                Returns
                -------
                g_val : array
                    Scattering integral value at a given point.

                """
                g_val = ((2 * np.sqrt(C_a[j]))/ np.pi) * \
                        (1 / (np.sqrt(1 + u**2) * np.sqrt(C_a[j]**2 + u**2)))
                return g_val

            for j in range(self.n_points):
                reslt, err = quad.quad(g_func, 0, np.inf, args=(j, self.C_a))
                gval[j] = reslt
            return gval

        elif self.model == "CIMP":
            g_ab = np.zeros(self.n_points)
            g_ba = np.zeros(self.n_points)

            def Puv(u, v, x):
                """
                https://dlmf.nist.gov/14.3
                """
                if x < 1:
                    val = ((1 + x) / (1 - x))**(u/2) * \
                          hyp2f1(v+1, -v, 1-u, (.5 - (.5 * x)))
                else:
                    val = ((1 + x) / (x - 1))**(u/2) * \
                          hyp2f1(v+1, -v, 1-u, (.5 - (.5 * x)))
                return val

            def g_func(u):
                """
                Eq. (34) in [4].
                """
                x_arg = (1 + u**2) / (2*u)
                if u >= 1:
                    g_val = np.sqrt(np.pi / u) * ((Puv(0, -.5, x_arg)) +
                                                  ((3/2) *
                                                   (Puv(-1, -.5, x_arg))))
                else:
                    g_val = np.sqrt(np.pi / u) * ((Puv(0, -.5, x_arg)) -
                                                  ((3/2) *
                                                   (Puv(-1, -.5, x_arg))))
                return g_val

            self.g_ab = np.zeros(self.n_points)
            self.g_ba = np.zeros(self.n_points)
            for i in range(self.n_points):
                val_ab = g_func(self.a[i] / self.b[i])
                val_ba = g_func(self.b[i] / self.a[i])
                g_ab[i] = val_ab
                g_ba[i] = val_ba
            return g_ab, g_ba

    def get_scatter_T(self,
                      vabq=None,
                      v1aq=None,
                      v1bq=None,
                      g_ab=None,
                      g_ba=None,
                      gval=None):
        """
        Computes the growth rate from the scattering integrals:
            Piwinski: Eq. (16-18) in [2].
            Bane: Eq (14) and (20) in [3].
            CIMP: Eq (35-37) in [4].
              
        The growth rate can be defined as:
                1/T_ibs = 1/sigma * d(sigma)/dt
        Parameters
        ----------
        If self.model == "PM" or "PS":
            vabq, v1aq, v1bq: arrays
                Takes integral values from scatter() method. 
                The default is None.
        If self.model == "Bane":
            gval: array
                Takes elliptical integral array from scatter() method. 
                The default is None.
        If self.model == "CIMP:
            g_ab, g_ba: arrays
            Takes analytical function arrays, g(a/b) and g(b/a) respectively. 
            The default is None.
            
        Returns
        -------
        T_x: Array
            Array of IBS growth rates on the horizontal plane over the entire 
            ring in [1/s].
        T_y: Array
            Array of IBS growth rates on the vertical plane over the entire 
            ring in [1/s].
        T_p: Array
            Array of IBS growth rates on the longitudinal plane over the entire 
            ring in [1/s].

        """
        if self.model in ["PS", "PM"]:
            T_p = self.A * (vabq * (self.sigma_H**2 / self.sigma_p**2))
            T_x = self.A * (v1bq + (vabq * ((self.dispX**2 * self.sigma_H**2) /
                                            (self.beta_x * self.emit_x))))
            T_y = self.A * (v1aq + (vabq * ((self.dispY**2 * self.sigma_H**2) /
                                            (self.beta_y * self.emit_y))))

        elif self.model == "Bane":
            T_pp = (self.r_0**2 * self.N * self.C_log * self.sigma_H * gval *
                    (self.beta_x * self.beta_y)**(-1 / 4)) / (
                        16 * self.ring.gamma**3 * self.emit_x**(3 / 4) *
                        self.emit_y**(3 / 4) * self.sigma_s * self.sigma_p**3)
            T_p = np.average(T_pp)
            T_x = (self.sigma_p**2 * self.H_x * T_pp) / self.emit_x
            T_y = (self.sigma_p**2 * self.H_y * T_pp) / self.emit_y

        elif self.model == "CIMP":
            K_a = ((np.log(self.q**2 / self.a**2) * g_ba) / self.a) + (
                (np.log(self.q**2 / self.b**2) * g_ab) / self.b)
            T_p = 2 * np.pi**(3 / 2) * self.A * (
                (self.sigma_H**2 / self.sigma_p**2) * K_a)
            T_x = 2 * np.pi**(3 / 2) * self.A * (
                (-self.a * np.log(self.q**2 / self.a**2) * g_ba) +
                (((self.H_x * self.sigma_H**2) / self.emit_x) * K_a))
            T_y = 2 * np.pi**(3 / 2) * self.A * (
                (-self.b * np.log(self.q**2 / self.b**2) * g_ab) +
                (((self.H_y * self.sigma_H**2) / self.emit_y) * K_a))

        return T_x, T_y, T_p

    def kick(self, bunch, T_x, T_y, T_p):
        """
        Apply kick to the bunch coordinates by converting growth rate to 
        momentum change [1].

        Parameters
        ----------
        bunch : Object
            Bunch or Beam object.
        T_p : Array
            Growth rate of 'delta' in [1/s].
        T_x : Array
            Growth rate of 'xp' in [1/s].
        T_y : Array
            Growth rate of 'yp' in [1/s].
            
        Reference:
        ----------
        [1] R. Bruce, J. M. Jowett, M. Blaskiewicz, and W. Fischer, Time 
        Evolution of the Luminosity of Colliding Heavy-Ion Beams in BNL 
        Relativistic Heavy Ion Collider and CERN Large Hadron Collider, 
        Phys. Rev. ST Accel. Beams 13, 091001 (2010).
        https://doi.org/10.1103/PhysRevSTAB.13.091001
        
        """
        # Growth rate operations are moved here to get real values incase of benchmarking with other codes
        T_p[T_p <=0] = 0
        T_x[T_x <=0] = 0
        T_y[T_y <=0] = 0
        if self.distrbuted is False:
            T_x = np.average(T_x)
            T_y = np.average(T_y)
            T_p = np.average(T_p)
        
        if self.n_bin > 1:
            bins, sorted_index, profile, center = bunch.binning(
                n_bin=self.n_bin)

            normalized_profile = profile / max(profile)
            Rho = normalized_profile[sorted_index]
        else:
            Rho = 1.0

        N_mp = len(bunch)
        len_index = len(self.indexes[0])
        #if distributed is True, the kick will be applied to each point that is preselected according to the area with high
        # bending, since that is where the IBS is more likely to occur.
        if self.distrbuted is True:
            for i,el in enumerate(self.indexes[0]):
                Delta_pz = self.sigma_p * np.sqrt(
                    np.sqrt(2) * T_p[el] * self.ring.T0/self.n_points *
                    Rho) * np.random.normal(size=N_mp)
                Delta_px = self.sigma_px * np.sqrt(
                    np.sqrt(2) * T_x[el] * self.ring.T0/self.n_points *
                    Rho) * np.random.normal(size=N_mp)
                Delta_py = self.sigma_py * np.sqrt(
                    np.sqrt(2) * T_y[el] * self.ring.T0/self.n_points *
                    Rho) * np.random.normal(size=N_mp)
                bunch['xp'] += Delta_px
                bunch['yp'] += Delta_py
                bunch['delta'] += Delta_pz
        # if sectormap set to True, the kick will be applied to each sector of the ring, however there is no for loop
        # thus it is needed to be applied with TransverseSectorMap
        # here the code will compute Tibs for one sector and kick only once, thus the function should be called after each
        # sector.   
        elif self.sectormap is True:
            Delta_pz = self.sigma_p * np.sqrt(
                np.sqrt(2) * T_p * (self.ring.T0/self.sectors) *
                Rho) * np.random.normal(size=N_mp)
            Delta_px = self.sigma_px * np.sqrt(
                np.sqrt(2) * T_x * (self.ring.T0/self.sectors) *
                Rho) * np.random.normal(size=N_mp)
            Delta_py = self.sigma_py * np.sqrt(
                np.sqrt(2) * T_y * (self.ring.T0/self.sectors) *
                Rho) * np.random.normal(size=N_mp)
            bunch['xp'] += Delta_px
            bunch['yp'] += Delta_py
            bunch['delta'] += Delta_pz
        #This is the default case, where the kick can be applied to the entire ring if sectormap is False 
        # and sectors is set to 1.
        # In this case the code will compute Tibs for the entire ring and kick only once per turn.
        # however you can also set sectors to the number of cells in the lattice, thus the kick will be applied to each sector.       
        else: 
            for i in range(self.sectors): #after results of the tests this can be removed and this section compute the old way for more efficiency
                Delta_pz = self.sigma_p * np.sqrt(
                    np.sqrt(2) * T_p * (self.ring.T0/self.sectors) *
                    Rho) * np.random.normal(size=N_mp)
                Delta_px = self.sigma_px * np.sqrt(
                    np.sqrt(2) * T_x * (self.ring.T0/self.sectors) *
                    Rho) * np.random.normal(size=N_mp)
                Delta_py = self.sigma_py * np.sqrt(
                    np.sqrt(2) * T_y * (self.ring.T0/self.sectors) *
                    Rho) * np.random.normal(size=N_mp)
            bunch['xp'] += Delta_px
            bunch['yp'] += Delta_py
            bunch['delta'] += Delta_pz

    @Element.parallel
    @Element.track_bunch_if_non_empty
    def track(self, bunch):
        """
        Tracking method of IntrabeamScattering takes T_(x,y,p) and apply 
        momentum change to the coordinates of macroparticles inside the bunch.
        
        parameters:
        -----------
        bunch: Object
            Bunch or Beam object.
        """

        self.initialize(bunch)
        if self.model in ["PM", "PS"]:
            vabq, v1aq, v1bq = self.scatter()
            T_x, T_y, T_p = self.get_scatter_T(vabq=vabq, v1aq=v1aq, v1bq=v1bq)
        elif self.model == "Bane":
            gval = self.scatter()
            T_x, T_y, T_p = self.get_scatter_T(gval=gval)
        elif self.model == "CIMP":
            g_ab, g_ba = self.scatter()
            T_x, T_y, T_p = self.get_scatter_T(g_ab=g_ab, g_ba=g_ba)
        self.kick(bunch, T_x, T_y, T_p)
