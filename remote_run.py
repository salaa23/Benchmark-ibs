from remote_benchmark import _collective_

_inputs = {
    'n_macroparticles': int(100_000),
    'n_turns' : int(30_000),
    'comment':'_ibs_mbtrack_SOLEIL_contx-2-20mA_lattice_', #comment for the run
    # 'Np' : float(2.5e5), #number of particles
    'modelname': 'CIMP',#CIMP Bane PM PS
    'n_ibs_slices': int(100),
    'ibs_formalism': 'analytical', #or kinetic
    'bunch_style': 'unmatched', #or matched
    'location' : float(32.942356133075705), #Beta0 value for map location
    'ibs_toggle': True, #if to include ibs
    'load_lattice': True, #if mbtrack load the lattice file
    '_codes' : 'mbtrack2',#preffered code: mbtrack2, xsuite, both
    'slicing' : int(200), #slicing of the map, for mbtrack2 ibs module
    'distributed' : False, #Using distributed kick
    'sectormap' : True, #if to use sector map
    'sectors': int(1), #number of sectors for kick
}
func = _collective_(_inputs = _inputs)
