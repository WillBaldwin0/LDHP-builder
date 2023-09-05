import ase.io
from anaAtoms import find_molecs, split_molecs, wrap_molecs, scan_vol
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import warnings
from ase.neighborlist import neighbor_list, natural_cutoffs
import itertools



# for anaAtoms
cutoffs_split_inorg = {'Pb': 0.1, 'I': 0.1, 'Br': 0.1, 'Cl': 0.1, 'H':0.5, 'C':1, 'N':1},
cutoffs_connected_inorganic = {'Pb': 3.0, 'I': 0.1, 'Br': 0.1, 'Cl': 0.1, 'H':0.0, 'C':0.0, 'N':0.0, 'O': 0.0}
cutoffs_connected_organic = {'Pb': 0.1, 'I': 0.1, 'Br': 0.1, 'Cl': 0.1, 'H':0.5, 'C':1.0, 'N':1.0, 'O': 1.0}
cutoffs_cross_connections = {'Pb': 1.0, 'I': 1.0, 'Br': 1.0, 'Cl': 1.0, 'H':0.5, 'C':1.0, 'N':1.0, 'O': 1.0}

# general
inorganics = ['Pb', 'I', 'Br', 'Cl', 'Cs']
halides = ['I', 'Br', 'Cl']
organics = ['C', 'N', 'H', 'O', 'S', 'I', 'Br', 'Cl' ]
all_elements = list(set(inorganics + halides + organics))

# for neighbourlists
cutoffs_octahedron = {}
for element1 in all_elements:
    for element2 in all_elements:
        if element1 == 'Pb' and element2 in halides:
            cutoffs_octahedron[(element1, element2)] = 4.0
        elif element2 == 'Pb' and element1 in halides:
            cutoffs_octahedron[(element1, element2)] = 4.0
        else:
            cutoffs_octahedron[(element1, element2)] = 0.0


def strip_atoms(ats, spec_list):
    """ useful for removing all the leads and iodines """
    new_ats = deepcopy(ats)
    todel = []
    for att in new_ats:
        if att.symbol in spec_list:
            todel.append(att.index)
    for ii in reversed(todel):
        del new_ats[ii]
    return new_ats


def strip_atoms_by_index(ats, ind_list):
    new_ats = deepcopy(ats)
    for ii in reversed(sorted(ind_list)):
        del new_ats[ii]
    return new_ats


def find_octahedra(ats):
    all_indices = np.array(list(range(ats.get_global_number_of_atoms())))
    i, j, D = ase.neighborlist.neighbor_list('ijD', ats, cutoffs_octahedron)
    syms = np.array(ats.get_chemical_symbols())
    leads = list(all_indices[syms == 'Pb'])
    halides = set()
    for att_ind in leads:
        neighbours = j[i == att_ind]
        halides = halides.union(set(neighbours))
    return leads, list(halides)

            

def get_inorganic_layers(perovskite):
    """ takes a perovksite, splits it into inorganic layers (can be more than monolayer) and returns the layers"""
    inorg_only = strip_atoms(perovskite, organics)
    find_molecs([inorg_only], cutoffs_connected_inorganic)
    wrap_molecs([inorg_only])
    layers = split_molecs([inorg_only])
    return layers


def extract_all_mols_with_charge(perovskite):
    lead_indices, oct_halide_indices = find_octahedra(perovskite)
    # TODO: check homogeneous halides?
    inorganic_charge = 2 * len(lead_indices)
    inorganic_charge -= len(oct_halide_indices)

    organic_only = strip_atoms_by_index(perovskite, lead_indices + oct_halide_indices)
    
    find_molecs([organic_only])
    wrap_molecs([organic_only])
    mols = split_molecs([organic_only])

    total_num_mols = len(mols)

    counts = []
    distinct = []
    distinct_syms = []
    for mol in mols:
        specs = sorted(mol.get_chemical_symbols())

        if not (specs in distinct_syms):
            distinct_syms.append(specs)
            distinct.append(mol)
            counts.append(1)
        else:
            idx = distinct_syms.index(specs)
            counts[idx] += 1
    
    num_distinct_mols = len(distinct)

    # calc charges
    # first check for MA or FA and fill in a +1
    known_charges= {}
    for idx, syms in enumerate(distinct_syms):
        if syms in [
            ["C", "H", "H", "H", "H", "H", "H", "N"], 
            ["C", "H", "H", "H", "H", "H", "N", "N"],
        ]:
            known_charges[idx] = 1.0

    # work out all combinatinos, but fill in known values. 
    trial_charges = [1,2,3]
    trial_charge_combinations = [trial_charges]*num_distinct_mols
    for key, value in known_charges.items():
        trial_charge_combinations[key] = [value]
    
    trial_charge_combinations = itertools.product(*trial_charge_combinations)
    matches = []
    for charge_combos in trial_charge_combinations:
        organic_charge = sum([charge*count for charge, count in zip(charge_combos, counts)])
        if organic_charge == -inorganic_charge:
            matches.append(charge_combos)

    if len(matches) != 1:
        matches = [None]
            
    return distinct, counts, matches[0]

""" 

def extract_all_mols_with_charge(perovskite):
    # only for 2d perovskites
    inorganic_layers = get_inorganic_layers(perovskite)
    inorganic_charge = 0
    for layer in inorganic_layers:
        inorganic_charge += 2 * layer.get_chemical_symbols().count('Pb')
        assert len(set(layer.get_chemical_symbols()).intersection(halides)) == 1
        inorganic_charge -= layer.get_chemical_symbols().count('I')
        inorganic_charge -= layer.get_chemical_symbols().count('Br')
        inorganic_charge -= layer.get_chemical_symbols().count('Cl')
    
    # extract molecules with counts
    organic_only = strip_atoms(perovskite, inorganics)
    find_molecs([organic_only], cutoffs_connected_organic)
    wrap_molecs([organic_only])
    mols = split_molecs([organic_only])

    total_num_mols = len(mols)

    counts = []
    distinct = []
    distinct_syms = []
    for mol in mols:
        specs = sorted(mol.get_chemical_symbols())

        if not (specs in distinct_syms):
            distinct_syms.append(specs)
            distinct.append(mol)
            counts.append(1)
        else:
            idx = distinct_syms.index(specs)
            counts[idx] += 1
    
    num_distinct_mols = len(distinct)

    # calc charges
    # first check for MA or FA and fill in a +1
    known_charges= {}
    for idx, syms in enumerate(distinct_syms):
        if syms in [
            ["C", "H", "H", "H", "H", "H", "H", "N"], 
            ["C", "H", "H", "H", "H", "H", "N", "N"],
        ]:
            known_charges[idx] = 1.0

    # work out all combinatinos, but fill in known values. 
    trial_charges = [1,2,3]
    trial_charge_combinations = [trial_charges]*num_distinct_mols
    for key, value in known_charges.items():
        trial_charge_combinations[key] = [value]
    
    trial_charge_combinations = itertools.product(*trial_charge_combinations)
    matches = []
    for charge_combos in trial_charge_combinations:
        organic_charge = sum([charge*count for charge, count in zip(charge_combos, counts)])
        if organic_charge == -inorganic_charge:
            matches.append(charge_combos)

    if len(matches) != 1:
        matches = [None]
            
    return distinct, counts, matches[0] """