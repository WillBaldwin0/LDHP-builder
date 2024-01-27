# Will Baldwin January 2023, wjb48@cam.ac.uk
from aseMolec.anaAtoms import find_molecs, split_molecs, wrap_molecs
import numpy as np
from copy import deepcopy
from ase.neighborlist import neighbor_list
import ase.neighborlist
from .utils import *


def find_octahedra(ats):
    """finds octahedra in a perovskite
    
    Parameters
    ----------
    ats : ase.Atoms
        perovskite structure
    
    Returns
    -------
    list
        list of the indices of the lead atoms
    list
        list of lists of the indices of the halide atoms. Each 
        inner list corresponds to halides around a lead atom
    list
        list of lists of the vectors pointing from the lead atom to each halide
    """

    all_indices = np.array(list(range(ats.get_global_number_of_atoms())))
    i, j, D = ase.neighborlist.neighbor_list('ijD', ats, CUTOFFS_OCTAHEDRON)
    syms = np.array(ats.get_chemical_symbols())
    leads = list(all_indices[syms == 'Pb'])
    halides = []
    vectors = []
    for att_ind in leads:
        neighbours = j[i == att_ind]
        halides.append(neighbours)
        vectors.append(D[i == att_ind])
    return leads, halides, vectors
            

def sort_octahedron_halide_indices(halides, vectors, normal):
    """sort halides in a top-bottom-clockwise_round_equator order"""
    normed_vectors = vectors / np.linalg.norm(np.asarray(vectors), axis=-1)[:,None]

    # get top and bottom
    value = normed_vectors @ normal
    top_index = np.argmax(value)
    bot_index = np.argmin(value)

    remaining = list(range(6))
    remaining.remove(top_index)
    remaining.remove(bot_index)

    # work out the ordering. pick a first one, then find best matches around the circle
    index_2 = remaining[0]
    vec2 = normed_vectors[index_2]
    perp_vec = np.cross(vec2, normed_vectors[top_index])
    index_3 = np.argmin(np.linalg.norm( (normed_vectors - perp_vec) , axis=-1))
    vec3 = normed_vectors[index_3]
    
    index_4 = np.argmin(np.linalg.norm( (normed_vectors + 2*vec2) , axis=-1))
    index_5 = np.argmin(np.linalg.norm( (normed_vectors + 2*vec3) , axis=-1))

    ordered_halides = halides[[top_index, bot_index, index_2, index_3, index_4, index_5]]
    ordered_vectors = vectors[[top_index, bot_index, index_2, index_3, index_4, index_5]]

    return ordered_halides, ordered_vectors


def get_inorganic_layers(perovskite):
    """Split a perovskite into inorganic layers
    
    Parameters
    ----------
    perovskite : ase.Atoms
        perovskite structure
    
    Returns
    -------
    list
        list of the inorganic layers (each layer is an ase.Atoms object)
    """

    inorg_only = strip_atoms(perovskite, ORGANICS)
    find_molecs([inorg_only], CUTOFFS_CONNECTED_INORGANIC)
    wrap_molecs([inorg_only])
    layers = split_molecs([inorg_only])
    return layers


def extract_one_inorganic_layer(perovskite):
    """Exracts a single inorganic layer from a perovskite"""

    return get_inorganic_layers(perovskite)[0]


def extract_one_molecule(perovskite):
    """exacts a single organic molecule from a perovskite """
    organic_only = strip_atoms(perovskite, INORGANICS)
    find_molecs([organic_only], CUTOFFS_CONNECTED_ORGANIC)
    wrap_molecs([organic_only])
    mols = split_molecs([organic_only])
    return mols[0]


def extract_one_molecule_with_charge(perovskite):
    """Exracts a molecule from a perovskite layer, and returns its charge.

    The charge assumes there is only a single type of molecule in the perovksite
    """

    total_number_of_leads = perovskite.get_chemical_symbols().count('Pb')
    inorganic_charge = -2 * total_number_of_leads
    
    organic_only = strip_atoms(perovskite, INORGANICS)
    find_molecs([organic_only], CUTOFFS_CONNECTED_ORGANIC)
    wrap_molecs([organic_only])
    mols = split_molecs([organic_only])

    num_mols = len(mols)
    charge_per_mol = (-inorganic_charge) // num_mols

    return mols[0], charge_per_mol


def extract_distinct_mols(perovskite):
    """Exracts distinct molecules from a perovskite layer.

    Returns a list of the distinct organic molecules in a perovskite. Two molucules
    are classed as the same simply if they have the same list of chemical symbols.
    """
    total_number_of_leads = perovskite.get_chemical_symbols().count('Pb')
    
    organic_only = strip_atoms(perovskite, INORGANICS)
    find_molecs([organic_only], CUTOFFS_CONNECTED_ORGANIC)
    wrap_molecs([organic_only])
    mols = split_molecs([organic_only])

    num_mols = len(mols)

    distinct = []
    distinct_syms = []
    for mol in mols:
        specs = sorted(mol.get_chemical_symbols())
        if not (specs in distinct_syms):
            distinct_syms.append(specs)
            distinct.append(mol)

    return distinct


def extract_all_mols_with_charge(perovskite):
    # only for 2d perovskites
    inorganic_layers = get_inorganic_layers(perovskite)
    inorganic_charge = 0
    for layer in inorganic_layers:
        inorganic_charge += 2 * layer.get_chemical_symbols().count('Pb')
        assert len(set(layer.get_chemical_symbols()).intersection(HALIDES)) == 1
        inorganic_charge -= layer.get_chemical_symbols().count('I')
        inorganic_charge -= layer.get_chemical_symbols().count('Br')
        inorganic_charge -= layer.get_chemical_symbols().count('Cl')
    
    # extract molecules with counts
    organic_only = strip_atoms(perovskite, INORGANICS)
    find_molecs([organic_only], CUTOFFS_CONNECTED_ORGANIC)
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


def find_inorganic_layer_normal(monolayer):
    """Find the normal vector to an inorganic monolayer
    
    Parameters
    ----------
    monolayer : ase.Atoms
        inorganic monolayer
    
    Returns
    -------
    np.array
        normal vector
    int
        index of the cell vector with the best match to the normal vector
    """

    # idea is to replicate, extract 1 layer, then best fit a plane to the leads atoms.
    cp = deepcopy(monolayer)
    layers = get_inorganic_layers(cp * (2,2,2))

    # we've enlarged cell, so there are now many layers, so just take one
    layer = layers[0]
    layer = strip_atoms(layer, HALIDES)
    lead_positions = layer.get_positions()

    # best fit plane
    G = lead_positions.sum(axis=0) / lead_positions.shape[0]
    u, s, vh = np.linalg.svd(lead_positions - G)
    normal = vh[2, :]
    normalised_lattice_constants = layer.get_cell() / np.linalg.norm(layer.get_cell(), axis=-1)[:,None]

    # we need to allow for the normal to be facing the wrong way
    differences_p = np.linalg.norm(normalised_lattice_constants - normal, axis=-1)
    differences_n = np.linalg.norm(normalised_lattice_constants + normal, axis=-1)
    
    if np.min(differences_p) < np.min(differences_n):
        differences = differences_p
    else:
        differences = differences_n

    # also get the index of cell vectors with best match 
    best_match_cell_vector = np.argmin(differences)
    return normal, best_match_cell_vector


def get_2d_pseudocubic_lattice_vectors(monolayer, normal_index, fitted_normal):
    """Get the pseudocubic lattice vectors perovskite monolayer
    
    Parameters
    ----------
    monolayer : ase.Atoms
        inorganic monolayer
    normal_index : int
        index of the cell vector with the best match to the normal vector
    fitted_normal : np.array
        fitted normal vector to the monolayer
    
    Returns
    -------
    np.array
        vector 1
    np.array
        vector 2
    """

    # pick one lead atom and look for lead atoms within 7.0 A
    leads_only = strip_atoms(monolayer, ['I', 'Br', 'Cl', 'C', 'N', 'H', 'O'])
    replicate_tuple = [2,2,2]
    replicate_tuple[normal_index] = 1
    replicate_tuple = tuple(replicate_tuple)
    leads_only = leads_only * replicate_tuple

    # pick one lead atom
    i, j, vectors = ase.neighborlist.primitive_neighbor_list(
        'ijD',
        leads_only.get_pbc(), 
        leads_only.get_cell(), 
        leads_only.get_positions(),
        7.0
    )
    vectors = vectors[i == 0]

    normed_vectors = vectors / np.linalg.norm(np.asarray(vectors), axis=-1)[:,None]
    n_normal = fitted_normal / np.linalg.norm(fitted_normal)

    # i is the index into neighbours_of_lead
    i1 = 0
    v1 = normed_vectors[i1]
    perp_vec = np.cross(v1, n_normal)
    i2 = np.argmin(np.linalg.norm(normed_vectors-perp_vec, axis=-1))
    return vectors[i1], vectors[i2]