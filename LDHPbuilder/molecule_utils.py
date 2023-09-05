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
inorganics = ['Pb', 'I', 'Br', 'Cl']
halides = ['I', 'Br', 'Cl']
organics = ['C', 'N', 'H', 'O']

# for neighbourlists
cutoffs_octahedron = {}
for element1 in organics + inorganics:
    for element2 in organics + inorganics:
        if element1 == 'Pb' and element2 in halides:
            cutoffs_octahedron[(element1, element2)] = 4.0
        elif element2 == 'Pb' and element1 in halides:
            cutoffs_octahedron[(element1, element2)] = 4.0
        else:
            cutoffs_octahedron[(element1, element2)] = 0.0



def get_rotation_matrix(vec2, vec1):
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = R.align_vectors(vec2, vec1)
    return r[0].as_matrix()


def random_points_on_cap(angle, N, centerline):
    phis = np.random.uniform(0, 2*np.pi, N)
    def inverse_cdf(x):
        return np.arccos(1-x)
    s  = np.random.uniform(0, 1-np.cos(angle*np.pi / 180), N)
    thetas = inverse_cdf(s)
    points = np.array([[np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)] for theta, phi in zip(thetas, phis)])
    mat = get_rotation_matrix(np.array([1.,0.,0.]), centerline)
    return points @ mat#.transpose()


def check_molecule_intersection(ats, num_mol):
    new_ats = strip_atoms(ats, inorganics)
    find_molecs([new_ats], cutoffs_connected_organic)
    mols_split = split_molecs([new_ats])
    if len(mols_split) == num_mol:
        return True
    elif len(mols_split) < num_mol:
        return False
    else:
        raise ValueError("weird")
    

def get_mol_to_inorganic_intersections(ats):
    find_molecs([ats], cutoffs_cross_connections)
    mols_split = split_molecs([ats])
    return len(mols_split)


def furtherst_heavy_atom_indices(molecule):
    """
    finds the two most distant heavy atoms in a molecule
    """
    mat = molecule.get_all_distances()
    positions = molecule.get_positions()
    syms = np.array(molecule.get_chemical_symbols())
    ind = np.unravel_index(np.argmax(mat), mat.shape)
    vector = positions[ind[0]] - positions[ind[1]]
    heavy_atom_mask = np.logical_or(syms == 'C', syms == 'N')
    
    components = (positions - np.mean(positions, axis=0)) @ vector
    components[np.logical_not(heavy_atom_mask)] = 0.0

    return (np.argmin(components), np.argmax(components)), vector


def fix_to_only_nitrogens(molecule, bp_indices):
    is_N = np.asarray(molecule.get_chemical_symbols())[np.asarray(bp_indices)] == 'N'
    if not any(is_N):
        warnings.warn("warning: neither bp is a nitrogen")
        return bp_indices, (False, True)
    elif all(is_N):
        return bp_indices, (False, True)
    else:
        if is_N[0]:
            return (bp_indices[0], bp_indices[0]), (False, False)
        else:
            return (bp_indices[1], bp_indices[1]), (True, True)


def fix_to_only_nitrogens_with_cutoff(molecule, bp_indices):
    sender, receiver = neighbor_list(
        "ij",
        molecule,
        cutoff=np.asarray(natural_cutoffs(molecule))  + 0.3,
        self_interaction=True,
    )

    is_N = []
    for index in bp_indices:
        bonds_from_index = sender == index
        bp_neighbours = receiver[bonds_from_index]
        if 'N' in np.asarray(molecule.get_chemical_symbols())[bp_neighbours]:
            is_N.append(True)
        else:
            is_N.append(False)

    is_N = np.asarray(is_N)

    if not any(is_N):
        warnings.warn("warning: neither bp is a nitrogen")
        return bp_indices, (False, True)
    elif all(is_N):
        return bp_indices, (False, True)
    else:
        if is_N[0]:
            return (bp_indices[0], bp_indices[0]), (False, False)
        else:
            return (bp_indices[1], bp_indices[1]), (True, True)
        

def get_inertia_tensor(molecule, weights=None):
    """
    Calculate the symmetric inertia tensor for a molecule. from pyxtal
    """
    coords = molecule.get_positions()
    weights = np.ones(len(coords))
    coords -= np.mean(coords, axis=0)
    Inertia = np.zeros([3,3])
    Inertia[0,0] = np.sum(weights*coords[:,1]**2 + weights*coords[:,2]**2)
    Inertia[1,1] = np.sum(weights*coords[:,0]**2 + weights*coords[:,2]**2)
    Inertia[2,2] = np.sum(weights*coords[:,0]**2 + weights*coords[:,1]**2)
    Inertia[0,1] = Inertia[1,0] = -np.sum(weights*coords[:,0]*coords[:,1])
    Inertia[0,2] = Inertia[2,0] = -np.sum(weights*coords[:,0]*coords[:,2])
    Inertia[1,2] = Inertia[2,1] = -np.sum(weights*coords[:,1]*coords[:,2])
    return Inertia


def principle_axes_of_molecule(molecule):
    """ from pyxtal_molecule.get_principle_axes """
    Inertia = get_inertia_tensor(molecule)
    evals, matrix = np.linalg.eigh(Inertia)
    return matrix.transpose()


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


def rotate_perovskite(perovskite, rot_mat):
    perovskite.set_positions(perovskite.get_positions() @ rot_mat)
    perovskite.set_cell(perovskite.get_cell() @ rot_mat)


def rotate_molecule(molecule, rot_mat):
    molecule.set_positions(molecule.get_positions() @ rot_mat)


def reflect_perovskite(perovskite, normal):
    nnormal = normal / np.linalg.norm(normal)
    ref_mat = np.eye(3)
    ref_mat -= 2* np.outer(nnormal, nnormal)
    perovskite.set_positions(perovskite.get_positions() @ ref_mat)
    perovskite.set_cell(perovskite.get_cell() @ ref_mat)


def refect_molecule(molecule, normal):
    nnormal = normal / np.linalg.norm(normal)
    ref_mat = np.eye(3)
    ref_mat -= 2* np.outer(nnormal, nnormal)
    molecule.set_positions(molecule.get_positions() @ ref_mat)


def find_octahedra(ats):
    all_indices = np.array(list(range(ats.get_global_number_of_atoms())))
    i, j, D = ase.neighborlist.neighbor_list('ijD', ats, cutoffs_octahedron)
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
    """ sort halides in a top-bottom-clockwise_round_equator order """
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
    """ takes a perovksite, splits it into inorganic layers (can be more than monolayer) and returns the layers"""
    inorg_only = strip_atoms(perovskite, organics)
    find_molecs([inorg_only], cutoffs_connected_inorganic)
    wrap_molecs([inorg_only])
    layers = split_molecs([inorg_only])
    return layers


def extract_one_inorganic_layer(perovskite):
    """ takes a perovksite, splits it into inorganic layers (can be more than monolayer). returns a SINGLE layer"""
    return get_inorganic_layers(perovskite)[0]



def extract_one_molecule(perovskite):
    organic_only = strip_atoms(perovskite, inorganics)
    find_molecs([organic_only], cutoffs_connected_organic)
    wrap_molecs([organic_only])
    mols = split_molecs([organic_only])
    return mols[0]


def extract_one_molecule_with_charge(perovskite):
    """ exacts a molecule from a perovskite layer.
    the charge is only correct if there is a single type of molecule in the perovksite """
    total_number_of_leads = perovskite.get_chemical_symbols().count('Pb')
    inorganic_charge = -2 * total_number_of_leads
    
    organic_only = strip_atoms(perovskite, inorganics)
    find_molecs([organic_only], cutoffs_connected_organic)
    wrap_molecs([organic_only])
    mols = split_molecs([organic_only])

    num_mols = len(mols)
    charge_per_mol = (-inorganic_charge) // num_mols

    return mols[0], charge_per_mol


def extract_distinct_mols(perovskite):
    total_number_of_leads = perovskite.get_chemical_symbols().count('Pb')
    #inorganic_charge = -2 * total_number_of_leads
    
    organic_only = strip_atoms(perovskite, inorganics)
    find_molecs([organic_only], cutoffs_connected_organic)
    wrap_molecs([organic_only])
    mols = split_molecs([organic_only])

    num_mols = len(mols)
    #charge_per_mol = (-inorganic_charge) // num_mols

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
            
    return distinct, counts, matches[0]



def find_inorganic_layer_normal(monolayer):
    """ given a monolayer, finds the normal vector, and the lattice vector which is closest to it. """
    # idea is to replicate, extract 1 layer, then best fit a plane to the leads atoms.
    cp = deepcopy(monolayer)
    layers = get_inorganic_layers(cp * (2,2,2))

    # we've enlarged cell, so there are now many layers, so just take one
    layer = layers[0]
    layer = strip_atoms(layer, halides)
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
    """ given a monolayer, returns the pseudocubic lattice vectors of the layer """
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
    #print(vectors)

    # lead atom 0
    #neighbours_of_lead = j[i == 0]
    #vectors = vectors[neighbours_of_lead]
    vectors = vectors[i == 0]
    #print(vectors)

    normed_vectors = vectors / np.linalg.norm(np.asarray(vectors), axis=-1)[:,None]
    n_normal = fitted_normal / np.linalg.norm(fitted_normal)

    # i is the index into neighbours_of_lead
    i1 = 0
    v1 = normed_vectors[i1]
    perp_vec = np.cross(v1, n_normal)
    #print(perp_vec)
    #print(np.linalg.norm(normed_vectors-perp_vec, axis=-1))
    i2 = np.argmin(np.linalg.norm(normed_vectors-perp_vec, axis=-1))
    return vectors[i1], vectors[i2]


def get_unique_arrays_bonding_points(lead_positions, ps_lattice_constants):
    # bonding points are found by looping over the atoms. 
    # for some leads in a layer:
    #       A  B
    #       C  D
    # given translational invariance, distinct arrays are
    # [[1,0],[0,0]], [1,1,0,0], [1,0,1,0], [1,0,0,1]
    # [1,1,1,0], [], []
    pass