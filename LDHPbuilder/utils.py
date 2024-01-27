# Will Baldwin January 2023, wjb48@cam.ac.uk
from aseMolec.anaAtoms import find_molecs, split_molecs
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from ase.neighborlist import neighbor_list, natural_cutoffs

import warnings
# the way we generate random points on a spherical cap has an unused degree of freedom
warnings.filterwarnings("ignore", message="Optimal rotation is not uniquely or poorly defined for the given sets of vectors.")

# constants
INORGANICS = ['Pb', 'I', 'Br', 'Cl']
HALIDES = ['I', 'Br', 'Cl']
ORGANICS = ['C', 'N', 'H', 'O']

# cutoffs for molecule detections in aseMolec
CUTOFFS_CONNECTED_INORGANIC = {'Pb': 3.0, 'I': 0.1, 'Br': 0.1, 'Cl': 0.1, 'H':0.0, 'C':0.0, 'N':0.0, 'O': 0.0}
CUTOFFS_CONNECTED_ORGANIC = {'Pb': 0.1, 'I': 0.1, 'Br': 0.1, 'Cl': 0.1, 'H':0.5, 'C':1.0, 'N':1.0, 'O': 1.0}
CUTOFFS_CROSS_CONNECTIONS = {'Pb': 1.0, 'I': 1.0, 'Br': 1.0, 'Cl': 1.0, 'H':0.5, 'C':1.0, 'N':1.0, 'O': 1.0}
CUTOFFS_OCTAHEDRON = {}
for element1 in ORGANICS + INORGANICS:
    for element2 in ORGANICS + INORGANICS:
        if element1 == 'Pb' and element2 in HALIDES:
            CUTOFFS_OCTAHEDRON[(element1, element2)] = 4.0
        elif element2 == 'Pb' and element1 in HALIDES:
            CUTOFFS_OCTAHEDRON[(element1, element2)] = 4.0
        else:
            CUTOFFS_OCTAHEDRON[(element1, element2)] = 0.0


def get_rotation_matrix(vec2, vec1):
    """ rotation matrix to rotate vector 1 onto vector 2 """
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = R.align_vectors(vec2, vec1)
    return r[0].as_matrix()


def random_points_on_cap(angle, N, centerline):
    """Generate a set of poitns uniformly distributed over a spherical cap
    
    Parameters
    ----------
    angle : float
        half angle subtended by the cap in degrees
    N : int
        number of points to generate
    centerline : np.array
        vector pointing along the centerline of the cap
    
    Returns
    -------
    np.array
        N points on the cap
    """

    phis = np.random.uniform(0, 2*np.pi, N)
    def inverse_cdf(x):
        return np.arccos(1-x)
    s  = np.random.uniform(0, 1-np.cos(angle*np.pi / 180), N)
    thetas = inverse_cdf(s)
    points = np.array([[np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)] for theta, phi in zip(thetas, phis)])
    mat = get_rotation_matrix(np.array([1.,0.,0.]), centerline)
    return points @ mat


def check_molecule_intersection(ats, num_mol):
    """Check for intersections between organic cations in a perovskite
    
    Parameters
    ----------
    ats : ase.Atoms
        perovskite structure
    num_mol : int
        number of organic cations to check for
        
    Returns
    -------
    bool
        True if there are no intersections, False otherwise"""

    new_ats = strip_atoms(ats, INORGANICS)
    find_molecs([new_ats], CUTOFFS_CONNECTED_ORGANIC)
    mols_split = split_molecs([new_ats])
    if len(mols_split) == num_mol:
        return True
    elif len(mols_split) < num_mol:
        return False
    else:
        raise ValueError("weird")
    

def get_mol_to_inorganic_intersections(ats):
    """check for intersections between organic cations and inorganic layers

    This function returns the number of disconnected pieces after 
    forming intramolecular bonds and bonds between organic cations and 
    inorganic layers according to CUTOFFS_CROSS_CONNECTIONS.
    
    Parameters
    ----------
    ats : ase.Atoms
        perovskite structure
    
    Returns
    -------
    int
        number of disconnected pieces
    """
    
    find_molecs([ats], CUTOFFS_CROSS_CONNECTIONS)
    mols_split = split_molecs([ats])
    return len(mols_split)


def check_mol_to_inorganic_intersections(ats):
    """check for intersections between organic cations and inorganic layers

    Parameters
    ----------
    ats : ase.Atoms
        perovskite structure
    
    Returns
    -------
    bool
        True if there are no intersections, False otherwise
    """

    find_molecs([ats], CUTOFFS_CROSS_CONNECTIONS)
    mols_split = split_molecs([ats])
    syms_list = [set(mol.get_chemical_symbols()) for mol in mols_split]
    for mol in syms_list:
        has_halide = ('Cl' in mol) or ('Br' in mol) or ('I' in mol)
        if has_halide and 'C' in mol:
            return False
    return True


def furtherst_heavy_atom_indices(molecule):
    """Finds the two most distant heavy atoms in a molecule

    Parameters
    ----------
    molecule : ase.Atoms
        molecule to analyze
    
    Returns
    -------
    tuple
        indices of the two atoms
    np.array
        vector pointing from the first atom to the second
    """

    mat = molecule.get_all_distances()
    positions = molecule.get_positions()
    syms = np.array(molecule.get_chemical_symbols())
    ind = np.unravel_index(np.argmax(mat), mat.shape)
    vector = positions[ind[0]] - positions[ind[1]]
    heavy_atom_mask = np.logical_or(syms == 'C', syms == 'N')
    
    components = (positions - np.mean(positions, axis=0)) @ vector
    components[np.logical_not(heavy_atom_mask)] = 0.0
    indices = (np.argmin(components), np.argmax(components))
    vector = positions[indices[0]] - positions[indices[1]]

    return indices, vector


def fix_to_only_nitrogens(molecule, bp_indices):
    """Given two atom indices, select only those which are Nitogrens"""

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
    """Given two atom indices, select Nitogrens within a cutoff radius"""

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
    """Calculate the symmetric inertia tensor for a molecule. from pyxtal"""

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
    """ Remove given chemical species from ase.Atoms object """

    new_ats = deepcopy(ats)
    todel = []
    for att in new_ats:
        if att.symbol in spec_list:
            todel.append(att.index)
    for ii in reversed(todel):
        del new_ats[ii]
    return new_ats


def rotate_perovskite(perovskite, rot_mat):
    """rotate structure with cell by a matrix"""

    perovskite.set_positions(perovskite.get_positions() @ rot_mat)
    perovskite.set_cell(perovskite.get_cell() @ rot_mat)


def rotate_molecule(molecule, rot_mat):
    """rotate structure without a cell by a matrix"""

    molecule.set_positions(molecule.get_positions() @ rot_mat)


def reflect_perovskite(perovskite, normal):
    """reflect structure with cell by a normal vector"""

    nnormal = normal / np.linalg.norm(normal)
    ref_mat = np.eye(3)
    ref_mat -= 2* np.outer(nnormal, nnormal)
    perovskite.set_positions(perovskite.get_positions() @ ref_mat)
    perovskite.set_cell(perovskite.get_cell() @ ref_mat)


def reflect_molecule(molecule, normal):
    """reflect structure without a cell by a normal vector"""

    nnormal = normal / np.linalg.norm(normal)
    ref_mat = np.eye(3)
    ref_mat -= 2* np.outer(nnormal, nnormal)
    molecule.set_positions(molecule.get_positions() @ ref_mat)


def reduced_random_binary_array(n): 
    """Generate a random binary array subject to certain symmetry constraints
    
    given N a power of 2, generate a random binary x array of length N
    subject to
        x[2^n:2^(n+1)] \in { x[0:2^n], not x[0:2^n] }

    Exmaples:
        [1,0,1,0] or [0,1,1,0],
        [1,0,0,1,1,0,0,1] or [0,1,0,1,0,1,0,1]
    non-examples:
        [1,0,1,1]

    Parameters
    ----------
    n : int
        length of the array
    
    Returns
    -------
    np.array
        random binary array
    """

    # n must be a power of 2
    assert ((n & (n-1) == 0) and n != 0)

    exp = int(np.log2(n))
    stuff = [np.random.choice([True,False])]
    for i in range(exp):
        stuff += stuff
        if np.random.choice([0,1]):
            stuff[2**i:] = list(np.logical_not(stuff[2**i:]))
    return np.asarray(stuff)