# Will Baldwin January 2023, wjb48@cam.ac.uk
import numpy as np
from aseMolec.anaAtoms import find_molecs, split_molecs, wrap_molecs, scan_vol
from copy import deepcopy
from ase.atoms import Atoms
from typing import List
from scipy.spatial.transform import Rotation as R

from .utils import (
    furtherst_heavy_atom_indices,
    fix_to_only_nitrogens_with_cutoff,
    principle_axes_of_molecule,
    random_points_on_cap,
    rotate_molecule,
    reflect_molecule,
    check_molecule_intersection,
    check_mol_to_inorganic_intersections,
    reduced_random_binary_array
)
from .analysis import (
    get_2d_pseudocubic_lattice_vectors,
    find_inorganic_layer_normal
)

class OrganicMolecule:
    """Holds data about organic cations.
    
    Organic Cations are manipulated by assigning a local coordinate system,
    and determining two salient points - 'bonding points' - which are used to
    attach the cation to the inorganic layer.
    
    Attributes
    ----------
    charge : int
        charge of the molecule
    directed_coordinate_system : np.ndarray
        3x3 matrix, each row is a unit vector in the local coordinate system. 
        The first vector points along the long axis of the molecule, from first to second bonding point.
    
    Methods
    -------
    get_atoms_shifted_rotated(bonding_index, vector, reference_vector, return_rotation_matrix=False)
        returns the molecule, with bonding atom shifted to the origin, and the molecule long axis aligned along vector
    """
    
    def __init__(self, ats, charge):
        """Initialize the molecule from an atoms object and a charge"""
        self._atoms = deepcopy(ats)
        self.charge = charge

        # find the furtherst sticking out atoms
        furtherst_heavy_atoms, self.long_vector = furtherst_heavy_atom_indices(ats)

        # we actually only want nitrogens
        self._bonding_atoms, self._to_flip = fix_to_only_nitrogens_with_cutoff(ats, furtherst_heavy_atoms)
        self._only_bond_to_nitrogens = True
        if self._to_flip[0] == self._to_flip[1]:
            self._number_of_bonding_points = 1
        else:
            self._number_of_bonding_points = 2

        # coorindate axes for rotations
        principle_axes = principle_axes_of_molecule(ats)
        # construct new basis
        vec1 = np.cross(principle_axes[2], self.long_vector)
        vec2 = np.cross(vec1, self.long_vector)
        self._coordinate_system = np.asarray([
            -self.long_vector,
            vec1,
            vec2
        ])
        self._coordinate_system /= np.linalg.norm(self._coordinate_system, axis=1)[:,None]
        self.directed_coordinate_system = self._coordinate_system

    def get_atoms_shifted_rotated(
        self, 
        bonding_index, 
        vector, 
        reference_vector, 
        return_rotation_matrix=False
    ):
        """Shift and rotate the molecule
        
        Parameters
        ----------
        bonding_index : int
            index of the atom to be shifted to the origin
        vector : np.ndarray
            vector to align the molecule long axis to
        reference_vector : np.ndarray
            Since aligning two vectors is ill-determined, use a reference vector 
            to uniquely define the rotation.
        return_rotation_matrix : bool, optional
            if True, return the rotation matrix, by default False
        
        Returns
        -------
        ase.Atoms
            the shifted and rotated molecule
        np.ndarray
            the rotation matrix, if return_rotation_matrix is True
        """
        
        new_ats = deepcopy(self._atoms)
        new_ats.set_positions(new_ats.get_positions() - new_ats.get_positions()[self._bonding_atoms[bonding_index]])
        if self._to_flip[bonding_index]:
            directed_coordinate_system = - self._coordinate_system
        else:
            directed_coordinate_system = self._coordinate_system

        intermediate_1 = np.cross(vector, reference_vector)
        target_coordinate_system = np.asarray([
            vector, 
            -intermediate_1,
            np.cross(vector, intermediate_1)
        ])
        target_coordinate_system /= np.linalg.norm(target_coordinate_system, axis=1)[:,None]
        mat = directed_coordinate_system.transpose() @ target_coordinate_system
        rotate_molecule(new_ats, mat)
        if return_rotation_matrix:
            return new_ats, mat
        else:
            return new_ats
            
    def _rotation_matrix_onto_new_coords(self, new_coords):
        # this matrix left multiples column vectors
        roatation_matrix = self.directed_coordinate_system.transpose() @ new_coords
        return roatation_matrix
    
    def __repr__(self):
        string = f"OrganicMolecule: {self._atoms.get_chemical_formula()}"
        string += f"\n charge: {self.charge}"
        string += f"\n only_bond_to_nitrogens = {self._only_bond_to_nitrogens}"
        string += f"\n number_of_bonding_sites = {self._number_of_bonding_points}"
        return string


class InorganicMonolayer:
    """Holds data about monolayers.

    This class is currently restricted to **corner sharing** monolayers.
    
    Attributes
    ----------
    atoms : ase.Atoms
        the monolayer
    fitted_normal : np.ndarray
        the normal vector to the monolayer
    two_d_direction : int
        the index of the cell vector which is the 2d direction of the monolayer
    ps_lattice_constants : np.ndarray
        the pseudocubic lattice constants of the monolayer    
    """
    def __init__(self, monolayer_ats):
        """Initialize the monolayer from an atoms object"""

        self.atoms = deepcopy(monolayer_ats)
        self.fitted_normal, self.two_d_direction = find_inorganic_layer_normal(
            monolayer_ats
        )
        self.ps_lattice_constants = np.asarray(
            get_2d_pseudocubic_lattice_vectors(monolayer_ats, self.two_d_direction, self.fitted_normal)
        )
        new_lc2 = np.cross(self.fitted_normal, self.ps_lattice_constants[0])
        syms = np.asarray(monolayer_ats.get_chemical_symbols())
        self.lead_positions = monolayer_ats.get_positions()[syms == 'Pb']
        self.atoms.cell[self.two_d_direction] *= 3.0
        self.layer_charge = -2 * self.lead_positions.shape[0]

    @classmethod
    def from_species_specification(cls, B_site, X_site, num_unit_cell_octahedra):
        """create the monolayer object from a specification of the species

        For now, only Pb halide monolayers are supported. The size of the unit cell is 
        specified by the number of octahedra, which for now must be 1, 2, or 4. This is used to
        create the natural cubic or orthorhombic unit cell.
        
        Parameters
        ----------
        B_site : str
            the B site species
        X_site : str
            the X site species
        num_unit_cell_octahedra : int
            the number of octahedra in the unit cell
        """

        assert B_site == 'Pb'
        assert num_unit_cell_octahedra in [1,2,4] # for now...
        typical_distances = {
            ('Pb', 'I') : 3.2,
            ('Pb', 'Br') : 3.0,
            ('Pb', 'Cl') : 2.85
        }
        dist = typical_distances[(B_site, X_site)] * (2**0.5)
        height = dist*4.0/(2**0.5)
        unit_cell_dims = {1:(1,1), 2:(1,2), 4:(2,2)}[num_unit_cell_octahedra]
        cell_options = {
            1: np.array([[dist, dist, 0.],[dist, -dist, 0.],[0.,0.,height]]),
            2: np.array([[2*dist, 0., 0.],[0., 2*dist, 0.],[0.,0.,height]]),
            4: np.array([[2*dist, 2*dist, 0.],[2*dist, -2*dist, 0.],[0.,0.,height]])
        }
        cell = cell_options[num_unit_cell_octahedra]
        species = []
        positions = []
        for index_a in range(unit_cell_dims[0]):
            for index_b in range(unit_cell_dims[1]):
                species.append(B_site)
                positions.append([dist*(index_a + index_b), dist*(index_a - index_b), 0.0])
        for index_a in range(unit_cell_dims[0]):
            for index_b in range(unit_cell_dims[1]):
                species += [X_site]*4
                positions += [
                    [dist*(index_a + index_b) + dist/2, dist*(index_a - index_b) + dist/2, 0.0],
                    [dist*(index_a + index_b) + dist/2, dist*(index_a - index_b) - dist/2, 0.0],
                    [dist*(index_a + index_b), dist*(index_a - index_b), +dist/2**0.5],
                    [dist*(index_a + index_b), dist*(index_a - index_b), -dist/2**0.5]
                ]
        monolayer = Atoms(symbols=species, positions=positions, cell=cell, pbc=[True, True, True])
        return cls(monolayer)

    def get_bonding_points(self, normal_displacement=3.5):
        """Returns the coordinates of the points in between salient halides. 
        
        The returned points are displaced by normal_displacement 
        relative to the centerline of the monolayer. north and south refer to 
        the two sides of the layer. """

        bonding_points_north = []
        bonding_points_south = []
        ps1 = self.ps_lattice_constants[0]
        ps2 = self.ps_lattice_constants[1]

        for i in range(self.lead_positions.shape[0]):
            bonding_points_north.append(self.lead_positions[i] + self.fitted_normal * normal_displacement + 0.5*(ps1+ps2))
            bonding_points_south.append(self.lead_positions[i] - self.fitted_normal * normal_displacement + 0.5*(ps1+ps2))

        return bonding_points_north, bonding_points_south
        
    def __repr__(self):
        string = "InorganicMonolayer: "
        string += f"\n {self.atoms.get_chemical_formula()}"
        string += f"\n normal = {self.fitted_normal}, 2d axis is direction {self.two_d_direction}"
        return string


class PerovskiteBuilder:
    """Class for assembling perovskites from molecules and inorganic layers.

    The implementation is based on the following assumptions:
        - the inorganic layer is a corner sharing monolayer
        - the B site is Pb
        - the molecules 'bond' to the inorganic layer in a specific way. 
        wherein a saleint point on the molecule sits between four octahrdea in the inorganic layer.
    
    Currently, methods for generating perovskites with just one type of 
    organic cation are implemented. The user can specify an inorganic monolayer,
    a cation, and the number of layers in the unit cell. The size of the unit 
    cell in the in-plane directions is dictated by the number of octahedra in
    the monolayer.

    For each 'bonding point' on the inorganic layer, one molecule is placed.

    The freedom in the generation is:
        - which end of the molecule to bond to the inorganic layer
        - whether to reflect the molecule in the in-plane directions
        - the angle of the 'longest vector' of the molecule relative to the inorganic layer normal
        - an additional rotation angle of the molecule about the 'longest vector'
    """

    def __init__(self):
        pass

    def generate_homogeneous_perovskite_samples(
        self,
        inorganic_layer,
        organic_molecule,
        num_samples=1,
        num_layers=1,
        max_num_attempts=None,
        max_attempts_per_symmetry=10,
        stacking_method='total_thickness',
        layer_spacing=1.5,
    ):
        """Generate a number of perovskite samples with the same organic molecule.

        Parameters
        ----------
        inorganic_layer : InorganicMonolayer
            the inorganic layer
        organic_molecule : OrganicMolecule
            the organic molecule
        num_samples : int, optional
            the number of samples to generate, by default 1
        num_layers : int, optional
            the number of layers in the unit cell, by default 1
        max_num_attempts : int, optional
            the maximum number of attempts to generate a sample, by default None    
        max_attempts_per_symmetry : int, optional
            the maximum number of attempts to generate a sample with a given symmetry, by default 10. 
            The symmetry is specified by the discrete vectors determining molecule reflections. to achieve
            even coverage of the discrete symmetries, try several times for each discrete symmetry by
            generating new rotations.
        stacking_method : str, optional
            how to stack the layers, by default 'total_thickness'. 
        layer_spacing : float, optional
            the spacing between layers, by default 1.5
        """

        if max_num_attempts is None:
            max_num_attempts = 10 * num_samples

        # get required number of molcules which determines the size of the random state
        num_leads = inorganic_layer.lead_positions.shape[0]
        num_mol_per_layer = (2*num_leads) // organic_molecule.charge
        num_molecules = num_mol_per_layer * num_layers

        # select the correct function for building the molcule
        if organic_molecule.charge == 1:
            f = self.generate_homogeneous_layer_charge_1
        elif organic_molecule.charge == 2:
            f = self.generate_homogeneous_layer_charge_2
        else:
            raise ValueError('only +1 or +2 cations are supported')

        perovskite_structures = []
        num_attempted_orientations = 0
        for i in range(max_num_attempts):
            # The random state is the combination of reflections, bonding points, and rotations
            # 1. molecule bonding points - this is a choice of either end of the molecule, one for each molecule
            molecule_reference_point_indices = reduced_random_binary_array(num_molecules)

            # 2. reflections are in the two in plane directions. desrcibed by [n,m]. n=0,1. [1,1] means reflect in both
            reflections = reduced_random_binary_array(num_molecules)
            reflections = np.vstack((reduced_random_binary_array(num_molecules), reflections)).transpose()
            molecule_rotation_direction = reduced_random_binary_array(num_molecules)
            
            # we want to broadly sample the discrete symmetries, but for low symmetries it takes several attempts to 
            # get an acceptable structure
            for inner_counter in range(max_attempts_per_symmetry):
                # 3. rotations
                molecule_long_vector = random_points_on_cap(45, 1, inorganic_layer.fitted_normal)[0]
                axial_rotations = np.random.uniform(low=0., high=2*np.pi)
                axial_rotations_2 = np.random.uniform(low=0., high=2*np.pi)

                # make layers
                perovskite_layers = []
                for layer_counter in range(num_layers):
                    chunk = slice(num_mol_per_layer*layer_counter, num_mol_per_layer*(layer_counter+1))
                    perovskite_layers.append(f(
                        inorganic_layer,
                        organic_molecule,
                        molecule_reference_point_indices[chunk],
                        reflections[chunk],
                        molecule_long_vector,
                        axial_rotations,
                        axial_rotations_2,
                        molecule_rotation_direction
                    ))

                # stack them
                structure = self.stack_layers(
                    perovskite_layers,
                    two_d_direction=inorganic_layer.two_d_direction,
                    method=stacking_method,
                    spacer=layer_spacing,
                )
                # check intersections
                if check_molecule_intersection(structure, num_molecules) and check_mol_to_inorganic_intersections(structure):
                    perovskite_structures.append(structure)
                    print(f"found {len(perovskite_structures)} structures after {num_attempted_orientations} attempts")
                    break

            num_attempted_orientations +=1

            if num_samples == len(perovskite_structures):
                break

        if num_attempted_orientations == max_num_attempts:
            warnings.warn( f"reached max number of attempts having only generated {len(perovskite_structures)} structures." )
        else:
            print(f"generated {num_samples} samples after {num_attempted_orientations} attempts")
        
        return perovskite_structures

    def stack_layers(
        self, 
        layers_objects,
        two_d_direction,
        method='total_thickness',
        apply_shear=True,
        spacer=1.5
    ):
        """ given a list of layers, stack them together into one atoms object,
        - method: describes how to set the distance between layers.
        for now, use total thickness for +2 cations and for +1 cations which are 'short'.
        use half thickness for +1 cations which are 'long' """

        assert method in ['total_thickness', 'half_thickness']

        if method == 'total_thickness':
            # center each unit 
            for layer in layers_objects:
                layer.center(vacuum=spacer, axis=[two_d_direction])
            
            # then stack them
            atoms = layers_objects[0]

            for i in range(1, len(layers_objects)):
                displacement = atoms.cell[two_d_direction]
                if apply_shear:
                    random_shift = np.random.randn(3) * 5.0
                    random_shift[two_d_direction] = 0.0
                    displacement += random_shift
                layers_objects[i].set_positions(layer.get_positions() + displacement)
                atoms.extend(layers_objects[i])
            
            if apply_shear:
                random_shift = np.random.randn(3) * 5.0
                random_shift[two_d_direction] = 0.0
                new_cell = atoms.get_cell()
                new_cell[two_d_direction] = new_cell[two_d_direction] + random_shift
                atoms.set_cell(new_cell)
            
            atoms.center(vacuum=spacer, axis=[two_d_direction])
        elif method == 'half_thickness':
            # get the thickness from the bottom of the inorganic layer
            # and the top of the organic layer.
            # do this for the layer to be added each time

            # stack
            atoms = layers_objects[0]
            for i in range(1, len(layers_objects)):
                # top of layer i-1
                top = layers_objects[i-1].get_positions()[:,two_d_direction].max()
                # bottom of layer i
                lead_pos = layers_objects[i].get_positions()[np.array(layers_objects[0].get_chemical_symbols()) == 'Pb']
                bottom = lead_pos[:,two_d_direction].min() - 5.0 - spacer
                displacement = np.array([0,0,top-bottom])

                if apply_shear:
                    random_shift = np.random.randn(3) * 5.0
                    random_shift[two_d_direction] = 0.0
                    displacement += random_shift
                
                layers_objects[i].set_positions(layers_objects[i].get_positions() + displacement)
                atoms.extend(layers_objects[i])

            # centering
            top = atoms.get_positions()[:,two_d_direction].max()
            lead_pos = atoms.get_positions()[np.array(atoms.get_chemical_symbols()) == 'Pb']
            bottom = lead_pos[:,two_d_direction].min() - 5.0 - spacer
            atoms.cell[two_d_direction] = np.array([0,0,top-bottom])
        
        return atoms

    def generate_homogeneous_layer_charge_1(
        self, 
        inorganic_layer,
        molecule_object,
        molecule_reference_point_indices,
        molecule_reflections,
        molecule_long_vector,
        molecule_axial_rotation,
        molecule_axial_rotation_2
    ):
        """ using the internally stored molecule and inorganic layer, make one perovskite layer. 
        The layer is homogeneous, meaning that all molecules are the same, and the 'long vector' is also shared."""

        top_layer_bonding_points, bottom_layer_bonding_points = inorganic_layer.get_bonding_points(normal_displacement=4.0)
        num_mols = len(top_layer_bonding_points) * 2

        # check sizes
        assert molecule_reflections.shape[0] == len(molecule_reference_point_indices) == num_mols

        # initialize atoms object
        perovskite_obj = deepcopy(inorganic_layer.atoms)
        perovskite_obj.cell[inorganic_layer.two_d_direction] *= 3.0

        # place molecules on top
        for (layer_bp, mol_bp, reflection) in zip(
            top_layer_bonding_points, 
            molecule_reference_point_indices[:num_mols//2], 
            molecule_reflections[:num_mols//2]
        ):
            self.place_molecule(
                perovskite_obj,
                inorganic_layer,
                molecule_object,
                layer_bp,
                mol_bp,
                reflection,
                molecule_long_vector,
                molecule_axial_rotation,
                molecule_axial_rotation_2
            )
        
        # place molecules on bottom
        for (layer_bp, mol_bp, reflection) in zip(
            bottom_layer_bonding_points, 
            molecule_reference_point_indices[num_mols//2:], 
            molecule_reflections[num_mols//2:]
        ):
            self.place_molecule(
                perovskite_obj,
                inorganic_layer,
                molecule_object,
                layer_bp,
                mol_bp,
                reflection,
                -1. * molecule_long_vector, # flip the vector
                -1. * molecule_axial_rotation,
                -1. * molecule_axial_rotation_2
            )

        return perovskite_obj

    def generate_homogeneous_layer_charge_2(
        self, 
        inorganic_layer,
        molecule_object,
        molecule_reference_point_indices,
        molecule_reflections,
        molecule_long_vector,
        molecule_axial_rotation,
        molecule_axial_rotation_2,
        molecule_rotation_direction
    ):
        """ using the internally stored molecule and inorganic layer, make one perovskite layer. 
        The layer is homogeneous, meaning that all molecules are the same, and the 'long vector' is also shared."""

        top_layer_bonding_points, bottom_layer_bonding_points = inorganic_layer.get_bonding_points(normal_displacement=4.0)
        num_mols = len(top_layer_bonding_points)

        # check sizes
        assert molecule_reflections.shape[0] == len(molecule_reference_point_indices) == num_mols

        # initialize atoms object
        perovskite_obj = deepcopy(inorganic_layer.atoms)
        perovskite_obj.cell[inorganic_layer.two_d_direction] *= 3.0

        # place molecules on top
        for (layer_bp, mol_bp, reflection, rot_dir) in zip(
            top_layer_bonding_points, 
            molecule_reference_point_indices, 
            molecule_reflections,
            molecule_rotation_direction
        ):
            self.place_molecule(
                perovskite_obj,
                inorganic_layer,
                molecule_object,
                layer_bp,
                mol_bp,
                reflection,
                molecule_long_vector,
                molecule_axial_rotation,
                molecule_axial_rotation_2,
                rot_dir,
            )

        return perovskite_obj

    def place_molecule(
        self,
        perovskite_atoms: Atoms,
        layer: InorganicMonolayer,
        molecule: List[OrganicMolecule],
        layer_reference_point: np.ndarray,
        molecule_reference_point_index: int,
        molecule_reflections: np.ndarray,
        molecule_long_vector: np.ndarray,
        molecule_axial_rotation: float,
        molecule_axial_rotation_2: float,
        rot_dir: np.ndarray,
        rotate_first=True,
    ):
        """ 
        Given an incomplete perovskite atoms object, an inorganic layer object, a molecule object, 
        add the molecule to the incomplete perovskite atoms object. 
        it is assumed that the perovskite already contains the inorganic layer. 
        """

        # shift the molecule and align to the 'long vector'
        mol, mat1 = molecule.get_atoms_shifted_rotated(
            molecule_reference_point_index, 
            molecule_long_vector, 
            layer.ps_lattice_constants[0],
            return_rotation_matrix=True,
        )
        # get the basis for reflections, and the matrix for the axial rotation
        reflection_basis = molecule.directed_coordinate_system @ mat1
        rotation_direction = rot_dir * 2 -1
        axial_rotation_matrix = R.from_rotvec(molecule_axial_rotation * reflection_basis[0]).as_matrix()

        # apply the axial rotation
        mol_cp = deepcopy(mol)
        if rotate_first:
            rotate_molecule(mol_cp, axial_rotation_matrix)

        # apply the reflections
        for i in range(2):
            if molecule_reflections[i]:
                normal = axial_rotation_matrix.T @ reflection_basis[i+1]
                reflect_molecule(mol_cp, normal)
        
        # second rotation
        
        axial_rotation_matrix = R.from_rotvec(rotation_direction * molecule_axial_rotation_2 * reflection_basis[0]).as_matrix()
        rotate_molecule(mol_cp, axial_rotation_matrix)
        
        # shift the molecule to the layer bonding point and add to the perovskite
        mol_cp.set_positions(mol_cp.get_positions() + layer_reference_point)
        perovskite_atoms.extend(mol_cp)
    
    def _attempt_to_squash(self, perovskite_guess):
        pg = deepcopy(perovskite_guess) * (2,2,2)

        original_num_entities = get_mol_to_inorganic_intersections(pg)
        for step in range(1000):
            if get_mol_to_inorganic_intersections(pg) == original_num_entities:
                pg.center(vacuum=1.5 - 0.05*step, axis=[self.layer.two_d_direction])
            else:
                break
        if step == 1000:
            assert 0

        perovskite_guess.center(vacuum=1.5 - 0.05*(step-1), axis=[self.layer.two_d_direction])

        return perovskite_guess