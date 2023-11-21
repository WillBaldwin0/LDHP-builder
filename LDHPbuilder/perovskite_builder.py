from .molecule_utils import *
import numpy as np
from aseMolec.anaAtoms import find_molecs, split_molecs, wrap_molecs, scan_vol
from ase.atoms import Atoms


class OrganicMolecule:
    """ holds some data about molecules.
     the origin is shifted to the a 'bonding point'.
     
     for now, moleclues have two bonding points, one at each 'end' of the molecule. 
     These are stored in OrganicMolecule.furtherst_heavy_atoms which holds the two positions. """
    def __init__(self, ats, charge, smiles=None):
        self._atoms = deepcopy(ats)
        self.charge = charge
        self.smiles = smiles

        # find the furtherst sticking out atoms
        self.furtherst_heavy_atoms, self.long_vector = furtherst_heavy_atom_indices(ats)
        # we actually only want nitrogens
        self.bonding_atoms, self.to_flip = fix_to_only_nitrogens_with_cutoff(ats, self.furtherst_heavy_atoms)
        self.only_bond_to_nitrogens = True
        if self.to_flip[0] == self.to_flip[1]:
            self.number_of_bonding_points = 1
        else:
            self.number_of_bonding_points = 2


        # coorindate axes for rotations
        #self.coordinate_system = principle_axes_of_molecule(ats)
        #if np.dot(self.long_vector, self.coordinate_system[0]) < 0.0:
        #    self.coordinate_system = - self.coordinate_system
        #self.directed_coordinate_system = self.coordinate_system

        # coorindate axes for rotations
        principle_axes = principle_axes_of_molecule(ats)
        # construct new basis
        vec1 = np.cross(principle_axes[2], self.long_vector)
        vec2 = np.cross(vec1, self.long_vector)
        self.coordinate_system = np.asarray([
            -self.long_vector,
            vec1,
            vec2
        ])
        self.coordinate_system /= np.linalg.norm(self.coordinate_system, axis=1)[:,None]
        self.directed_coordinate_system = self.coordinate_system

    def get_atoms_shifted_rotated(self, bonding_index, vector, reference_vector):
        """ return the molecule, with bonding atom shifted to the origin, 
        and the molecule long axis aligned along vector """
        
        new_ats = deepcopy(self._atoms)
        new_ats.set_positions(new_ats.get_positions() - new_ats.get_positions()[self.bonding_atoms[bonding_index]])
        if self.to_flip[bonding_index]:
            directed_coordinate_system = - self.coordinate_system
        else:
            directed_coordinate_system = self.coordinate_system

        intermediate_1 = np.cross(vector, reference_vector)
        target_coordinate_system = np.asarray([
            vector, 
            -intermediate_1,
            np.cross(vector, intermediate_1)
        ])
        target_coordinate_system /= np.linalg.norm(target_coordinate_system, axis=1)[:,None]
        #txx = R.align_vectors(target_coordinate_system, directed_coordinate_system)
        #print("matrix = ", txx[0].as_matrix())
        mat = directed_coordinate_system.transpose() @ target_coordinate_system
        rotate_molecule(new_ats, mat)
        return new_ats
            
    def _rotation_matrix_onto_new_coords(self, new_coords):
        # this matrix left multiples column vectors
        roatation_matrix = self.directed_coordinate_system.transpose() @ new_coords
        return roatation_matrix
    
    def __repr__(self):
        string = f"OrganicMolecule: {self._atoms.get_chemical_formula()}"
        string += f"\n charge: {self.charge}"
        string += f"\n only_bond_to_nitrogens = {self.only_bond_to_nitrogens}"
        string += f"\n number_of_bonding_sites = {self.number_of_bonding_points}"
        string += f"\n given (not calculated) smiles = {self.smiles}"
        return string


class InorganicMonolayer:
    """ holds some extra data about monolayers.
    importantly, stores:
    - the normal vector of the layer
    - the index of the non-periodic direction of the cell matrix
     """
    def __init__(self, monolayer_ats):
        # we want a monolayer, we will at least check for a layer.
        self.check_input(monolayer_ats)

        self.atoms = deepcopy(monolayer_ats)
        self.fitted_normal, self.two_d_direction = find_inorganic_layer_normal(
            monolayer_ats
        )
        self.ps_lattice_constants = np.asarray(
            get_2d_pseudocubic_lattice_vectors(monolayer_ats, self.two_d_direction, self.fitted_normal)
        )
        new_lc2 = np.cross(self.fitted_normal, self.ps_lattice_constants[0])
        local_basis = np.asarray([
            np.cross(self.fitted_normal, new_lc2), 
            new_lc2,
            self.fitted_normal
        ])
        self.local_basis = local_basis / np.linalg.norm(local_basis, axis=1)[:,None]
        syms = np.asarray(monolayer_ats.get_chemical_symbols())
        self.lead_positions = monolayer_ats.get_positions()[syms == 'Pb']
        self.atoms.cell[self.two_d_direction] *= 3.0
        self.layer_charge = -2 * self.lead_positions.shape[0]

    @classmethod
    def from_species_specification(cls, B_site, X_site, num_unit_cell_octahedra):
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
        ase.io.write('monolayer.xyz', monolayer)
        return cls(monolayer)


    def get_bonding_points(self, normal_displacement=3.5):
        """ returns the coordinates of the points in between the octahedra, displaced by 
        normal_displacement relative to the centerline of the monolayer.
        north and south refer to the two sides of the layer. """
        bonding_points_north = []
        bonding_points_south = []
        ps1 = self.ps_lattice_constants[0]
        ps2 = self.ps_lattice_constants[1]

        for i in range(self.lead_positions.shape[0]):
            bonding_points_north.append(self.lead_positions[i] + self.fitted_normal * normal_displacement + 0.5*(ps1+ps2))
            bonding_points_south.append(self.lead_positions[i] - self.fitted_normal * normal_displacement + 0.5*(ps1+ps2))

        return bonding_points_north, bonding_points_south
    
    def check_input(self, ats_in):
        if not(isinstance(ats_in, ase.atoms.Atoms)):
            raise ValueError('monolayer input must be atoms object')
        
    def __repr__(self):
        string = "InorganicMonolayer: "
        string += f"\n {self.atoms.get_chemical_formula()}"
        string += f"\n normal = {self.fitted_normal}, 2d axis is direction {self.two_d_direction}"
        return string
    


class PerovskiteBuilder:
    """ class for assembling perovskites from molecules and inorganic layers.
    implementation is currently a collection of ad-hock rules, but gives decent results.
     
    the fundamental way in which perovskites are built involves first placing molecules at given sites, 
    - choosing which part of the molecule goes into the site
    - choosing whether to apply reflections to the molecule
     """
    def __init__(self, inorganic_layer, molecule):
        self.layer = inorganic_layer
        self.molecule = molecule
        assert (self.layer.lead_positions.shape[0] in [1,2,4,8]) # we only want powers of two please. 

    def __repr__(self):
        string = "PerovskiteBuilder:"
        string += f"\n inorganic: {self.layer.atoms.get_chemical_formula()}"
        string += f"\n organic: {self.molecule._atoms.get_chemical_formula()}"
        return string
    
    def reduced_random_binary_array(self, n): 
        # n must be a power of 2
        assert ((n & (n-1) == 0) and n != 0) # funky

        exp = int(np.log2(n))
        stuff = [np.random.choice([True,False])]
        for i in range(exp):
            stuff += stuff
            if np.random.choice([0,1]):
                stuff[2**i:] = list(np.logical_not(stuff[2**i:]))
        return np.asarray(stuff)

    def generate_ats(
        self,
        num_samples=1, 
        max_num_attempts=500,
        apply_shear=False,
        try_squash=False
    ):
        num_layers = 1

        if self.molecule.charge == 1:
            f = self._generate_guess_charge_1
        elif self.molecule.charge == 2:
            f = self._generate_guess_charge_2
        else:
            assert 0
        
        num_leads = self.layer.lead_positions.shape[0]
        num_molecules = (2*num_leads) // self.molecule.charge
        top_layer_bonding_points, bottom_layer_bonding_points = self.layer.get_bonding_points(normal_displacement=4.0)
        
        # create exaustive list 
        perovskite_structures = []
        num_attempted_orientations = 0

        while (num_attempted_orientations < max_num_attempts) and (len(perovskite_structures) < num_samples):

            molecule_bonding_points = self.reduced_random_binary_array(num_molecules)

            # reflections are in the two in plane directions. desrcibed by [n,m]. n=0,1. [1,1] means reflect in both
            reflections = self.reduced_random_binary_array(num_molecules)
            reflections = np.vstack((self.reduced_random_binary_array(num_molecules), reflections)).transpose()
            
            inner_counter = 0
            while inner_counter < 10: # try hard for lower symmetry cases
                molecule_long_vector = random_points_on_cap(45, 1, self.layer.fitted_normal)[0] # molecules share this vector

                ats = f(
                    molecule_long_vector,
                    top_layer_bonding_points,
                    bottom_layer_bonding_points,
                    molecule_bonding_points,
                    1,
                    0,
                    reflections,
                    apply_shear
                )
                if True: # check_molecule_intersection(ats, num_molecules) and check_mol_to_inorganic_intersections(ats):
                    perovskite_structures.append(ats)
                    inner_counter = 10
                else:
                    inner_counter += 1

            num_attempted_orientations +=1

        if num_attempted_orientations == max_num_attempts:
            warnings.warn( f"reached max number of attempts having only generated {len(perovskite_structures)} structures." )
        else:
            print(f"generated {num_samples} samples after {num_attempted_orientations} attempts")

        if try_squash:
            for struc in perovskite_structures:
                self._attempt_to_squash(struc)

        return perovskite_structures

    
    def _generate_guess_charge_2(
        self,
        mol_vector,
        top_layer_bonding_points,
        bottom_layer_bonding_points,
        molecule_bonding_points,
        num_layers, 
        layer_shifts,
        molecule_refections,
        apply_shear,
    ):
        """
        mol_vector: vector align molecule along
        layer_bonding_points: list of coordinates to dock on (w.r.t first layer)
        molecule_bonding_points: list of indices, either 0 or 1
        num_layers: int
        layer_shifts: list of vectors, one for each additional layer
        molecule_refections: list of lists. list j contains the sequence of relfections to be applied to molecule j
        """
        # get final rotation matrix
        if np.random.choice([True, False]):
            rr = R.from_rotvec(np.pi/4 * self.layer.fitted_normal / np.linalg.norm(self.layer.fitted_normal))
            mat = rr.as_matrix()
        else:
            mat = np.eye(3)

        atoms = deepcopy(self.layer.atoms)
        for (layer_bp, mol_bp, reflection) in zip(
            top_layer_bonding_points, 
            molecule_bonding_points, 
            molecule_refections
        ):
            mol = self.molecule.get_atoms_shifted_rotated(mol_bp, mol_vector)
            mol_cp = deepcopy(mol)
            for i in range(2):
                if reflection[i]:
                    normal = self.layer.ps_lattice_constants[i]
                    refect_molecule(mol_cp, normal)
            rotate_molecule(mol_cp, mat)
            mol_cp.set_positions(mol_cp.get_positions() + layer_bp)
            atoms.extend(mol_cp)

        if apply_shear:
            periodic_directions = np.array(list(set([0,1,2]) - set([self.layer.two_d_direction])))
            cell = atoms.get_cell()[:]
            rand_vec = np.random.randn((2))
            rand_vec = np.clip(rand_vec, -1.25, 1.25)
            added_vector = cell[periodic_directions].transpose() @ rand_vec * 0.75
            cell[self.layer.two_d_direction] += added_vector
            atoms.set_cell(cell)
        
        atoms.center(vacuum=1.5, axis=[self.layer.two_d_direction])
        return atoms


    def _generate_guess_charge_1(
        self,
        mol_vector,
        top_layer_bonding_points,
        bottom_layer_bonding_points,
        molecule_bonding_points,
        num_layers, 
        layer_shifts,
        molecule_refections, # list of lists. for each molecule, for each ps_direction, True if reflect, False if not. 
        apply_shear
    ):
        atoms = deepcopy(self.layer.atoms)

        # get final rotation matrix
        if np.random.choice([True, False]):
            rr = R.from_rotvec(np.pi/4 * self.layer.fitted_normal / np.linalg.norm(self.layer.fitted_normal))
            mat = rr.as_matrix()
        else:
            mat = np.eye(3)

        for (layer_bp, mol_bp, reflection) in zip(
            top_layer_bonding_points, 
            molecule_bonding_points[:len(top_layer_bonding_points)], 
            molecule_refections[:len(top_layer_bonding_points)]
        ):
            mol = self.molecule.get_atoms_shifted_rotated(mol_bp, mol_vector)
            mol_cp = deepcopy(mol)
            for i in range(2):
                if reflection[i]:
                    normal = self.layer.ps_lattice_constants[i]
                    refect_molecule(mol_cp, normal)
                    rotate_molecule(mol_cp, mat)
            mol_cp.set_positions(mol_cp.get_positions() + layer_bp)
            atoms.extend(mol_cp)
        
        pre_center = atoms.get_positions()[0,self.layer.two_d_direction]
        atoms.center(vacuum=1.8, axis=[self.layer.two_d_direction])
        disp = np.zeros(3)
        disp[self.layer.two_d_direction] = atoms.get_positions()[0,self.layer.two_d_direction] - pre_center

        for (layer_bp, mol_bp, reflection) in zip(
            bottom_layer_bonding_points, 
            molecule_bonding_points[len(top_layer_bonding_points):], 
            molecule_refections[len(top_layer_bonding_points):]
        ):
            mol = self.molecule.get_atoms_shifted_rotated(mol_bp, -mol_vector)
            mol_cp = deepcopy(mol)
            for i in range(2):
                if reflection[i]:
                    normal = self.layer.ps_lattice_constants[i]
                    refect_molecule(mol_cp, normal)
                    rotate_molecule(mol_cp, mat)
            mol_cp.set_positions(mol_cp.get_positions() + layer_bp + disp)
            atoms.extend(mol_cp)
        
        if apply_shear:
            periodic_directions = np.array(list(set([0,1,2]) - set([self.layer.two_d_direction])))
            cell = atoms.get_cell()
            rand_vec = np.random.randn((2))
            rand_vec = np.clip(rand_vec, -1.25, 1.25)
            added_vector = cell[periodic_directions].transpose() @ rand_vec * 0.75
            cell[self.layer.two_d_direction] += added_vector
            atoms.set_cell(cell)

        return atoms
    

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




class NewPerovskiteBuilder:
    """ class for assembling perovskites from molecules and inorganic layers.
    implementation is currently a collection of ad-hock rules, but gives decent results.
     
    the fundamental way in which perovskites are built involves first placing molecules at given sites, 
    - choosing which part of the molecule goes into the site
    - choosing whether to apply reflections to the molecule
     """
    def __init__(self, inorganic_layer, molecule):
        self.layer = inorganic_layer
        self.molecule = molecule
        assert (self.layer.lead_positions.shape[0] in [1,2,4,8]) # we only want powers of two please. 

    def __repr__(self):
        string = "PerovskiteBuilder:"
        string += f"\n inorganic: {self.layer.atoms.get_chemical_formula()}"
        string += f"\n organic: {self.molecule._atoms.get_chemical_formula()}"
        return string
    
    def reduced_random_binary_array(self, n): 
        # n must be a power of 2
        assert ((n & (n-1) == 0) and n != 0) # funky

        exp = int(np.log2(n))
        stuff = [np.random.choice([True,False])]
        for i in range(exp):
            stuff += stuff
            if np.random.choice([0,1]):
                stuff[2**i:] = list(np.logical_not(stuff[2**i:]))
        return np.asarray(stuff)

    def generate_ats(
        self,
        num_samples=1, 
        max_num_attempts=500,
        apply_shear=False,
        try_squash=False,
        num_layers = 1
    ):

        if self.molecule.charge == 1:
            f = self._generate_guess_charge_1
        elif self.molecule.charge == 2:
            f = self._generate_guess_charge_2
        else:
            assert 0
        
        num_leads = self.layer.lead_positions.shape[0]
        num_molecules = (2*num_leads) * num_layers // self.molecule.charge
        top_layer_bonding_points, bottom_layer_bonding_points = self.layer.get_bonding_points(normal_displacement=4.0)
        
        # create exaustive list 
        perovskite_structures = []
        num_attempted_orientations = 0

        while (num_attempted_orientations < max_num_attempts) and (len(perovskite_structures) < num_samples):

            molecule_bonding_points = self.reduced_random_binary_array(num_molecules)

            # reflections are in the two in plane directions. desrcibed by [n,m]. n=0,1. [1,1] means reflect in both
            reflections = self.reduced_random_binary_array(num_molecules)
            reflections = np.vstack((self.reduced_random_binary_array(num_molecules), reflections)).transpose()

            inner_counter = 0
            while inner_counter < 10: # try hard for lower symmetry cases
                molecule_long_vector = random_points_on_cap(45, 1, self.layer.fitted_normal)[0] # molecules share this vector
                ats = f(
                    molecule_long_vector,
                    top_layer_bonding_points,
                    bottom_layer_bonding_points,
                    molecule_bonding_points[:num_molecules//num_layers],
                    1,
                    0,
                    reflections[:num_molecules//num_layers],
                    apply_shear
                )
                for layer_counter in range(1, num_layers):
                    # make a new molecule vector
                    molecule_long_vector = random_points_on_cap(45, 1, self.layer.fitted_normal)[0]
                    layer = f(
                        molecule_long_vector,
                        top_layer_bonding_points,
                        bottom_layer_bonding_points,
                        molecule_bonding_points[(num_molecules//num_layers)*layer_counter:(num_molecules//num_layers)*(layer_counter+1)],
                        1,
                        0,
                        reflections[(num_molecules//num_layers)*layer_counter:(num_molecules//num_layers)*(layer_counter+1)],
                        apply_shear
                    )
                    layer.set_positions(layer.get_positions() + layer_counter * ats.cell[self.layer.two_d_direction])
                    ats.extend(layer)

                ats.center(vacuum=1.0, axis=[self.layer.two_d_direction])

                if check_molecule_intersection(ats, num_molecules) and check_mol_to_inorganic_intersections(ats):
                    print('success')
                    perovskite_structures.append(ats)
                    inner_counter = 10
                else:
                    inner_counter += 1
                    if False:
                        inner_counter = 10
                        perovskite_structures.append(ats)

            num_attempted_orientations +=1

        if num_attempted_orientations == max_num_attempts:
            warnings.warn( f"reached max number of attempts having only generated {len(perovskite_structures)} structures." )
        else:
            print(f"generated {num_samples} samples after {num_attempted_orientations} attempts")

        if try_squash:
            for struc in perovskite_structures:
                self._attempt_to_squash(struc)

        return perovskite_structures

    
    def _generate_guess_charge_2(
        self,
        mol_vector,
        top_layer_bonding_points,
        bottom_layer_bonding_points,
        molecule_bonding_points,
        num_layers, 
        layer_shifts,
        molecule_refections,
        apply_shear,
    ):
        atoms = deepcopy(self.layer.atoms)
        # get basis for reflections
        _mol = self.molecule.get_atoms_shifted_rotated(0, mol_vector, self.layer.ps_lattice_constants[0])
        _mol_obj = OrganicMolecule(_mol, 2)
        reflection_basis = _mol_obj.directed_coordinate_system

        axial_rotation = np.random.uniform(low=0., high=2*np.pi)
        axial_rotation_matrix = R.from_rotvec(axial_rotation * reflection_basis[0]).as_matrix()

        #print(mol_vector)
        #print(reflection_basis)
        #molecule_refections = [[False, False], [False, False]]
        #molecule_bonding_points = [False, False]

        for (layer_bp, mol_bp, reflection) in zip(
            top_layer_bonding_points, 
            molecule_bonding_points, 
            molecule_refections
        ):
            mol = self.molecule.get_atoms_shifted_rotated(mol_bp, mol_vector, self.layer.ps_lattice_constants[0])
            mol_cp = deepcopy(mol)
            rotate_molecule(mol_cp, axial_rotation_matrix)
            for i in range(2):
                if reflection[i]:
                    normal = reflection_basis[i+1]
                    refect_molecule(mol_cp, normal)
            
            mol_cp.set_positions(mol_cp.get_positions() + layer_bp)
            atoms.extend(mol_cp)

        if apply_shear:
            periodic_directions = np.array(list(set([0,1,2]) - set([self.layer.two_d_direction])))
            cell = atoms.get_cell()[:]
            rand_vec = np.random.randn((2))
            rand_vec = np.clip(rand_vec, -1.25, 1.25)
            added_vector = cell[periodic_directions].transpose() @ rand_vec * 0.75
            cell[self.layer.two_d_direction] += added_vector
            atoms.set_cell(cell)
        
        atoms.center(vacuum=1.5, axis=[self.layer.two_d_direction])
        return atoms


    def _generate_guess_charge_1(
        self,
        mol_vector,
        top_layer_bonding_points,
        bottom_layer_bonding_points,
        molecule_bonding_points,
        num_layers, 
        layer_shifts,
        molecule_refections, # list of lists. for each molecule, for each ps_direction, True if reflect, False if not. 
        apply_shear
    ):
        atoms = deepcopy(self.layer.atoms)
        # get basis for reflections
        _mol = self.molecule.get_atoms_shifted_rotated(0, mol_vector, self.layer.ps_lattice_constants[0])
        _mol_obj = OrganicMolecule(_mol, 2)
        reflection_basis = _mol_obj.directed_coordinate_system

        axial_rotation = np.random.uniform(low=0., high=2*np.pi)
        axial_rotation_matrix = R.from_rotvec(axial_rotation * reflection_basis[0]).as_matrix()

        for (layer_bp, mol_bp, reflection) in zip(
            top_layer_bonding_points, 
            molecule_bonding_points[:len(top_layer_bonding_points)], 
            molecule_refections[:len(top_layer_bonding_points)]
        ):
            mol = self.molecule.get_atoms_shifted_rotated(mol_bp, mol_vector, self.layer.ps_lattice_constants[0])
            mol_cp = deepcopy(mol)
            rotate_molecule(mol_cp, axial_rotation_matrix)
            for i in range(2):
                if reflection[i]:
                    normal = reflection_basis[i+1]
                    refect_molecule(mol_cp, normal)
            mol_cp.set_positions(mol_cp.get_positions() + layer_bp)
            atoms.extend(mol_cp)
        
        pre_center = atoms.get_positions()[0,self.layer.two_d_direction]
        atoms.center(vacuum=1.0, axis=[self.layer.two_d_direction])
        disp = np.zeros(3)
        disp[self.layer.two_d_direction] = atoms.get_positions()[0,self.layer.two_d_direction] - pre_center

        for (layer_bp, mol_bp, reflection) in zip(
            bottom_layer_bonding_points, 
            molecule_bonding_points[len(top_layer_bonding_points):], 
            molecule_refections[len(top_layer_bonding_points):]
        ):
            mol = self.molecule.get_atoms_shifted_rotated(mol_bp, -mol_vector, -self.layer.ps_lattice_constants[0])
            mol_cp = deepcopy(mol)
            rotate_molecule(mol_cp, axial_rotation_matrix)
            for i in range(2):
                if reflection[i]:
                    normal = reflection_basis[i+1]
                    refect_molecule(mol_cp, normal)
            mol_cp.set_positions(mol_cp.get_positions() + layer_bp + disp)
            atoms.extend(mol_cp)

        atoms.center(vacuum=1.0, axis=[self.layer.two_d_direction])
        
        if apply_shear:
            periodic_directions = np.array(list(set([0,1,2]) - set([self.layer.two_d_direction])))
            cell = atoms.get_cell()
            rand_vec = np.random.randn((2))
            rand_vec = np.clip(rand_vec, -1.25, 1.25)
            added_vector = cell[periodic_directions].transpose() @ rand_vec * 0.75
            cell[self.layer.two_d_direction] += added_vector
            atoms.set_cell(cell)

        return atoms
    

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