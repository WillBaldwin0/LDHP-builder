# LDHP-builder

This package provides tools for randomly generating low dimensional hybrid organic inorganic perovskites. See out preprint for example usage https://arxiv.org/abs/2403.06955

## Usage

The simplest intended use is to create random 2D hybrid perovskite structures, given an organic cation, a choice of halide, and some details about the unit cell.

First create an inorganic monolayer

```python
from LDHPbuilder.perovskite_builder import OrganicMolecule, InorganicMonolayer, PerovskiteBuilder

monolayer = InorganicMonolayer.from_species_specification('Pb', 'Br', num_unit_cell_octahedra=2)
```
`num_unit_cell_octahedra` is the number of lead atoms in each monolayer, in each unit cell. The `InorganicMonolayer` computes some useful properties when given an atoms object representing a monolayer. It also provides the useful class method `from_species_specification`.  

Then construct and `OrganicMolecule` from an `ase` `Atoms` object.

```python
molecule = OrganicMolecule(
    atoms,
    1. # charge
)
```

Random structures are then created:
```python
pb = PerovskiteBuilder()

samples = pb.generate_homogeneous_perovskite_samples( # homogeneous since just one kind of molecule
    monolayer, 
    molecule, 
    num_layers=2, # should be a power of 2, but 4 already implies a very large search task
    num_samples=10, 
    max_num_attempts=15, 
    stacking_method='total_thickness' # this can be either 'total_thickness' or 'half_thickness'. use half_thickness for long, thin +1 molecules
)
```

## Documentation

Key functions are documented through docstrings. 

## Contributing and Feature Requests

If you want to contribute to, or use our package for a scientific application, please feel free to contact us via the emails in the citation.

There are many extensions which could be easily added, please let us know via an issue.

## Dependencies

As well as `ase`, `numpy`, `scipy`, We use `aseMolec` for working with `ase.Atoms` objects representing organic molecules. https://github.com/imagdau/aseMolec.
