# LDHP-builder

This package provides tools for randomly generating low dimensional hybrid organic inorganic perovskites, as used in https://pubs.acs.org/doi/10.1021/jacs.4c06549.

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

## Citing

This code was introduced in the following paper:

```bibtext
@article{KarimitariBaldwin2024,
    author = {Karimitari, Nima and Baldwin, William J. and Muller, Evan W. and Bare, Zachary J. L. and Kennedy, W. Joshua and Csányi, Gábor and Sutton, Christopher},
    title = {Accurate Crystal Structure Prediction of New 2D Hybrid Organic–Inorganic Perovskites},
    journal = {Journal of the American Chemical Society},
    volume = {146},
    number = {40},
    pages = {27392-27404},
    year = {2024},
    doi = {10.1021/jacs.4c06549},
    note ={PMID: 39344597},
    URL = {https://doi.org/10.1021/jacs.4c06549}
}
```

## Contributing and Feature Requests

If you want to contribute to, or use our package for a scientific application, please feel free to contact us via the emails in the citation.

There are many extensions which could be easily added, please let us know via an issue.

## Dependencies

As well as `ase`, `numpy`, `scipy`, We use `aseMolec` for working with `ase.Atoms` objects representing organic molecules. https://github.com/imagdau/aseMolec.
