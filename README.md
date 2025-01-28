# SurfaceAnisotropyCalculator
Class to deal with a mesh and calculations on it.
The project is in a very rudimentary state. The intended scope of the project is to allow for easy calculations of (relevant) Minkowski tensors, as described in Schröder-Turk _et al_. "Tensors of Anisotropic Spatial Structure" New J. Phys. 2013, https://doi.org/10.1088/1367-2630/15/8/083028. 


## Info
- Created by: Geert Schulpen
- Email: g.h.a.schulpen@uu.nl
- Version: 0.3.9


## Installation

### PIP
This package can be installed directly from GitHub using pip:
```
pip install git+https://github.com/GeertUU/SurfaceAnisotropyCalculator
```
### Anaconda
When using the Anaconda distribution, it is safer to run the conda version of pip as follows:
```
conda install pip
conda install git
pip install git+https://github.com/GeertUU/SurfaceAnisotropyCalculator
```
### Updating
To update to the most recent version, use `pip install` with the `--upgrade` flag set:
```
pip install --upgrade git+https://github.com/GeertUU/SurfaceAnisotropyCalculator
```



## Changelog
0.3.8
Added scaled genus calculator

0.3.6
Added a stack processor that works from an array

0.3.5
Added image selection for .lif files that have multiple images and automatic recognition of number of pixels in x and y dimensions

0.3.4
Added functionality to calculate anisotropy paramater alpha (1-beta)

0.3.0
Added functionality to read in data from confocal stacks and find the correct mesh

0.2.0
Minkowski scalar W2 can now be calculated
The pandas package is actually listed as an install requirement now

0.1.10
Removed external verteces from W3 calculations

0.1.9
Neighbor lists converted to sets

0.1.0
The whole internal structure is changed to allow for more complex calculations. The class now uses Pandas dataframes.
The minkowski scalars W1 and W3 can now be calculated.
The `MeshCalculator.cropmesh()` method now actually cleans up verteces that are outside the specified region and removes verteces which are not part of any faces.
