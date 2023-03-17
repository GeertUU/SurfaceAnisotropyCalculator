# SurfaceAnisotropyCalculator
Class to deal with a mesh and calculations on it.
The project is in a very rudimentary state.

## Info
- Created by: Geert Schulpen
- Email: g.h.a.schulpen@uu.nl
- Version: 0.1.0


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

0.1.0
The `MeshCalculator.cropmesh()` method now actually cleans up verteces that are outside the specified region. 
