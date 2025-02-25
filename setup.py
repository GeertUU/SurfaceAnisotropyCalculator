'''
    Copyright (C) 2023-2025  Geert Schulpen
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''



from setuptools import setup, find_packages
from Cython.Build import cythonize


setup(
    name = 'surfaceanisotropycalculator',
    version = '0.5.0',

    url='https://github.com/GeertUU/SurfaceAnisotropyCalculator',
    author='Geert',
    author_email='g.h.a.schulpen@uu.nl',
    
    ext_modules = cythonize("surfaceanisotropycalculator\CythonFunctions.pyx")

    packages=find_packages(include=["surfaceanisotropycalculator", "surfaceanisotropycalculator.*"]),
    
    install_requires=[
        'numpy',
        'libigl',
        'pandas',
        'scikit-image',
        'readlif',
        'Cython' 
    ],
)
