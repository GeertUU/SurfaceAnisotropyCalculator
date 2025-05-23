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


__version__ = '0.5.8'

from surfaceanisotropycalculator.anisotropyclass import MeshCalculator_legacy, MeshCalculator
from surfaceanisotropycalculator.stackanalysis import MeshFromFile, MeshFromStack, MeshFromLif, MeshFromImageStack

#make visible for 'from imageAnalysis import *'
__all__ = [
    'MeshCalculator',
    'MeshCalculator_legacy',
    'MeshFromFile',
    'MeshFromStack',
    'MeshFromLif',
    'MeshFromImageStack',
]
