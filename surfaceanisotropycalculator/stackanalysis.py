# -*- coding: utf-8 -*-
"""
    Copyright (C) 2024-2025  Geert Schulpen
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
"""

import numpy as np

import igl
import skimage

from readlif.reader import LifFile
from surfaceanisotropycalculator.anisotropyclass import MeshCalculator_legacy
from surfaceanisotropycalculator.anisotropyclass import MeshCalculator


    
def createoptimalMesh(image, test='auto', **kwargs):
    """
    Find the threshold at which the mesh area is maximized

    Parameters
    ----------
    image : 3D or 4D array
        Image stack.
    channel : INT, optiional
        Which channel to use of the image. If the image has just one channel 
        this parmeter has no effect. The default is 0.
    test : iterable, optional
        An iterable containing all threshold values that should be tested. The
        default is 'auto', for which the function automatically generates an
        iterator containing all the integers between the smallest and biggest 
        intensity values in the image.
    **kwargs : dict
        Any optional/keyword arguments that should be passed to
        skimage.measure.marching_cubes.

    Returns
    -------
    bestmesh : tuple of 4 ndarrays
        Output of skimage.measure.marching_cubes for the mesh with the largest
        surface area.
    maxarea : float
        Surface area of bestmesh.
    correcttreshold : float
        Threshold value at which bestmesh is found.

    """

    if test == 'auto':
        info = [int(np.max(image)), int(np.min(image))]
        test = range(info[1], info[0])
    
    maxarea = 0
    bestmesh = ()
    try:
        for level in test:
            mymesh = skimage.measure.marching_cubes(image, level, **kwargs)
            myarea = skimage.measure.mesh_surface_area(mymesh[0], mymesh[1])
            if myarea > maxarea:
                maxarea = myarea
                bestmesh = mymesh
                correcttreshold = level
    except TypeError:
        bestmesh = skimage.measure.marching_cubes(image, test, **kwargs)
        maxarea = skimage.measure.mesh_surface_area(bestmesh[0], bestmesh[1])
        correcttreshold = test

    return bestmesh, maxarea, correcttreshold


def MeshFromFile(filename, legacy=False):
    """
    Wrapper function to import a mesh from a file and select correct base class

    Parameters
    ----------
    filename : str
        Path + filename + extension of the mesh file. Supported extensions
        .obj, .off, .stl, .wrl, .ply, .mesh.
    legacy : bool, optional
        Wether to use the "MeshCalculator" (False, default) or 
        "MeshCalculater_legacy" (True) class. The legacy version is more memory
        intensive.

    Returns
    -------
        Instance of the "MeshCalculator" or "MeshCalculator_legacy" class.
    """
    v, f = igl.read_triangle_mesh(filename)
    nv, _, _, nf = igl.remove_duplicate_vertices(v, f, 1e-7)
    if legacy:
        mesh = MeshCalculator_legacy(v, f)
    else:
        mesh = MeshCalculator(v, f)
    mesh.filename = filename
    return mesh


def MeshFromLif(filename, channel, n=0, test='auto', legacy=False, **kwargs):
    """
    Import mesh from an image stack in .lif format and select correct 
    MeshCalculator class.
    
    Parameters
    ----------
    filename : str
        Path + filename + extension (.lif) of the file. 
    channel : INT
        Which channel to use of the image stack.
    n : INT, optional
        What image in the lif file is to be used? The default is 0, the
        first image
    test : iterable, optional
        An iterable containing all threshold values that should be tested. The
        default is 'auto', for which the function automatically generates an
        iterator containing all the integers between the smallest and biggest 
        intensity values in the image.
    legacy : bool, optional
        Wether to use the "MeshCalculator" (False, default) or 
        "MeshCalculater_legacy" (True) class. The legacy version is more memory
        intensive.
    **kwargs : dict
        Any optional/keyword arguments that should be passed to
        skimage.measure.marching_cubes. 

    Returns
    -------
        Instance of the "MeshCalculator" or "MeshCalculator_legacy" class.
    """
    myimg = LifFile(filename).get_image(n)
    imgsize = (myimg.dims_n[1], myimg.dims_n[2])
    img = np.fromiter(myimg.get_iter_z(c=channel), np.dtype((int, imgsize)))
    scales = myimg.scale
    spacing = [1/np.abs(x) for x in scales[2::-1]]
    
    return MeshFromImageStack(img, channel, test, legacy, spacing=spacing, **kwargs)


def MeshFromStack(filename, channel, test='auto', legacy=False, **kwargs):
    """
    Import mesh from an image stack with manual spacings and select correct 
    MeshCalculator class.
    
    Parameters
    ----------
    filename : str
        Path + filename + extension of the image stack file. 
    channel : INT
        Which channel to use of the image stack.
    test : iterable, optional
        An iterable containing all threshold values that should be tested. The
        default is 'auto', for which the function automatically generates an
        iterator containing all the integers between the smallest and biggest 
        intensity values in the image.
    legacy : bool, optional
        Wether to use the "MeshCalculator" (False, default) or 
        "MeshCalculater_legacy" (True) class. The legacy version is more memory
        intensive.
    **kwargs : dict
        Any optional/keyword arguments that should be passed to
        skimage.measure.marching_cubes.

    Returns
    -------
        Instance of the "MeshCalculator" or "MeshCalculator_legacy" class.
    """

    myimg = skimage.io.imread(filename)
    return MeshFromImageStack(myimg, channel, test, legacy, **kwargs)
        
        
def MeshFromImageStack(image, channel, test='auto', legacy=False, **kwargs):
    """
    Import mesh from an image stack with manual spacings and select correct 
    MeshCalculator class.
    
    Parameters
    ----------
    image : 3D or 4D array
        Image stack. 
    channel : INT
        Which channel to use of the image stack.
    test : iterable, optional
        An iterable containing all threshold values that should be tested. The
        default is 'auto', for which the function automatically generates an
        iterator containing all the integers between the smallest and biggest 
        intensity values in the image.
    legacy : bool, optional
        Wether to use the "MeshCalculator" (False, default) or 
        "MeshCalculater_legacy" (True) class. The legacy version is more memory
        intensive.
    **kwargs : dict
        Any optional/keyword arguments that should be passed to
        skimage.measure.marching_cubes.

    Returns
    -------
        Instance of the "MeshCalculator" or "MeshCalculator_legacy" class.
    """
    
    try:
        grayimg = image[:,:,:, channel]
    except IndexError:
        grayimg = image[:,:,:]
    mesh, area, threshold = createoptimalMesh(grayimg, test, **kwargs)
    print(f'Best mesh at threshold {threshold}')
    v, f, normals, values = mesh
    nv, _, _, nf = igl.remove_duplicate_vertices(v, f, 1e-7)
    if legacy:
        mesh = MeshCalculator_legacy(nv, nf)
    else:
        mesh = MeshCalculator(nv, nf)
    # mesh.filename = filename
    return mesh

