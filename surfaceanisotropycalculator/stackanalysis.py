# -*- coding: utf-8 -*-
"""
    Copyright (C) 2023  Geert Schulpen
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

from surfaceanisotropycalculator.anisotropyclass import MeshCalculator

def createMesh(image, channel, threshold, **kwargs):
    '''
    Wrapper function for skimage.measure.marching_cubes

    Parameters
    ----------
    image : STR
        Imagefile (including path).
    channel : INT
        Which channel to use of the image.
    threshold : float
        intensitylevel used as separation value.
    **kwargs : dict
        Any optional/keyword arguments that should be passed to
        skimage.measure.marching_cubes.

    Returns
    -------
    mymesh : TYPE
        DESCRIPTION.

    '''
    myimg = skimage.io.imread(image)
    grayimg = myimg[:,:,:, channel]
    mymesh = skimage.measure.marching_cubes(grayimg, threshold, **kwargs)
    return mymesh

def createoptimalMesh(image, channel=0, test='auto', **kwargs):
    """
    Find the threshold at which the mesh area is maximized

    Parameters
    ----------
    image : STR
        Imagefile (including path).
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
    myimg = skimage.io.imread(image)
    try:
        grayimg = myimg[:,:,:, channel]
    except IndexError:
        grayimg = myimg[:,:,:]
    if test == 'auto':
        info = np.iinfo(myimg.dtype)
        test = range(info.min, info.max)
    
    maxarea = 0
    bestmesh = ()
    try:
        for level in test:
            mymesh = skimage.measure.marching_cubes(grayimg, level, **kwargs)
            myarea = skimage.measure.mesh_surface_area(mymesh[0], mymesh[1])
            if myarea > maxarea:
                maxarea = myarea
                bestmesh = mymesh
                correcttreshold = level
    except TypeError:
        bestmesh = createMesh(image, channel, test, **kwargs)
        maxarea = skimage.measure.mesh_surface_area(bestmesh[0], bestmesh[1])
        correcttreshold = test

    return bestmesh, maxarea, correcttreshold


class MeshFromStack(MeshCalculator):
    """
    Simple wrapper class to import mesh from an image stack
    
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
        **kwargs : dict
            Any optional/keyword arguments that should be passed to
            skimage.measure.marching_cubes.

    Returns
    -------
        None.
    """
    def __init__(self, image, channel, test='auto', **kwargs):
        """
        Initialize class instance
        """
        mesh, area, threshold = createoptimalMesh(image, channel, test, **kwargs)
        print(f'Best mesh at threshold {threshold}')
        v, f, normals, values = mesh
        nv, _, _, nf = igl.remove_duplicate_vertices(v, f, 1e-7)
        MeshCalculator.__init__(self, nv, nf)
        
