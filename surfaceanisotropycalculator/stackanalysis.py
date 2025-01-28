# -*- coding: utf-8 -*-
"""
    Copyright (C) 2024  Geert Schulpen
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
from surfaceanisotropycalculator.anisotropyclass import MeshCalculator, MeshCalculatorLowerMemory



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


class MeshFromLif(MeshCalculatorLowerMemory):
    """
    Simple wrapper class to import mesh from an image stack in .lif format.
    
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
        lowmem : bool
            Wether to use the "MeshCalculatorLowerMemory". The lower memory
            class does not create a dataframe with information on edges.
        **kwargs : dict
            Any optional/keyword arguments that should be passed to
            skimage.measure.marching_cubes. 

    Returns
    -------
        None.
    """
    def __init__(self, filename, channel, n=0, test='auto', lowmem=False, **kwargs):
        """
        Initialize class instance
        """    
        self.edgesinmemory = not lowmem
        myimg = LifFile(filename).get_image(n)
        imgsize = (myimg.dims_n[1], myimg.dims_n[2])
        img = np.fromiter(myimg.get_iter_z(c=channel), np.dtype((int, imgsize)))
        scales = myimg.scale
        spacing = [1/np.abs(x) for x in scales[2::-1]]
        
        mesh, area, threshold = createoptimalMesh(img, test, spacing = spacing, **kwargs)
        print(f'Best mesh at threshold {threshold}')
        v, f, normals, values = mesh
        nv, _, _, nf = igl.remove_duplicate_vertices(v, f, 1e-7)
        
        if self.edgesinmemory:
            MeshCalculator.__init__(self, nv, nf)
        else:
            MeshCalculatorLowerMemory.__init__(self, nv, nf)
            
        
    def getW2(self):
        if self.edgesinmemory:
            MeshCalculator.getW2(self)
        else:
            MeshCalculatorLowerMemory.getW2(self)
        




class MeshFromStack(MeshCalculatorLowerMemory):
    """
    Simple wrapper class to import mesh from an image stack with manual spacings
    
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
        lowmem : bool
            Wether to use the "MeshCalculatorLowerMemory". The lower memory
            class does not create a dataframe with information on edges.
        **kwargs : dict
            Any optional/keyword arguments that should be passed to
            skimage.measure.marching_cubes.

    Returns
    -------
        None.
    """
    def __init__(self, filename, channel, test='auto', lowmem=False, **kwargs):
        """
        Initialize class instance
        """
        self.edgesinmemory = not lowmem
        myimg = skimage.io.imread(filename)
        try:
            grayimg = myimg[:,:,:, channel]
        except IndexError:
            grayimg = myimg[:,:,:]
        mesh, area, threshold = createoptimalMesh(grayimg, test, **kwargs)
        print(f'Best mesh at threshold {threshold}')
        v, f, normals, values = mesh
        nv, _, _, nf = igl.remove_duplicate_vertices(v, f, 1e-7)
        if self.edgesinmemory:
            MeshCalculator.__init__(self, nv, nf)
        else:
            MeshCalculatorLowerMemory.__init__(self, nv, nf)
            
        
    def getW2(self):
        if self.edgesinmemory:
            MeshCalculator.getW2(self)
        else:
            MeshCalculatorLowerMemory.getW2(self)
        
        
        
class MeshFromImageStack(MeshCalculatorLowerMemory):
    """
    Simple wrapper class to import mesh from an image stack with manual spacings
    
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
        lowmem : bool
            Wether to use the "MeshCalculatorLowerMemory". The lower memory
            class does not create a dataframe with information on edges.
        **kwargs : dict
            Any optional/keyword arguments that should be passed to
            skimage.measure.marching_cubes.

    Returns
    -------
        None.
    """
    def __init__(self, image, channel, test='auto', lowmem=False, **kwargs):
        """
        Initialize class instance
        """
        self.edgesinmemory = not lowmem
        try:
            grayimg = image[:,:,:, channel]
        except IndexError:
            grayimg = image[:,:,:]
        mesh, area, threshold = createoptimalMesh(grayimg, test, **kwargs)
        print(f'Best mesh at threshold {threshold}')
        v, f, normals, values = mesh
        nv, _, _, nf = igl.remove_duplicate_vertices(v, f, 1e-7)
        if self.edgesinmemory:
            MeshCalculator.__init__(self, nv, nf)
        else:
            MeshCalculatorLowerMemory.__init__(self, nv, nf)
            
        
    def getW2(self):
        if self.edgesinmemory:
            MeshCalculator.getW2(self)
        else:
            MeshCalculatorLowerMemory.getW2(self)
