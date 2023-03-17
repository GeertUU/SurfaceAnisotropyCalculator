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


"""
Helper function to enable renumbering faces
"""
def binarySearchCount(arr, key):
    """
    Perform a binary search on a sorted list. Return both wether key is in arr
    and where it should be inserted to keep the list sorted

    Parameters
    ----------
    arr : list
        Sorted list to be searched.
    key : scalar with same TYPE as list
        item to be found.

    Returns
    -------
    bool
        Whether key is in arr.
    integer
        Index where key should be inserted in arr to keep a sorted array.

    """
    n = len(arr)
    left = 0
    right = n - 1

    count = 0

    while (left <= right):
        mid = int((right + left) / 2)

        # Check if we found key
        if(arr[mid] == key):
            return True, mid
        # Check if middle element is less than key
        elif (arr[mid] < key):
            # At least (mid + 1) elements exist whose values are less than
            # or equal to key
            count = mid + 1
            left = mid + 1
        # If key is smaller, ignore right half
        else:
            right = mid - 1

    return False, count



class MeshCalculator():
    """
    Class to calculate stuff based on a mesh.
    """

    def __init__(self, v, f):
        """
        Initialize class instance.

        Parameters
        ----------
        v : numpy array (N x 3)
            List of verteces of mesh.
        f : numpy array of integers (M x 3)
            Array of groups of 3 indeces which together form a vertex.

        Returns
        -------
        None.

        """
        self.v = v
        self.f = f

        # Collect the maximum and minimum values along the different axes
        self.maxs = np.max(self.v, axis=0)
        self.mins = np.min(self.v, axis=0)

        # to keep track of what has been calculated we initialize an empty dict
        self.resetcalc()

    def resetcalc(self):
        """
        Reset dictionary of calculated properties

        Returns
        -------
        None.

        """
        self._properties = {}

    def _getverteces(self):
        """
        Make a Mx3x3 array with the coordinates of verteces of faces.

        Returns
        -------
        None.

        """
        self.verteces = self.v[self.f]
        #We make a new key in the _properties dictionary to indicate we
        #calculated this property
        self._properties["verteces"] = True

    def _getcenters(self):
        """
        Make a Mx3 array of the center (average of the verteces) of every face

        Returns
        -------
        None.

        """
        #dict.get(key) returns False if key does not exist in dict.
        if not self._properties.get("verteces"):
            self.getverteces
        self.centers = np.sum(self.verteces, axis=1)/3
        self._properties["centers"] = True

    def _getnormals(self):
        """
        Make a Mx3 array of normal vectors for every face. The vectors are not 
        normalized.

        Returns
        -------
        None.

        """
        if not self._properties.get("verteces"):
            self._getverteces()
        self.normals = np.cross(self.verteces[:, 1]-self.verteces[:, 0],
                                self.verteces[:, 2]-self.verteces[:, 1])
        self._properties["normals"] = True

    def _getareas(self):
        """
        Make a list containing the area for every face

        Returns
        -------
        None.

        """
        if not self._properties.get("normals"):
            self._getnormals()
        self.areas = 0.5*np.linalg.norm(self.normals, axis=1)
        self._properties["areas"] = True

    def getW021(self):
        """
        Calculate the Minkowski tensor W^{0,2}_1

        Returns
        -------
        None.

        """
        # if/once the areas exist the normals also exist
        if not self._properties.get("areas"):
            self._getareas()
        self._localW021 = np.array([1/(4*a) * np.tensordot(n, n, axes=0)
                                    for a, n in zip(self.areas, self.normals)])
        self.W021 = np.sum(self._localW021, axis=0)/3
        self.W021eigenvals, self.W021eigenvecs = np.linalg.eigh(self.W021)
        self._properties["W021"] = True

    def getbeta021(self):
        """
        Calculate the ratio between the highest and lowest eigenvalue of the 
        Minkowski tensor W^{0,2}_1

        Returns
        -------
        None.

        """
        if not self._properties.get("W021"):
            self.getW021()
        self.beta021 = self.W021eigenvals[0] / self.W021eigenvals[-1]
        self._properties["beta021"] = True

    def cropmesh(self, maxx=None, minx=None, maxy=None, miny=None, maxz=None, minz=None):
        """
        Crop all verteces that are outside given region. Also removes faces.
        
        Parameters
        ----------
        maxx : float, optional
            Maximum x value to be accepted. The default is max(self.v[:,0]).
        minx : float, optional
            Minimum x value to be accepted. The default is min(self.v[:,0]).
        maxy : float, optional
            Maximum y value to be accepted. The default is max(self.v[:,1]).
        miny : float, optional
            Minimum y value to be accepted. The default is min(self.v[:,1]).
        maxz : float, optional
            Maximum z value to be accepted. The default is max(self.v[:,2]).
        minz : float, optional
            Minimum z value to be accepted. The default is min(self.v[:,2]).

        Returns
        -------
        None.

        """
        if not maxx:
            maxx = self.maxs[0]
        if not maxy:
            maxy = self.maxs[1]
        if not maxz:
            maxz = self.maxs[2]
        if not minx:
            minx = self.mins[0]
        if not miny:
            miny = self.mins[1]
        if not minz:
            minz = self.mins[2]

        minima = (minx, miny, minz)
        maxima = (maxx, maxy, maxz)
        
        myf = []
        subtract = np.zeros(3)
        remove = [True, True, True]
        
        #make a list of all "external" verteces
        unaccept = np.array([i for i, p in enumerate(self.v) if
                                         any(p < minima) or any(p > maxima)])

        for face in self.f:
        #check for each vertex if it is in unaccept and how many lower indices
        #are in unaccept. If no vertex is in unaccept calculate new triplet.
            for i, point in enumerate(face):
                remove[i], subtract[i] = binarySearchCount(unaccept, point)
            if not any(remove):
                myf.append(face - subtract)
        myf = np.array(myf, dtype=int)
        
        """
        #old implementation that does not delete verteces
        newf = np.array([face for face in self.f if not any(np.isin(face, unaccept))])
        """
        #delete all "external" verteces
        self.v = np.delete(self.v, unaccept, axis=0)
        self.f = myf
        
        #After changing the mesh one should reset all the performed calculations
        self.resetcalc()


class MeshFromFile(MeshCalculator):
    """
    Simple wrapper class to import mesh from a file
    """
    def __init__(self, filename):
        """
        Initialize class instance

        Parameters
        ----------
        filename : str
            Path + filename + extension of the mesh file. Supported extensions
            .obj, .off, .stl, .wrl, .ply, .mesh.

        Returns
        -------
        None.

        """
        v, f = igl.read_triangle_mesh(filename)
        nv, _, _, nf = igl.remove_duplicate_vertices(v, f, 1e-7)
        MeshCalculator.__init__(self, nv, nf)

