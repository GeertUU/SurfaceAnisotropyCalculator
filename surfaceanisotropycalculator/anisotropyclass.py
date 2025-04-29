# -*- coding: utf-8 -*-
"""
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
"""

import numpy as np
import pandas as pd

import igl

import CythonFunctions


"""
Helper function to iterate over ntuples in an array (wrapping)
"""
def ntuples(lst, n):
    return zip(*[np.concatenate((lst[i:],lst[:i])) for i in range(n)])

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


class MeshCalculator_legacy():
    """
    Old class to calculate stuff based on a mesh.
    """

    def __init__(self, v, f):
        """
        Initialize class instance.

        Parameters
        ----------
        v : numpy array (N x 3)
            List of verteces of mesh.
        f : numpy array of integers (M x 3)
            Array of groups of 3 indeces which together form a face.

        Returns
        -------
        None.

        """
        self.verteces = v
        self.faces = f

        # Collect the maximum and minimum values along the different axes
        self.maxs = np.max(self.verteces, axis=0)
        self.mins = np.min(self.verteces, axis=0)

        # To keep track of what has been calculated we initialize an empty dict
        self.resetcalc()
        
        edges = []
        faces = []
        lenedges = 0
        vfaces = [set() for _ in v]
        vneighbors = [set() for _ in v]
        vedges = [[] for _ in v]
        fneighbors = [set() for _ in f]
        fedges = [[] for _ in f]
        for numf, face in enumerate(self.faces):
            thisface = list(face)
            for v1, v2, v3 in ntuples(face, 3):
                #register the vertexcoordinates with the face
                thisface += list(self.verteces[v1])
                #register the face with the vertex
                vfaces[v1].add(numf)
                vneighbors[v1].update([v2,v3])
                #register the edge with both verteces
                vedges[v1].append(lenedges)
                vedges[v2].append(lenedges)
                #register the edge with the face
                fedges[numf].append(lenedges)
                #define the edge
                edges.append((v1, v2, numf, np.array(self.verteces[v1]),
                                np.array(self.verteces[v2])))
                lenedges += 1
            faces.append(thisface)

        self.v = pd.DataFrame(v, columns=["x","y","z"])
        self.v["faces"] = vfaces
        self.v["neighbors"] = vneighbors
        self.v["edges"] = vedges

        for numf, face in enumerate(self.faces):
            #find neighboring faces
            for v1, v2 in ntuples(face, 2):
                for neighbor in self.v["faces"][v1]:
                    if numf != neighbor and v2 in self.faces[neighbor]:
                        fneighbors[numf].add(neighbor)
                        
        self.f = pd.DataFrame(faces, columns=["v1","v2","v3", "x1", 'y1', 'z1', 
                                              "x2", 'y2', 'z2', "x3", 'y3', 'z3'])
        self.f["neighbors"] = fneighbors
        self.f["edges"] = fedges
        self.e = pd.DataFrame(edges, columns=['v1', 'v2', 'f', 'loc1', 'loc2'])
         

    def resetcalc(self):
        """
        Reset dictionary of calculated properties

        Returns
        -------
        None.

        """
        self._properties = {}

    def _getcenters(self):
        """
        Make a Mx3 array of the center (average of the verteces) of every face

        Returns
        -------
        None.

        """
        
        self.f['xc'] = (self.f.x1 + self.f.x2 + self.f.x3)/3
        self.f['yc'] = (self.f.y1 + self.f.y2 + self.f.y3)/3
        self.f['zc'] = (self.f.z1 + self.f.z2 + self.f.z3)/3
        self._properties["centers"] = True

    def _getnormals(self):
        """
        Make a Mx3 array of normal vectors for every face. The vectors are not 
        normalized. Vectors point away from high intensity if made from image.

        Returns
        -------
        None.

        """
        #take a cross product manually
        self.f['xn'] = -((self.f.y2-self.f.y1)*(self.f.z3-self.f.z1)-
                            (self.f.z2-self.f.z1)*(self.f.y3-self.f.y1))
        self.f['yn'] = -((self.f.z2-self.f.z1)*(self.f.x3-self.f.x1)-
                            (self.f.x2-self.f.x1)*(self.f.z3-self.f.z1))
        self.f['zn'] = -((self.f.x2-self.f.x1)*(self.f.y3-self.f.y1)-
                            (self.f.y2-self.f.y1)*(self.f.x3-self.f.x1))
            
        # self.normals = np.cross(self.verteces[:, 1]-self.verteces[:, 0],
        #                         self.verteces[:, 2]-self.verteces[:, 1])
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
        self.f['area'] = 0.5*np.sqrt(self.f.xn**2 + self.f.yn**2 + self.f.zn**2)
        self._properties["areas"] = True
        
    def _getinternalangles(self):
        """
        Calculate the internal angels for all verteces for every face

        Returns
        -------
        None.

        """
        #The angle between 2 vectors is arccos[(a.b)/((|a||b|)]
        self.f['a1'] = np.arccos(((self.f.x3-self.f.x1)*(self.f.x2-self.f.x1) +
                               (self.f.y3-self.f.y1)*(self.f.y2-self.f.y1) +
                               (self.f.z3-self.f.z1)*(self.f.z2-self.f.z1)) / 
                              (np.sqrt(((self.f.x3-self.f.x1)**2 + (self.f.y3-self.f.y1)**2
                                + (self.f.z3-self.f.z1)**2) * 
                               ((self.f.x2-self.f.x1)**2 + (self.f.y2-self.f.y1)**2
                                + (self.f.z2-self.f.z1)**2)))
                             )
        self.f['a2'] = np.arccos(((self.f.x1-self.f.x2)*(self.f.x3-self.f.x2) +
                               (self.f.y1-self.f.y2)*(self.f.y3-self.f.y2) +
                               (self.f.z1-self.f.z2)*(self.f.z3-self.f.z2)) / 
                              (np.sqrt(((self.f.x1-self.f.x2)**2 + (self.f.y1-self.f.y2)**2
                                + (self.f.z1-self.f.z2)**2) * 
                               ((self.f.x3-self.f.x2)**2 + (self.f.y3-self.f.y2)**2
                                + (self.f.z3-self.f.z2)**2)))
                             )
        self.f['a3'] = np.arccos(((self.f.x1-self.f.x3)*(self.f.x2-self.f.x3) +
                               (self.f.y1-self.f.y3)*(self.f.y2-self.f.y3) +
                               (self.f.z1-self.f.z3)*(self.f.z2-self.f.z3)) / 
                              (np.sqrt(((self.f.x1-self.f.x3)**2 + (self.f.y1-self.f.y3)**2
                                + (self.f.z1-self.f.z3)**2) * 
                               ((self.f.x2-self.f.x3)**2 + (self.f.y2-self.f.y3)**2
                                + (self.f.z2-self.f.z3)**2)))
                             )
        self._properties["internal angles"] = True

    def _getedgevector(self):
        """
        Calculate the edge vectors and edge lengths

        Returns
        -------
        None.

        """
        self.e['edge'] = self.e.loc2 - self.e.loc1
        self.e['norm'] = self.e.apply(lambda x: np.sqrt(np.dot(x.edge,x.edge)), axis=1)
        self._properties["edges"] = True
        
    def _getantifaces(self):
        """
        Find for each edge find the other face that borders that edge

        Returns
        -------
        None.

        """
        self.e['antiface'] = self.e.apply(lambda x: 
            self.e[(self.e['v2'] == x.v1) & (self.e['v1'] == x.v2)].f.iloc[0], axis=1)
        self._properties['antifaces'] = True
    
    def _findedgeverteces(self):
        """
        Fills self.v.edge with bools depending on if the vertex is on the edge

        Returns
        -------
        None.

        """
        self.v['edge'] = self.v.apply(lambda x: self._checkvertexedge(x.faces), axis=1)
        self._properties["edge verteces"] = True


    def _suminternalangles(self, faces, vindex):
        """
        Calculate the sum of internal angles for verteces.

        Parameters
        ----------
        faces : LIST
            List of faces which contain the vertex.
        vindex : INT
            Index of the vertex.

        Returns
        -------
        FLOAT
            If vertex is internal: sum of internal angles

        """
        return sum(self.f.loc[face, 'a1'] if self.f.loc[face,'v1'] == vindex 
                   else (self.f.loc[face, 'a2'] if self.f.loc[face,'v2'] == vindex 
                         else self.f.loc[face, 'a3'])
                   for face in faces)
    
    def _checkvertexedge(self, faces):
        """
        Check if the vertex is on the edge of the surface

        Parameters
        ----------
        faces : SET or LIST
            SET of faces that contain the vertex.

        Returns
        -------
        bool
            Wether the vertex is on the edge.

        """
        for f in faces:
            found = 0
            for n in self.f.loc[f, 'neighbors']:
                if n in faces:
                    if found == 2:
                        break
                    else:
                        found += 1
            if found != 2:
                return True
        return False

    def _getalphae(self, edge):
        """
        Calculate the dihedral angle for an edge

        The dihedral angle is calculated using the method suggested here:
        https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
        In short: we find the normal of one face in the coordinate system 
        composed of the other normal, the edge and the crossproduct of these.
        Then the arctan2 function allows for determination of correct angle.
        
        Parameters
        ----------
        edge : Series
            Pandas series with at least the values: 
                edge:
                    an numpy array of the vector pointing along the edge
                norm:
                    the length of the edge
                f:
                    the index of the face of which this is an edge
                antiface:
                    the index of the face which borders this edge.

        Returns
        -------
        alphae : float
            The dihedral angle.

        """
        normedge = edge.edge/edge.norm
        myface = self.f.loc[edge.f]
        facenormal = [myface.xn, myface.yn, myface.zn]
        antiface = self.f.loc[edge.antiface]
        neighbornormal = [antiface.xn, antiface.yn, antiface.zn]
        #Note that normals are not normalized, but that is not necessary since vector lenghts cancel in final calculation
        #finding the last coordinate direction
        m = np.cross(facenormal, normedge)
        #tan(alphae) = y/x, where x is parallel part of the 2 normals, y normal part
        alphae = np.arctan2(np.dot(neighbornormal, m), np.dot(neighbornormal, facenormal))
        return alphae
        
    
    def getW1(self):
        """
        Calculate the Minkowski scalar W_1, which corresponds to the total area

        Returns
        -------
        None.

        """
        if not self._properties.get("areas"):
            self._getareas()
        self.W1 = self.f.area.sum()
        self._properties["W1"] = True
        
    def getW2(self):
        """
        Calculate the Minkowski scalar W_2, i.e. the mean curvature

        Returns
        -------
        float
            Minkowski scalar W_2 of the current surface.

        """
        if not self._properties.get("normals"):
            self._getnormals()
        if not self._properties.get("edges"):
            self._getedgevector()
        if not self._properties.get("antiface"):
            self._getantifaces()

        self.e['alphae'] = self.e.apply(lambda x: self._getalphae(x), axis=1)
        self.e['w2'] = self.e.norm * self.e.alphae * 0.25
        self.W2 = 1/3*self.e.w2.sum()
        self._properties["w2"] = True
        return self.W2
      
    def getW3(self, Legacy=False):
        """
        Calculate the Minkowski scalar W_3, i.e. the total Guassian curvature

        Returns
        -------
        None.

        """
        
        if not self._properties.get("internal angles"):
            self._getinternalangles()
        
        if not self._properties.get("edge verteces"):
            self._findedgeverteces()

        self.v['summedangles'] = self.v.apply(lambda x: self._suminternalangles(x.faces, x.name), axis=1)
        if Legacy:
            self.v['w3'] = (1-self.v.edge) * (2*np.pi - self.v.summedangles)
        else:
            self.v['w3'] = 2*np.pi - np.pi*self.v.edge - self.v.summedangles
        self.W3 = 1/3*self.v.w3.sum()
        self._properties["W3"] = True
        
    def scaledGenus(self, volume = None):
        """
        Calculates the scaled genus.
        
        Parameters
        ----------
        volume : float, optional
            The volume to be used in the calculation.
            The default value is equal to the box size

        Returns
        -------
        scaledgenus : TYPE
            DESCRIPTION.

        """
        
        if not self._properties.get("W1"):
            self.getW1()
        if not self._properties.get("W3"):
            self.getW3()
        if volume == None:
            volume = np.prod(self.maxs - self.mins)
        
        genus = 1 - (3*self.W3)/(4*np.pi)
        scaledgenus = genus * volume**2/(self.W1**3)
        
        return scaledgenus
        
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
        
        # We calculate only 6 of the 9 elements due to symmetry
        self.f['w021_xx'] = 1/(4*self.f.area) * self.f.xn * self.f.xn
        self.f['w021_xy'] = 1/(4*self.f.area) * self.f.xn * self.f.yn
        self.f['w021_xz'] = 1/(4*self.f.area) * self.f.xn * self.f.zn
        self.f['w021_yy'] = 1/(4*self.f.area) * self.f.yn * self.f.yn
        self.f['w021_yz'] = 1/(4*self.f.area) * self.f.yn * self.f.zn
        self.f['w021_zz'] = 1/(4*self.f.area) * self.f.zn * self.f.zn
        # self._localW021 = np.array([1/(4*a) * np.tensordot(n, n, axes=0)
        #                             for a, n in zip(self.areas, self.normals)])
        W021_xx = self.f.w021_xx.sum()/3
        W021_xy = self.f.w021_xy.sum()/3
        W021_xz = self.f.w021_xz.sum()/3
        W021_yy = self.f.w021_yy.sum()/3
        W021_yz = self.f.w021_yz.sum()/3
        W021_zz = self.f.w021_zz.sum()/3
        
        self.W021 = np.array([[W021_xx, W021_xy, W021_xz], 
                              [W021_xy, W021_yy, W021_yz], 
                              [W021_xz, W021_yz, W021_zz]])
        self.W021eigenvals, self.W021eigenvecs = np.linalg.eigh(self.W021)
        self._properties["W021"] = True
        
    def getaalpha021(self):
        """
        Calculate one minus the ratio between the highest and lowest eigenvalue
        of the Minkowski tensor W^{0,2}_1
        
        Returns
        -------
        None.
        
        """
        if not self._properties.get("W021"):
            self.getW021()
        self.alpha021 = 1 - self.W021eigenvals[0] / self.W021eigenvals[-1]
        self._properties["alpha021"] = True

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

    def savemesh(self, file):
        """
        Save the mesh. Supported extensions .obj, .off, .stl, .wrl, .ply .mesh.

        Parameters
        ----------
        name : str
            Path + filename + extension of the file to be saved. Supported extensions
            .obj, .off, .stl, .wrl, .ply, .mesh.

        Returns
        -------
        None.

        """
        myv = self.v.loc[:,['x', 'y', 'z']]
        myf = self.f.loc[:,['v1', 'v2', 'v3']]
        myv['newindex'] = myv.reset_index().index
        myf['exv1'] = myf.v1.apply(lambda x: myv.loc[x, 'newindex'])
        myf['exv2'] = myf.v2.apply(lambda x: myv.loc[x, 'newindex'])
        myf['exv3'] = myf.v3.apply(lambda x: myv.loc[x, 'newindex'])
        
        self.exportv = myv[['x','y','z']].to_numpy()
        self.exportf = myf[['exv1','exv2','exv3']].to_numpy()
        
        testsave = igl.write_triangle_mesh(file, self.exportv, self.exportf)
        if testsave:
            print("Mesh saved succesfully")
        else:
            print("Saving failed")
            
class MeshCalculator(MeshCalculator_legacy):
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
        self.verteces = v
        self.faces = f

        # Collect the maximum and minimum values along the different axes
        self.maxs = np.max(self.verteces, axis=0)
        self.mins = np.min(self.verteces, axis=0)

        # To keep track of what has been calculated we initialize an empty dict
        self.resetcalc()
        
        fs = np.zeros(f.shape, dtype = np.long)
        normals = np.zeros(f.shape, dtype = np.double)
        vfaces = [set() for _ in v]
        vneighbors = [set() for _ in v]
        pn = np.zeros(3, dtype=np.double)
        discarded = 0
        for numf, face in enumerate(self.faces):
            c1 = np.array(v[face[0]], dtype = np.double)
            c2 = np.array(v[face[1]], dtype = np.double)
            c3 = np.array(v[face[2]], dtype = np.double)
            test = CythonFunctions.getnormal(c1, c2, c3, pn)
            #if the area of the face is 0 we do not want to include it
            if pn[0]==pn[1]==pn[2]==0: 
                discarded += 1
                continue
            for v1, v2, v3 in ntuples(face, 3):
                #register the face with the vertex
                vfaces[v1].add(numf - discarded)
                vneighbors[v1].update([v2,v3])
            fs[numf - discarded] = face
            normals[numf - discarded] = pn

        #remove all empty rows
        fs = fs[:numf - discarded + 1]
        normals = normals[:numf - discarded + 1]
        fneighbors = [set() for _ in fs]
        #create dataframe for vertices
        self.v = pd.DataFrame(v, columns=["x","y","z"])
        self.v["faces"] = vfaces
        self.v["neighbors"] = vneighbors

        for numf, face in enumerate(fs):
            #find neighboring faces
            for v1, v2 in ntuples(face, 2):
                for neighbor in self.v["faces"][v1]:
                    if numf != neighbor and v2 in fs[neighbor]:
                        fneighbors[numf].add(neighbor)
        #create dataframe for faces
        self.f = pd.DataFrame(fs, columns=["v1","v2","v3"])
        self.f["neighbors"] = fneighbors
        temp = pd.DataFrame(normals, columns=["xn","yn","zn"])
        self.f[["xn","yn","zn"]] = temp
        self._properties["normals"] = True
        
    
    
    def _getcenters(self):
        """
        Make a Mx3 array of the center (average of the verteces) of every face

        Returns
        -------
        None.

        """
        self.f['xc'] = self.f.apply(lambda l: self.v.loc[[l.v1, l.v2,l.v3],"x"].sum()/3, axis=1)
        self.f['yc'] = self.f.apply(lambda l: self.v.loc[[l.v1, l.v2,l.v3],"y"].sum()/3, axis=1)
        self.f['zc'] = self.f.apply(lambda l: self.v.loc[[l.v1, l.v2,l.v3],"z"].sum()/3, axis=1)
        self._properties["centers"] = True
        

    def _getnormals(self):
        """
        Make a Mx3 array of normal vectors for every face. The vectors are not 
        normalized. Vectors point away from high intensity if made from image.

        Returns
        -------
        None.

        """
        fs = self.f.loc[:,["v1","v2","v3"]].to_numpy(dtype=np.long)
        norms = np.zeros(fs.shape, dtype=np.double)
        vs = self.v.loc[:,["x","y","z"]].to_numpy(dtype=np.double)
        test = CythonFunctions.getnormals(fs, vs, norms)
        norms = pd.DataFrame(norms, columns=["xn","yn","zn"])
        self.f = self.f.join(norms)
        self._properties["normals"] = True
        
    
    def _getinternalangles(self):
        """
        Calculate the internal angels for all verteces for every face

        Returns
        -------
        None.

        """
        fs = self.f.loc[:,["v1","v2","v3"]].to_numpy(dtype=np.long)
        angles = np.zeros(fs.shape, dtype=np.double)
        vs = self.v.loc[:,["x","y","z"]].to_numpy(dtype=np.double)
        test = CythonFunctions.getinternalangles(fs, vs, angles)
        angles = pd.DataFrame(np.arccos(angles), columns=["a1","a2","a3"])
        self.f = self.f.join(angles)
        self._properties["internal angles"] = True
    
    def _getw2edge(self, v1, v2, f1, f2):
        """
        Calculate (1/4 * of edge length * dihedral angle) for an edge

        The dihedral angle is calculated using the method suggested here:
        https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
        In short: we find the normal of one face in the coordinate system 
        composed of the other normal, the edge and the crossproduct of these.
        Then the arctan2 function allows for determination of correct angle.
        
        Parameters
        ----------
        v1 : int
            Index of vertex at end of edge
        v2 : int
            Index of vertex at beginning of edge
        f1 : int
            Index of the face on the right hand side of the edge
        f2 : int
            Index of the face on the left hand side of the edge

        Returns
        -------
        alphae : float
            The dihedral angle.

        """
        dx = self.v.at[v1, 'x'] - self.v.at[v2, 'x']
        dy = self.v.at[v1, 'y'] - self.v.at[v2, 'y']
        dz = self.v.at[v1, 'z'] - self.v.at[v2, 'z']
        edge = np.array([dx,dy,dz])
        edgelen = np.linalg.norm(edge)
        normedge = edge/edgelen
        facenormal = [self.f.at[f1, 'xn'], self.f.at[f1, 'yn'], self.f.at[f1, 'zn']]
        neighbornormal = [self.f.at[f2, 'xn'], self.f.at[f2, 'yn'], self.f.at[f2, 'zn']]
        #finding the last coordinate direction
        m = np.cross(facenormal, normedge)
        #tan(alphae) = y/x, where x is parallel part of the 2 normals, y normal part
        alphae = np.arctan2(np.dot(neighbornormal, m), np.dot(neighbornormal, facenormal))
        return (edgelen * alphae * 0.25)
    
    
    def _getlocalw2(self, face):
        w2face = 0
        edges1 = {(face.v1,face.v2), (face.v2,face.v3), (face.v3,face.v1)}
        for f2 in face.neighbors:
            negedges2 = ntuples([self.f.at[f2,'v3'],self.f.at[f2,'v2'], self.f.at[f2,'v1']],2)
            for edge in negedges2:
                if edge in edges1:
                    w2face += self._getw2edge(edge[1], edge[0], face.name, f2)
        return w2face
            
    
    def getW2(self):
        """
        Calculate the Minkowski scalar W_2, i.e. the mean curvature

        Returns
        -------
        float
            Minkowski scalar W_2 of the current surface.

        """
        if not self._properties.get("normals"):
            self._getnormals()
        # if not self._properties.get("edges"):
        #     self._getedgevector()
        # if not self._properties.get("antiface"):
        #     self._getantifaces()
        
        self.f['w2'] = self.f.apply(self._getlocalw2, axis=1)

        self.W2 = 1/3*self.f.w2.sum()
        self._properties["w2"] = True
        return self.W2
    
    
    
