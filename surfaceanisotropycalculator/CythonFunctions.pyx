# -*- coding: utf-8 -*-
"""
    Copyright (C) 2025  Geert Schulpen
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
cimport numpy as np
from libc.math cimport sqrt


cpdef int getinternalangles(long long int[:,:] f, double[:,:] v, double[:,:] result):
    cdef long long int i, lenf, v0, v1, v2
    cdef double[:] p0, p1, p2
    cdef double dxa, dya, dza, dxb, dyb, dzb, dxc, dyc, dzc, a1, a2, a3, temp
    lenf = f.shape[0]
    for i in range(lenf):
        v0 = f[i,0]
        v1 = f[i,1]
        v2 = f[i,2]        
        p0 = v[v0,0:3]
        p1 = v[v1,0:3]
        p2 = v[v2,0:3]
        dxa = p1[0] - p0[0]
        dya = p1[1] - p0[1]
        dza = p1[2] - p0[2]
        dxb = p2[0] - p1[0]
        dyb = p2[1] - p1[1]
        dzb = p2[2] - p1[2]
        dxc = p0[0] - p2[0]
        dyc = p0[1] - p2[1]
        dzc = p0[2] - p2[2]
        temp = -dxc*dxa - dyc*dya - dzc*dza
        temp /= sqrt(dxc*dxc + dyc*dyc + dzc*dzc)
        temp /= sqrt(dxa*dxa + dya*dya + dza*dza)
        result[i,0] = temp
        temp = -dxa*dxb - dya*dyb - dza*dzb
        temp /= sqrt(dxb*dxb + dyb*dyb + dzb*dzb)
        temp /= sqrt(dxa*dxa + dya*dya + dza*dza)
        result[i,1] = temp
        temp = -dxb*dxc - dyb*dyc - dzb*dzc
        temp /= sqrt(dxb*dxb + dyb*dyb + dzb*dzb)
        temp /= sqrt(dxc*dxc + dyc*dyc + dzc*dzc)
        result[i,2] = temp
    return 1
        
        
cpdef int getnormals(long long int[:,:] f, double[:,:] v, double[:,:] result):
    cdef long long int i, lenf, v0, v1, v2
    cdef double[:] p0, p1, p2, pn
    cdef double xa, ya, za, xb, yb, zb
    p0 = np.zeros(3, dtype=np.double)
    p1 = np.zeros(3, dtype=np.double)
    p2 = np.zeros(3, dtype=np.double)
    pn = np.zeros(3, dtype=np.double)
    lenf = f.shape[0]
    for i in range(lenf):
        v0 = f[i,0]
        v1 = f[i,1]
        v2 = f[i,2]
        p0 = v[v0,0:3]
        p1 = v[v1,0:3]
        p2 = v[v2,0:3]
        xa = p2[0] - p0[0]
        ya = p2[1] - p0[1]
        za = p2[2] - p0[2]
        xb = p1[0] - p0[0]
        yb = p1[1] - p0[1]
        zb = p1[2] - p0[2]
        result[i,0] = ya * zb - za * yb
        result[i,1] = za * xb - xa * zb
        result[i,2] = xa * yb - ya * xb
    return 1
