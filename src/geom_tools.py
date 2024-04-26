
#    This file is part of PEAXIS.
#
#    PEAXIS is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# 
# Copyright (C) Maciej Bartkowiak, 2019-2021

__doc__ = """
Simple calculations in 3D space,
working on single points or on whole arrays.
Based on numpy functions, should work at a sensible pace.
Maciej Bartkowiak, 28 Mar 2014
"""

import numpy as np
import scipy.optimize as scopt
# from itertools import permutations

zdir = np.array([0.0, 0.0, 1.0])
ydir = np.array([0.0, 1.0, 0.0])

def distance(point1, point2):
    """
    Calculates Carthesian distance between two points
    expressed as same-sized numpy arrays.
    """
    return np.sqrt(((point1-point2)*(point1-point2)).sum(1))

def length(vector):
    """
    Calculates the length of a vector (or n vectors) expressed
    in Carthesian coordinates as a numpy array.
    Returns an array of shape (n,1)
    """
    if len(vector.shape) > 1:
        return np.sqrt((vector*vector).sum(1))
    else:
        return np.sqrt((vector*vector).sum())

def normalise(vector):
    """
    Divides a vector, expressed in Carthesian coordinates,
    by its length, returning an unary vector expressing
    a direction.
    """
    return (vector.T / length(vector).T).T

def angle_v1v2(vec1 = np.array([1, 0, 0]), vec2 = np.array([0, 0, 1])):
    n1 = normalise(vec1)
    n2 = normalise(vec2)
    # third = np.cross(n1, n2)
    other = np.dot(n1, n2)
    ang1 = np.arccos(other)
    return np.degrees(ang1)
    
def angle_sign(cross = np.array([1, 0, 0]), cast_axis = np.array([0, 0, 1])):
    if length(cross) == 0:
        return 0
    other = np.dot(cross, cast_axis)
    if other > 0:
        return 1
    else:
        return -1

def dist_coords(indices, base, point):
    result = indices[0]*base[0] + indices[1]*base[1] + indices[2]*base[2]
    return np.array((point - result))[0]
    
def find_coords(indices, base, point):
    newind = scopt.leastsq(dist_coords, indices, args = (base, point))[0]
    return newind

def rotY(degrees):
    angle = np.radians(degrees)
    rotmat = np.array([[ np.cos(angle), 0.0 , -np.sin(angle)],
                       [ 0.0, 1.0 , 0.0],
                       [ np.sin(angle), 0.0, np.cos(angle)]])
    return rotmat

def rotX(degrees):
    angle = np.radians(degrees)
    rotmat = np.array([[ 1.0, 0.0, 0.0],
                       [ 0.0, np.cos(angle),  np.sin(angle) ],
                       [ 0.0, -np.sin(angle), np.cos(angle)]])
    return rotmat

def rotZ(degrees):
    angle = np.radians(degrees)
    rotmat = np.array([[ np.cos(angle),  np.sin(angle), 0.0],
                       [ -np.sin(angle), np.cos(angle), 0.0],
                       [ 0.0, 0.0, 1.0]])
    return rotmat
    
def rotate(rotmat, arr):
    """
    A rotation matrix 'rotmat' of the form of a 3x3 array
    is used to mupliply a (x, 3) array 'arr' of coordinates.
    """
    return rotmat.dot(arr.T).T

def sph_to_cart(coords):
    """
    Converts a [:,3] array of spherical coordinates
    into a same-sized array of Cartesian coordinates.
    """
    xyz = coords.copy()
    r, theta, phi = coords[:, 0], np.radians(coords[:, 1]), np.radians(coords[:, 2])
    xyz[:,0] = r * np.sin(theta) * np.cos(phi)
    xyz[:,1] = r * np.sin(theta) * np.sin(phi)
    xyz[:,2] = r * np.cos(theta)
    return xyz
    
def cart_to_sph(coords):
    """
    Converts a [:,3] array of Cartesian coordinates
    into a same-sized array of spherical coordinates.
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    rtp = coords.copy()
    #rtp[:,0] = np.sqrt(x*x + y*y + z*z)
    #rtp[:,1] = np.arccos(z / rtp[:,0])
    #rtp[:,2] = np.arctan(y / x)
    #rtp[:,1:3] = np.nan_to_num(rtp[:,1:3])
    xy = x**2 + y**2
    rtp[:,0] = np.sqrt(xy + z**2)
    rtp[:,1] = np.arctan2(np.sqrt(xy), z)
    rtp[:,2] = np.arctan2(y, x)
    rtp[:,1:] = np.degrees(rtp[:,1:])
    rtp[:,1:3] = np.nan_to_num(rtp[:,1:3])
    return rtp

def arb_rotation(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degrees.
    """
    axis = np.asarray(axis)
    theta = np.asarray(np.radians(theta))
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
                     
def arb_rotation_rad(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degrees.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def line_point_distance(line_origin, line_vector, point):
    l = line_origin.copy()
    v = line_vector.copy()
    p = point.copy()
    v = v / length(v)
    temp1 = l - p
    temp2 = np.dot(temp1, v) * v
    convec = temp1 - temp2
    return convec

def line_plane_intersection(line_origin, line_vector, plane_origin, plane_normal):
    l = line_origin.copy()
    v = line_vector.copy()
    p0 = plane_origin.copy()
    n = plane_normal.copy()
    if abs(np.dot(v,n) ) > 1e-9:
        d= np.dot((p0-l), n) / np.dot(v,n)
        result = l + d*v
    else:
        result = -1.0
    return result

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
