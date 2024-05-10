#    This file is part of ADLER.
#
#    ADLER is free software: you can redistribute it and/or modify
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
# Copyright (C) Maciej Bartkowiak, 2019-2023

import math
import numpy as np
import scipy.interpolate as scint
import scipy.optimize as scopt

# from geom_tools import *
# from units import p_lambda,  p_energy,  p_lambda_meters, p_k
# from sample import Sample
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QMutex, QTimer
from numba import jit

h = 6.626068 * 10 ** (-34)  # it is Planck's constant given in J*s
meV = 1.602176487 * 10 ** (-22)  # that is 1 meV expressed in Joule
mn = 1.67492729 * 10 ** (-27)  # in kilograms
elc = 1.602176487 * 10 ** (-19)  # e, the elementary charge in Coulombs
c_light = 299792458  # light velocity in vacuum
hbar = h / (2 * np.pi)
H_P = 6.626068 * 10 ** (-34)  # it is Planck's constant given in J*s
E_C = 1.602176487 * 10 ** (-19)  # e, the elementary charge in Coulombs
C_LIGHT = 299792458  # light velocity in vacuum
GA_MIN = 15.0
GA_MAX = 65.0
R1_MIN = 650.0
R1_MAX = 1230.0
R2_MIN = 2011.4
R2_MAX = 4079.9
L_MAX = 4749.9


@jit(nopython=True)
def fast_rotate(rotmat, arr):
    """
    A rotation matrix 'rotmat' of the form of a 3x3 array
    is used to mupliply a (x, 3) array 'arr' of coordinates.
    """
    return rotmat.dot(arr.T).T


@jit(nopython=True)
def fast_rotation(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degrees.
    """
    axis = np.asarray(axis)
    theta = np.asarray(np.radians(theta))
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


@jit(nopython=True)
def fast_Rmatrix(tilt, manip_angles, ext_xaxis, ext_yaxis, ext_zaxis):
    ang_offsets = tilt
    angles = manip_angles
    # first the chamber tilt due to arm motion
    Cham_rot = fast_rotation(ext_xaxis[0], ang_offsets)
    a = fast_rotate(Cham_rot, ext_xaxis)
    b = fast_rotate(Cham_rot, ext_yaxis)
    c = fast_rotate(Cham_rot, ext_zaxis)
    # then we rotate everything according to zdir
    Z_rot = fast_rotation(c[0], angles[2])
    a = fast_rotate(Z_rot, a)
    b = fast_rotate(Z_rot, b)
    c = fast_rotate(Z_rot, c)
    # then the tilt around the x-direction due to inclination
    X_rot = fast_rotation(a[0], angles[0])
    a = fast_rotate(X_rot, a)
    b = fast_rotate(X_rot, b)
    c = fast_rotate(X_rot, c)
    # then we rotate everything according to rot
    Y_rot = fast_rotation(b[0], angles[1])
    a = fast_rotate(Y_rot, a)
    b = fast_rotate(Y_rot, b)
    c = fast_rotate(Y_rot, c)
    u_rspace = a[0].copy()
    v_rspace = b[0].copy()
    rotmat = np.dot(Z_rot, Cham_rot)
    rotmat = np.dot(X_rot, rotmat)
    rotmat = np.dot(Y_rot, rotmat)
    # rotmat = np.dot(rotmat,  X_rot)
    Rmatrix = rotmat.copy()
    return u_rspace, v_rspace, Rmatrix


def printMatrix(matrix, label):
    print(label)
    print("  ".join([str(x) for x in matrix[0]]))
    print("  ".join([str(x) for x in matrix[1]]))
    print("  ".join([str(x) for x in matrix[2]]))


def sq(var):
    return var**2


def sample_chamber_rotation(alpha=0, omega=27.05):
    """
    Alpha turned out to be the relative rotation of
    the bottom half of the sample chamber.
    Omega should be a constant, as it is the maximum tilt of
    the vertical axis away from the ideal position.
    """
    # a problem here: Klaus defined omega as 27.0, Tommy as 30.0
    k = math.pi / 180.0
    ux = 0.0
    uy = -1.0 * math.cos((90 - omega) * k)
    uz = math.sin((90 - omega) * k)
    v = np.array([0.0, 1.0, 0.0])
    R11 = ux * ux * (1 - math.cos(alpha * k)) + math.cos(alpha * k)
    R12 = ux * uy * (1 - math.cos(alpha * k)) - uz * math.sin(alpha * k)
    R13 = ux * uz * (1 - math.cos(alpha * k)) + uy * math.sin(alpha * k)
    R21 = ux * uy * (1 - math.cos(alpha * k)) + uz * math.sin(alpha * k)
    R22 = uy * uy * (1 - math.cos(alpha * k)) + math.cos(alpha * k)
    R23 = uy * uz * (1 - math.cos(alpha * k)) - ux * math.sin(alpha * k)
    R31 = ux * uz * (1 - math.cos(alpha * k)) - uy * math.sin(alpha * k)
    R32 = uy * uz * (1 - math.cos(alpha * k)) + ux * math.sin(alpha * k)
    R33 = math.cos(alpha * k) + uz * uz * (1 - math.cos(alpha * k))
    M = np.array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])
    d = normalise(np.dot(M, v))
    # here comes the mystery line
    # it means that we are solvinge the equation: np.dot(R[2], d) = 0
    a, b = d[1], d[2]
    interm = math.sqrt(b**2 / (a**2 + b**2))
    sol1, sol2 = math.asin(interm), math.asin(-interm)
    # print(sol1,sol2, 'in math.radians, ', sol1/k, sol2/k, "in degrees")
    if sol1 > sol2:
        beta = sol1
    else:
        beta = sol2
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(beta), -math.sin(beta)],
            [0.0, math.sin(beta), math.cos(beta)],
        ]
    )
    # print('alpha', alpha)
    # print('d', d)
    # print('beta', beta)
    vv = normalise(np.dot(R, d))
    # print('vv', vv)
    theta = (
        math.atan2(vv[0], vv[1]) / k
    )  # so that we convert to degrees at the same time
    beta = beta / k
    return theta, beta
    # the returned values are:
    # theta: scattering angle
    # beta: rotation of the sample around the beam, which will be corrected for
    # depending on the zdir angle


def CalcPar(
    E_Range=True, E=530.0, bLimits=True, bAlpha=False, alpha_in=None, gamma_in=None
):
    alpha = 0.0
    beta = 0.0
    gamma = (
        0.0  #  [deg]  grating inclination, diffraction angle and detector inclination
    )
    # alpha0,  alpha2,     gamma0,   #  [deg]  corresponding standard values
    # S1, S2,                        #  [mm]   source and detector resolution
    # R,                             #  [mm]   radius of the grating
    # E0,                            #  [eV]   standard energy
    r1 = 0.0
    r2 = 0.0  #  [mm]   distances between source and grating and grating and detector (for standard and actual setting)
    dr2 = (
        0.0  #  [mm]   energy dependent deviation of focal point from theoretical value
    )
    # L,                             #  [mm]   total length for the actual setting
    del_alpha = 0.001  #         change in alpha to find proper value
    # a0, a1, a2, a3,                #         parameters defining the density distribution of the grating
    # Emin,      Emax;               #  [eV]   min. and max. photon energy for the grating
    l = 0
    dl = 1
    l_opt = 0  #         indices for alpha-variation
    k = 1  #         order of diffraction
    rc = True  #         return code

    # set fixed instrument parameters and limits
    S1 = 0.0030  #  [mm]  source height
    S2 = 0.0135  #  [mm]  detector pixel size
    gamma0 = 20.0
    # ParLookup.ParVal("R1_MIN")    =  650.0;
    # ParLookup.ParVal("R1_MAX")    = 1250.0;
    # gamma_min =   17.0;
    # gamma_max =   43.6;

    if E_Range:
        # real grating after ruling
        alpha0 = 88.133
        Emin = 400.0
        E0 = 691.0
        Emax = 1200.0  # grating 2          grating 2 wrong       grating 1 ?
        R = 41345.0  #                                          41306.0;
        a0 = 2399.4  # lines per mm
        a1 = 1.3081e-04 * 2 * a0  #   1.3056e-04       1.3070e-04 * 2 * a0;
        a2 = -8.8900e-08 * 3 * a0  #  -8.8200e-08      -8.9200e-08 * 3 * a0;
        a3 = 1.7900e-10 * 4 * a0  #   1.7900e-10       1.7600e-10 * 4 * a0;
    else:
        # real grating after ruling              ordered
        alpha0 = 87.285
        Emin = 200.0
        E0 = 386.5
        Emax = 600.0  # measured             calculated
        R = 27071.0
        a0 = 2399.972
        a1 = 1.0940e-04 * 2 * a0  #   1.0942e-04 * 2 * a0;
        a2 = -9.3570e-08 * 3 * a0  #  -9.3700e-08 * 3 * a0;
        a3 = 1.7400e-10 * 4 * a0  #   1.7400e-10 * 4 * a0;

    # default values for initial values of alpha and gamma
    if alpha_in is None:
        alpha_in = alpha0
    if gamma_in is None:
        gamma_in = gamma0
    alpha2 = alpha_in

    # estimate correction in r2 value
    dr2 = 0.0  # -0.75*(E-E0)/E - 2.5*(E-E0)/E0;

    # check if energy matches the given range
    if (E < Emin) or (E > Emax):
        rc = False

    if rc:
        alpha = alpha_in
        beta = GratingEq(k, E, alpha, a0)
        r1 = Calc_r1n(E, alpha, beta, R, a0, a1, a2)
        r2 = Calc_r2(alpha, beta, R, r1, a0, a1)
        L = r1 + r2
        gamma = CalcGamma(beta, R, r2, a0, a1)

    # step 2: increase/decrease alpha until gamma is as small as possible or has the wanted value
    if rc and not bAlpha:
        if gamma < gamma_in:
            dl = -1
        else:
            dl = 1
        l = 0
        l_opt = 0
    while dl * gamma > dl * gamma_in:
        l_opt = l  # l_opt=maxi(0,l);
        l += dl
        alpha = alpha_in + l * del_alpha
        beta = GratingEq(k, E, alpha, a0)
        r1 = Calc_r1n(E, alpha, beta, R, a0, a1, a2)
        r2 = Calc_r2(alpha, beta, R, r1, a0, a1)
        L = r1 + r2
        gamma = CalcGamma(beta, R, r2, a0, a1)
        if abs(l) > 10000:
            rc = False
            break

    alpha2 = alpha_in + l_opt * del_alpha
    alpha = alpha2
    beta = GratingEq(k, E, alpha, a0)
    r1 = Calc_r1n(E, alpha, beta, R, a0, a1, a2)
    r2 = Calc_r2(alpha, beta, R, r1, a0, a1)
    L = r1 + r2
    gamma = CalcGamma(beta, R, r2, a0, a1)

    # step 3: search alpha that complies with the limits for r1 and gamma
    if rc and bLimits:
        l = 0
        while (
            (gamma < GA_MIN)
            or (gamma > GA_MAX)
            or (r1 < R1_MIN)
            or (r1 > R1_MAX)
            or (r2 < R2_MIN)
            or (r2 > R2_MAX)
            or ((r1 + r2) > L_MAX)
        ):
            alpha = min(89.6, alpha2 + pow(-1.0, l) * l * del_alpha)
            beta = GratingEq(k, E, alpha, a0)
            r1 = Calc_r1n(E, alpha, beta, R, a0, a1, a2)
            r2 = Calc_r2(alpha, beta, R, r1, a0, a1)
            L = r1 + r2
            gamma = CalcGamma(beta, R, r2, a0, a1)
            l += 1
            if l > 10000:
                rc = False
                break

        if rc:
            return alpha, beta, gamma, r1, r2 + dr2, rc
        else:  # initial values set if search for proper values was not successful
            alpha = alpha_in
            beta = GratingEq(k, E, alpha_in, a0)
            pR1 = Calc_r1n(E, alpha, beta, R, a0, a1, a2)
            pR2 = Calc_r2(alpha, beta, R, pR1, a0, a1)
            gamma = CalcGamma(beta, R, pR2, a0, a1)
            return alpha, beta, gamma, R1, R2, rc


def GratingEq(k, E, alphaD, a0):
    beta = math.asin(math.sin(math.radians(alphaD)) - k * 1e-7 * p_lambda(E) * a0)
    return math.degrees(beta)


def GratingEnergy(k, beta, alphaD, a0):
    E = PhotonEnergy(
        (math.sin(math.radians(alphaD)) - math.sin(math.radians(beta))) / (k * a0)
    )
    return E


def Calc_r1n(E, alphaD, betaD, R, a0, a1, a2):
    r1 = 0.0
    Ke = 1000.0 * H_P / E_C * C_LIGHT
    alpha = math.radians(alphaD)
    beta = math.radians(betaD)
    B = a2 * E * R * sq(math.cos(alpha)) / (3 * a0) + a1 * (
        a1 * Ke * R + E * (math.cos(alpha) + math.cos(beta))
    ) * math.sin(alpha) / (2 * a0)
    W = sq(E * R * math.cos(alpha) * math.cos(beta)) * (
        8
        * a0
        * Ke
        * R
        * (
            B * math.sin(beta)
            - a2 * E * R * sq(math.cos(beta)) * math.sin(alpha) / (3 * a0)
        )
        + sq(E * math.sin(alpha + beta))
    )
    N = (
        2
        * sq(E * R)
        * (math.sin(alpha) * math.cos(beta) - sq(math.cos(alpha)) * math.tan(beta))
    )

    y1 = (
        2 * E * R * (E + a1 * Ke * R / math.cos(beta)) * math.sin(alpha)
        + (
            sq(E) * R * math.cos(alpha) * math.sin(alpha - beta)
            - math.sqrt(W) / math.cos(beta)
        )
        / sq(math.cos(beta))
    ) / N
    y2 = (
        2 * E * R * (E + a1 * Ke * R / math.cos(beta)) * math.sin(alpha)
        + (
            sq(E) * R * math.cos(alpha) * math.sin(alpha - beta)
            + math.sqrt(W) / math.cos(beta)
        )
        / sq(math.cos(beta))
    ) / N

    x1 = (
        a1 * Ke * R
        + E * (math.cos(alpha) + math.cos(beta))
        - E * R * y1 * sq(math.cos(beta))
    ) / (E * R * sq(math.cos(alpha)))
    x2 = (
        a1 * Ke * R
        + E * (math.cos(alpha) + math.cos(beta))
        - E * R * y2 * sq(math.cos(beta))
    ) / (E * R * sq(math.cos(alpha)))

    if x1 > 0.0:
        r1 = 1.0 / x1
    elif x2 > 0.0:
        r1 = 1.0 / x2
    else:
        r1 = -1.0
    return r1


def Calc_r2(alphaD, betaD, R, r1, a0, a1):
    alpha = math.radians(alphaD)
    beta = math.radians(betaD)

    A = sq(math.cos(beta))
    B = (
        (math.cos(alpha) + math.cos(beta)) / R
        - sq(math.cos(alpha)) / r1
        + (math.sin(alpha) - math.sin(beta)) * a1 / a0
    )

    r2 = A / B
    return r2


def CalcGamma(betaD, R, r2, a0, a1):
    beta = math.radians(betaD)
    gamma = math.atan2(
        math.cos(beta), 2 * math.sin(beta) - r2 * (math.tan(beta) / R + a1 / a0)
    )
    return math.degrees(gamma)


# Energy as a function of the detector channel
# --------------------------------------------
def Edet(Ectr, r2, alpha, beta, gamma, eGrtgType, iChan, kOrder):
    rc = True
    nChan = None  #          number of vertical channels
    mChan = None  #          channel corresponding to central energy
    E = None  #  [eV]    photon energy detected in the given channel
    a0 = None  # [1/mm]   line density of the grating at center position
    Hcell = None  #  [mm]    height of a detector cell
    # Res,                   # [mm/eV]  resolution
    # Hchan,                 #  [mm]    vertical difference between neighboring channels
    # Echan;                 #  [eV]    energy difference between neighboring channels

    # check energy
    if (Ectr is None) or (Ectr < 200) or (Ectr > 1200):
        rc = False
        return rc

    # set detector parameters
    if eGrtgType:  # this is high-energy grating
        a0 = 2399.4
        nChan = 2048
        mChan = 1024 + (Ectr - 400) / 8
        Hcell = 0.0135
    else:  # this is low-energy grating
        a0 = 2399.972
        nChan = 2048
        mChan = 1024 + (Ectr - 200) / 4
        Hcell = 0.0135

    # determined Energy corresponding to the given detector channel
    if rc:
        Hchan = Hcell * math.sin(math.radians(gamma))
        Res = Eres(Ectr, r2, alpha, beta, a0, kOrder)
        Echan = Hchan / Res
        if iChan is None:
            E = Echan
        else:
            E = Ectr + (iChan - mChan) * Echan
    return E


# calculates dh/dE in mm/eV
# -------------------------
def Eres(E, r2, alpha, beta, a0, k):
    wav = 1e-7 * p_lambda(E)  #  [mm]    photon wavelength of the central energy
    Res = (
        -r2
        * wav
        / E
        * (k * a0 * math.sin(math.radians(alpha)) - k * k * a0 * a0 * wav)
        / (k * math.cos(math.radians(beta)))
    )
    return Res


alphas = np.arange(-90, 90.1, 0.1)
angles = np.zeros((len(alphas), 2))
for n, i in enumerate(alphas):
    angles[n] = sample_chamber_rotation(i)
    # print(i, angles[n])
tilt_values = scint.interp1d(angles[:, 0], angles[:, 1], fill_value="extrapolate")
# print (np.column_stack([np.linspace(-10,90,101), tilt_values(np.linspace(-10,90,101))]))
print(tilt_values(40))

holder_normal = np.array([0.0, -1.0, 0.0])

rotation_centre = np.array([0.0, 0.0, 0.0])

manip_axisx = np.array([1.0, 0.0, 0.0])
manip_axisy = np.array([0.0, 1.0, 0.0])
manip_axisz = np.array([0.0, 0.0, 1.0])

ext_xaxis = manip_axisx.copy().reshape((1, 3))
ext_yaxis = manip_axisy.copy().reshape((1, 3))
ext_zaxis = manip_axisz.copy().reshape((1, 3))
ext_normvec = holder_normal.copy().reshape((1, 3))

holder_dimensions = np.array([18.0, 15.0, 1.0])  # width, length, thickness
holder_offset = np.array([0.0, -1.0, 0.0])  # along x, y, z

holder_vertices = (
    np.array(
        [
            [
                holder_dimensions[1] / 2.0,
                holder_dimensions[2] / 2.0,
                holder_dimensions[0] / 2.0,
            ],
            [
                holder_dimensions[1] / 2.0,
                holder_dimensions[2] / 2.0,
                -holder_dimensions[0] / 2.0,
            ],
            [
                -holder_dimensions[1] / 2.0,
                holder_dimensions[2] / 2.0,
                holder_dimensions[0] / 2.0,
            ],
            [
                -holder_dimensions[1] / 2.0,
                holder_dimensions[2] / 2.0,
                -holder_dimensions[0] / 2.0,
            ],
            [
                holder_dimensions[1] / 2.0,
                -holder_dimensions[2] / 2.0,
                holder_dimensions[0] / 2.0,
            ],
            [
                holder_dimensions[1] / 2.0,
                -holder_dimensions[2] / 2.0,
                -holder_dimensions[0] / 2.0,
            ],
            [
                -holder_dimensions[1] / 2.0,
                -holder_dimensions[2] / 2.0,
                holder_dimensions[0] / 2.0,
            ],
            [
                -holder_dimensions[1] / 2.0,
                -holder_dimensions[2] / 2.0,
                -holder_dimensions[0] / 2.0,
            ],
        ]
    )
    + holder_offset
)

####
####
# This part is going to be tough, since my plan is to convert between 3 different coordinate systems:
# 1. a quasi-2D system of the beam position on the sample holder ("quasi" because of the sample thickness)
# 2. a 3D coordinate system inside the vacuum chamber
# 3. the coordinates defined as the manipulator motor positions.
# Calibration is needed on a regular basis to keep the following details up to date:
# a) the beam position in 3D
# b) the manipulator centre of rotation in 3D.
# As an arbitrary decision, I will define the origin of the coordinate system in the following way:
# 0,0,0 is the bottom-left corner of the sample holder space (where the corner should be, if the holder was a rectangle)
# when XPOS=YPOS=ZPOS=0, ARM_THETA=90, ZDIR=90, INCL=ROT=0.
####
####


def holder_position(
    xyz=np.array([0.0, 0.0, 0.0]),
    angles=np.array([0.0, 0.0, 0.0]),
    arm_theta=90.0,
    rot_offset=np.array([0, 0, 0]),
):
    """
    Returns the vertices of the holder plate after applying all the transformations.
    Takes in the manipulator translations (xyz) and rotations (angles).
    Angles are (zdir, rot, incl).
    The rot_offset corresponds to the position of the centre of rotation.
    """
    points = holder_vertices.copy() + rot_offset
    xaxis = manip_axisx.copy().reshape((1, 3))
    yaxis = manip_axisy.copy().reshape((1, 3))
    zaxis = manip_axisz.copy().reshape((1, 3))
    normvec = holder_normal.copy().reshape((1, 3))
    rotlist = [points, xaxis, yaxis, zaxis, normvec]
    ang_offsets = tilt_values(abs(90 - arm_theta))
    # print(ang_offsets)
    # now we have temporary objects to work on
    # first we tilt the chamber according to the arm theta
    Cham_rot = arb_rotation(rotlist[1][0], -ang_offsets)
    for n, i in enumerate(rotlist):
        rotlist[n] = rotate(Cham_rot, i)
    # then we rotate everything according to zdir
    Z_rot = arb_rotation(rotlist[3][0], 180.0 - angles[0])
    for n, i in enumerate(rotlist):
        rotlist[n] = rotate(Z_rot, i)
    # then we rotate everything according to rot
    Y_rot = arb_rotation(rotlist[2][0], angles[1])
    for n, i in enumerate(rotlist):
        rotlist[n] = rotate(Y_rot, i)
    # then we rotate everything according to incl
    X_rot = arb_rotation(rotlist[1][0], angles[2])
    for n, i in enumerate(rotlist):
        rotlist[n] = rotate(X_rot, i)
    # the last step: the translations
    points = rotlist[0] - rot_offset
    points[:, 0] += xyz[0]
    points[:, 1] += xyz[1]
    points[:, 2] += xyz[2]
    face = points[:4]
    holderx = face[2] - face[3]
    holdery = face[1] - face[3]
    holderx = holder_dimensions[0] * holderx / length(holderx)
    holdery = holder_dimensions[1] * holdery / length(holdery)
    return points, (face, normvec), (holderx, holdery)


# print(holder_vertices)
# print(holder_position(angles = np.array([90.0, 0.0,10.0])))


def which_way(txtinput):
    temp = txtinput.lower().strip()
    return temp[0]


def parse_database(filelist):
    database = []
    for i in filelist:
        source = open(i, "r")
        dict = {}
        for line in source:
            toks = line.split(":")
            leftside = toks[0].strip(" |?/\\][]\n(){}")
            rightside = toks[1].strip(" |?/\\][]\n(){}")
            print(leftside)
            print(rightside)
            try:
                data = np.array([float(x) for x in rightside.split(",")])
            except:
                dict[leftside] = rightside
            else:
                dict[leftside] = data
        print(dict)
        database.append(dict)
    return database


def spot_on_sample(data_dict):
    distortion = data_dict["Matrix"]
    beam = data_dict["BeamMark"]
    edges = data_dict["EdgeMark"]
    trans = data_dict["XYZ"]
    try:
        armtheta = data_dict["ArmTheta"]
    except:
        armtheta = 90.0
    rots = data_dict["ZRI"]
    wway = data_dict["WhichWay"]
    # here we sort out the PIXEL COORDINATES
    ymin = edges[:2].min()
    ymax = edges[:2].max()
    xmin = edges[2:].min()
    xmax = edges[2:].max()
    beamx = (beam[0] - xmin) / (xmax - xmin)
    beamy = (beam[1] - ymin) / (ymax - ymin)
    # now we make sure that the coordinates are relative to the orientation facing the top
    hdir = which_way(wway)
    if hdir == "t":
        beampos = np.array([beamx, beamy, 0.0])
    elif hdir == "b":
        beampos = np.array([1.0 - beamx, 1.0 - beamy, 0.0])
    elif hdir == "l":
        beampos = np.array([1.0 - beamy, beamx, 0.0])
    elif hdir == "r":
        beampos = np.array([beamy, 1.0 - beamx, 0.0])
    else:
        beampos = np.array([0.0, 0.0, 0.0])
    # let's get the spatial coordinates of the beam spot, using the manipulator parameters
    vertices, normal, basis = holder_position(trans, rots, armtheta)
    beamspot = vertices[3] + beampos[0] * basis[0] + beampos[1] * basis[1]
    return beamspot, normal, vertices


def beam_spot_dist(args, spots):
    line_origin = np.array([-1000.0, args[0], args[1]])
    line_vector = args[2:5]
    result = []
    for s in spots:
        result.append(line_point_distance(line_origin, line_vector, s))
    return np.concatenate(result)


def beam_spot_dist_fixed(args, spots):
    line_origin = np.array([-1000.0, args[0], args[1]])
    line_vector = np.array([1.0, 0.0, 0.0])
    result = []
    for s in spots:
        result.append(line_point_distance(line_origin, line_vector, s))
    return np.concatenate(result)


def find_beam(spots, change_direction=False):
    start_dir = np.array([1.0, 0.0, 0.0])
    ys, zs = [], []
    for s in spots:
        ys.append(s[1])
        zs.append(s[2])
    ys = np.array(ys)
    zs = np.array(zs)
    ymean, ysdt = ys.mean(0), ys.std(0)
    zmean, zsdt = zs.mean(0), zs.std(0)
    start_args = np.concatenate([np.array([ymean, zmean]), start_dir])
    if (len(spots) == 1) and not change_direction:
        return np.array([ymean, zmean]), np.array([-1.0, -1.0])
    if not change_direction:
        pfit, pcov, infodict, errmsg, success = scopt.leastsq(
            beam_spot_dist_fixed, start_args[:2], args=(spots), full_output=1
        )
        if pcov is not None:
            s_sq = (beam_spot_dist_fixed(pfit, spots) ** 2).sum() / (
                len(np.array(spots).ravel()) - len(pfit)
            )
            pcov = pcov * s_sq
    else:
        pfit, pcov, infodict, errmsg, success = scopt.leastsq(
            beam_spot_dist, start_args, args=(spots), full_output=1
        )
        if pcov is not None:
            s_sq = (beam_spot_dist(pfit, spots) ** 2).sum() / (
                len(np.array(spots).ravel()) - len(pfit)
            )
            pcov = pcov * s_sq
    error = []
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i]) ** 0.5)
        except:
            error.append(0.00)
    return pfit, error


def vector_components(args, vecs, start, target):
    newone = start + args[0] * vecs[0] + args[1] * vecs[1]
    return newone - target


def find_beam_on_sample(xyz, zri, armtheta, beam_ori, beam_dir):
    vertices, face, basis = holder_position(xyz, zri, armtheta)
    face_verts, face_normal = face[0].copy(), face[1].copy()
    plane_normal = normalise(np.cross(basis[0], basis[1]))
    beam_dir = normalise(np.array(beam_dir))
    point = line_plane_intersection(beam_ori, beam_dir, face_verts[3], plane_normal)
    try:
        plen = len(point)
    except:
        point = None
        onsample = ("Surface parallel to the beam", None)
        return point, onsample, vertices
    pfit, pcov, infodict, errmsg, success = scopt.leastsq(
        vector_components,
        0.5 * np.ones(2),
        args=(basis, face_verts[3], point),
        full_output=1,
    )
    if pcov is not None:
        s_sq = (vector_components(pfit, basis, face_verts[3], point) ** 2).sum() / (
            len(point.ravel()) - len(pfit)
        )
        pcov = pcov * s_sq
    error = []
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i]) ** 0.5)
        except:
            error.append(0.00)
    if np.dot(face_normal, beam_dir) > 0.0:
        onsample = ("Sample facing away from the beam", None)
    elif pfit[0] > 0.0 and pfit[0] < 1.0 and pfit[1] > 0.0 and pfit[1] < 1.0:
        onsample = (
            pfit * np.array([holder_dimensions[0], holder_dimensions[1]]),
            error * np.array([holder_dimensions[0], holder_dimensions[1]]),
        )
    else:
        onsample = ("Beam not on sample", None)
    return point, onsample, vertices


class ComputeKernel(QObject):
    hkl_result = pyqtSignal(object)
    hkl_limits = pyqtSignal(object)
    total_result = pyqtSignal(object)

    def __init__(self, parent):
        if parent is not None:
            super().__init__(parent)
        else:
            super().__init__(parent)
        self.Rmatrix = None
        self.ki = None
        self.kf = None
        self.Q = None
        self.hkl = None
        self.dE = 0.0
        self.Qmod = None
        self.Q_perp = None
        self.Q_par = None
        self.u_rspace = None
        self.v_rspace = None
        self.rubmat = None
        self.inv_rubmat = None
        self.need_new_values = False
        self.lock = QMutex()

    @pyqtSlot(object)
    def minimal_input(self, range_limits):
        self.theta_lims = range_limits[1]["arm_theta"]
        self.Ei_lims = range_limits[1]["Ei"]

    @pyqtSlot(object)
    def parse_input(self, all_inputs):
        sval, inval, rixval = all_inputs
        self.sample_vals = sval
        self.instrument_vals = inval
        self.rixs_vals = rixval
        self.latt_const = np.concatenate(
            [
                self.sample_vals[1]["a"],
                self.sample_vals[1]["b"],
                self.sample_vals[1]["c"],
            ]
        )
        self.latt_angs = np.concatenate(
            [
                self.sample_vals[1]["alpha"],
                self.sample_vals[1]["beta"],
                self.sample_vals[1]["gamma"],
            ]
        )
        # print(latt_const)
        # print(latt_angs)
        self.u, self.v = self.sample_vals[1]["u"], self.sample_vals[1]["v"]
        self.manip_angles = np.concatenate(
            [
                self.instrument_vals[1]["incl"],
                self.instrument_vals[1]["rot"],
                self.instrument_vals[1]["zdir"],
            ]
        )
        if "dE" in self.rixs_vals[1].keys():
            self.dE = self.rixs_vals[1]["dE"]
        self.gonio_angles = self.sample_vals[1]["misalignment"]
        self.arm_theta = self.rixs_vals[1]["arm_theta"][0]
        self.scatt_angle = np.radians(self.rixs_vals[1]["arm_theta"])[0]
        self.Ei = self.rixs_vals[1]["Ei"]
        self.lock.lock()
        self.need_new_values = True
        self.lock.unlock()

    @pyqtSlot()
    def start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.produce_values)
        self.timer.setInterval(1000)
        self.timer.start()

    @pyqtSlot()
    def produce_values(self):
        result = False
        self.lock.lock()
        result = self.need_new_values == True
        self.lock.unlock()
        if result:
            self.calc_Rmatrix()
            self.calc_hkl()
            self.lock.lock()
            self.need_new_values = False
            self.lock.unlock()

    def new_sample(self, latt_abc, latt_angs, uvec, vvec, gonio):
        self.latt_const = latt_abc
        self.latt_angs = latt_angs
        self.u = uvec
        self.v = vvec
        self.gonio_angles = gonio
        self.need_new_values = True

    def new_angles(self, angs):
        self.manip_angles = np.concatenate([angs[1], angs[2], angs[0]])
        self.calc_Rmatrix()

    def new_hkl_fast(self):
        self.calc_hkl_fast()
        return self.hkl, self.hkl_lims

    def new_hkl(self):
        self.calc_hkl()
        return self.hkl, self.hkl_lims

    def new_fullparams(self, args, Ei):
        # ARM_THETA H K L THETA INPLANE_H INPLANE_K INPLANE_L Q_PAR Q_PERP
        pdict = {}
        plist = [
            "ARM_THETA",
            "H",
            "K",
            "L",
            "THETA",
            "INPLANE_H",
            "INPLANE_K",
            "INPLANE_L",
            "Q",
            "Q_PAR",
            "Q_PERP",
        ]
        argcount = 0
        for n, par in enumerate(fixedpars):
            if flags[n]:
                pdict[plist[n]] = par
            else:
                pdict[plist[n]] = None

    def new_params(self, angs, atheta, ener):
        self.Ei = ener
        self.arm_theta = atheta
        self.scatt_angle = np.radians(atheta)
        incl = angs[1]
        rot = angs[2]
        zdir = angs[0]
        if rot < 0.0:
            rot += 360.0
        self.manip_angles = np.array([incl, rot, zdir])
        self.calc_Rmatrix()
        return True

    def current_Q_par(self):
        # print('current u vector = ',  self.u_rspace)
        self.Q_par = np.dot(self.Q, self.u_rspace)
        return self.Q_par

    def current_Q_perp(self):
        # print('current v vector = ',  self.v_rspace)
        self.Q_perp = np.dot(self.Q, self.v_rspace)
        return self.Q_perp

    def current_Q_par_inplane(self):
        # print('current u vector = ',  self.u_rspace)
        new_uvec = self.u_rspace.copy()
        new_uvec[-1] = 0.0
        new_uvec = normalise(new_uvec)
        new_qpar = np.dot(self.Q, new_uvec)
        return new_qpar

    def current_Q_perp_inplane(self):
        # print('current v vector = ',  self.v_rspace)
        new_vvec = self.v_rspace.copy()
        new_vvec[-1] = 0.0
        new_vvec = normalise(new_vvec)
        new_qperp = np.dot(self.Q, new_vvec)
        return new_qperp

    @pyqtSlot()
    def total_coverage(self):
        Ei = np.arange(self.Ei_lims[0], self.Ei_lims[2], self.Ei_lims[1])
        scatt_angle = np.arange(
            self.theta_lims[0],
            self.theta_lims[2] + self.theta_lims[1] * 0.1,
            self.theta_lims[1],
        )
        scatt_angle_rad = np.radians(scatt_angle)
        plambda = p_lambda(Ei)
        pmomentum = 2 * np.pi / plambda
        result = np.zeros([len(Ei), len(scatt_angle)])
        for n, pm in enumerate(pmomentum):
            ki = np.array([1.0, 0.0, 0.0]) * pm
            kf = (
                np.column_stack(
                    [
                        np.cos(scatt_angle_rad),
                        np.sin(scatt_angle_rad),
                        np.zeros(scatt_angle.shape),
                    ]
                )
                * pm
            )
            Q = kf - ki
            Qmod = length(Q)
            result[n, :] = Qmod
        qmin = result.min()
        qmax = result.max()
        scalefac = np.array([1.0, 1.0])
        qstep = (qmax - qmin) / len(scatt_angle) / scalefac[1]
        qax = np.arange(qmin - qstep * 0.5, qmax + qstep * 0.501, qstep)
        Emin = Ei.min()
        Emax = Ei.max()
        Estep = (Emax - Emin) / len(Ei)
        Eax = np.arange(Emin - Estep * 0.5, Emax + Estep * 0.501, Estep)
        remapped_result = np.zeros(result.shape * scalefac.astype(int))
        remapped_norm = np.zeros(result.shape * scalefac.astype(int)).astype(int)
        gap_filler = np.zeros(result.shape * scalefac.astype(int)).astype(int)
        replaced_vals = np.zeros(result.shape * scalefac.astype(int))
        for nn in range(len(Ei)):
            for mm in range(len(scatt_angle)):
                qval = result[nn][mm]
                newind = int((qval - qmin) / qstep)
                if newind >= (len(qax) - 2):
                    newind = -1
                newval = scatt_angle[mm]
                remapped_result[nn][newind] += newval
                remapped_norm[nn][newind] += 1
        remapped_result /= remapped_norm
        for nn in range(len(Ei) - 2):
            gap_filler[1 + nn, :] = remapped_norm[nn, :] + remapped_norm[nn + 2, :]
            replaced_vals[1 + nn, :] = (
                remapped_result[nn, :] + remapped_result[nn + 2, :]
            ) / 2.0
        crit = np.where(np.logical_and(gap_filler > 1, remapped_result == 0.0))
        remapped_result[crit] = replaced_vals[crit]
        self.total_result.emit([Eax, qax, scatt_angle, remapped_result])

    def calc_mixed(self):
        temp_sample = Sample(self.latt_const, self.latt_angs)
        #        if self.Rmatrix is not None:
        #            invRmatrix = np.linalg.inv(self.Rmatrix)
        #            temp_u = np.dot(invRmatrix,  self.u)
        #            temp_v = np.dot(invRmatrix,  self.v)
        #            # temp_u = np.dot(temp_sample.Binv, np.dot(self.Rmatrix,  np.dot(temp_sample.B, self.u)))
        #            # temp_v = np.dot(temp_sample.Binv, np.dot(self.Rmatrix,  np.dot(temp_sample.B, self.v)))
        #            temp_sample.orient(temp_u,  temp_v,  0.0,  self.gonio_angles)
        #            # self.u_rspace = np.dot(invRmatrix, np.array([1.0, 0.0, 0.0]))
        #            # self.v_rspace = np.dot(invRmatrix, np.array([0.0, 1.0, 0.0]))
        #            self.u_rspace = np.dot(temp_sample.Rspace_Rmatrix, self.temp_u_rspace)
        #            self.v_rspace = np.dot(temp_sample.Rspace_Rmatrix, self.temp_v_rspace)
        #        else:
        #            temp_sample.orient(self.u, self.v, 0.0, self.gonio_angles)
        #            self.u_rspace = np.array([1.0, 0.0, 0.0])
        #            self.v_rspace = np.array([0.0, 1.0, 0.0])
        temp_sample.orient(self.u, self.v, 0.0, self.gonio_angles)
        rmat = self.Rmatrix
        ubmat = temp_sample.themat
        rubmat = np.dot(rmat, ubmat)
        # print('u vector = ', normalise(np.dot(rubmat, [1, 0, 0])))
        # print('v vector = ', normalise(np.dot(rubmat, [0, 1, 0])))
        inv_rubmat = np.linalg.inv(rubmat)
        self.rubmat = rubmat
        self.inv_rubmat = inv_rubmat
        Ei = self.Ei
        if Ei >= 600:
            HiE = True
        else:
            HiE = False
        try:
            alpha, beta, gamma, r1, r2, rc = CalcPar(HiE, Ei)
        except:
            angle_var = 0.0
            angle_var_vert = 0.0
        else:
            angle_var = math.atan2(13.5, r2)  # half the detector width is 13.5 mm
            grt_height = (
                math.sin(math.radians(90.0 - alpha)) * 150.0
            )  # projection of the grating onto a vertical axis
            angle_var_vert = math.atan2(
                grt_height / 2.0, r1
            )  # half the grating length is 75 mm.
        plambda = p_lambda(Ei)
        flambda = p_lambda(Ei - self.dE)
        # print(Ei,  plambda)
        pmomentum = 2 * np.pi / plambda
        fmomentum = 2 * np.pi / flambda
        self.ki = np.array([1.0, 0.0, 0.0]) * pmomentum
        self.kf = (
            np.array([np.cos(self.scatt_angle), np.sin(self.scatt_angle), 0.0])
            * fmomentum
        )
        # new part for resolution
        all_angles = np.linspace(
            self.scatt_angle - angle_var, self.scatt_angle + angle_var, 3
        )
        all_angles = np.concatenate([all_angles, all_angles, all_angles])
        vert_angles = (
            np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]) * angle_var_vert
        )
        kf_spread = (
            normalise(
                np.column_stack(
                    [np.cos(all_angles), np.sin(all_angles), np.sin(vert_angles)]
                )
            )
            * fmomentum
        )
        all_Q = kf_spread - self.ki
        all_hkl = fast_rotate(inv_rubmat, all_Q / (2 * np.pi))
        min_hkl = all_hkl.min(0)
        max_hkl = all_hkl.max(0)
        #
        self.Q = self.kf - self.ki
        self.Q_perp = np.dot(self.Q, self.v_rspace)
        self.hkl = np.dot(inv_rubmat, self.Q / (2 * np.pi))
        self.hkl_lims = [min_hkl, max_hkl]
        self.Qmod = length(self.Q)
        if self.Qmod > abs(self.Q_perp):
            self.Q_par = math.sqrt(self.Qmod**2 - self.Q_perp**2)
        else:
            self.Q_par = 0.0
        self.hkl_result.emit(
            [[self.Qmod, self.Q_perp, self.Q_par], self.hkl, self.plot_vectors()]
        )
        self.hkl_limits.emit([min_hkl, max_hkl])

    def calc_hkl_fast(self):
        temp_sample = Sample(self.latt_const, self.latt_angs)
        temp_sample.orient(self.u, self.v, 0.0, self.gonio_angles)
        rmat = self.Rmatrix
        ubmat = temp_sample.themat
        rubmat = np.dot(rmat, ubmat)
        # calc theta
        samp_normal = np.dot(rmat, manip_axisy)
        projected_normal = samp_normal.copy()
        projected_normal[2] = 0.0
        projected_normal = normalise(projected_normal)
        # theta = np.degrees(np.arccos(np.dot(projected_normal, manip_axisy)))
        theta = np.degrees(np.arccos(np.dot(samp_normal, manip_axisy)))
        # print('u vector = ', normalise(np.dot(rubmat, [1, 0, 0])))
        # print('v vector = ', normalise(np.dot(rubmat, [0, 1, 0])))
        inv_rubmat = np.linalg.inv(rubmat)
        self.rubmat = rubmat
        self.inv_rubmat = inv_rubmat
        Ei = self.Ei
        plambda = p_lambda(Ei)
        flambda = p_lambda(Ei - self.dE)
        # print(Ei,  plambda)
        pmomentum = 2 * np.pi / plambda
        fmomentum = 2 * np.pi / flambda
        self.ki = np.array([1.0, 0.0, 0.0]) * pmomentum
        self.kf = (
            np.array([np.cos(self.scatt_angle), np.sin(self.scatt_angle), 0.0])
            * fmomentum
        )
        #
        self.theta = min(theta, 180.0 - theta)
        self.Q = self.kf - self.ki
        self.Q_perp = np.dot(self.Q, self.v_rspace)
        self.hkl = np.dot(inv_rubmat, self.Q / (2 * np.pi))
        self.hkl_lims = [self.hkl, self.hkl]
        self.Qmod = length(self.Q)
        if self.Qmod > abs(self.Q_perp):
            self.Q_par = math.sqrt(self.Qmod**2 - self.Q_perp**2)
        else:
            self.Q_par = 0.0
        # self.hkl_result.emit([[self.Qmod,  self.Q_perp,  self.Q_par], self.hkl,  self.plot_vectors(), self.theta])

    def calc_hkl(self):
        temp_sample = Sample(self.latt_const, self.latt_angs)
        #        if self.Rmatrix is not None:
        #            invRmatrix = np.linalg.inv(self.Rmatrix)
        #            temp_u = np.dot(invRmatrix,  self.u)
        #            temp_v = np.dot(invRmatrix,  self.v)
        #            # temp_u = np.dot(temp_sample.Binv, np.dot(self.Rmatrix,  np.dot(temp_sample.B, self.u)))
        #            # temp_v = np.dot(temp_sample.Binv, np.dot(self.Rmatrix,  np.dot(temp_sample.B, self.v)))
        #            temp_sample.orient(temp_u,  temp_v,  0.0,  self.gonio_angles)
        #            # self.u_rspace = np.dot(invRmatrix, np.array([1.0, 0.0, 0.0]))
        #            # self.v_rspace = np.dot(invRmatrix, np.array([0.0, 1.0, 0.0]))
        #            self.u_rspace = np.dot(temp_sample.Rspace_Rmatrix, self.temp_u_rspace)
        #            self.v_rspace = np.dot(temp_sample.Rspace_Rmatrix, self.temp_v_rspace)
        #        else:
        #            temp_sample.orient(self.u, self.v, 0.0, self.gonio_angles)
        #            self.u_rspace = np.array([1.0, 0.0, 0.0])
        #            self.v_rspace = np.array([0.0, 1.0, 0.0])
        temp_sample.orient(self.u, self.v, 0.0, self.gonio_angles)
        rmat = self.Rmatrix
        ubmat = temp_sample.themat
        rubmat = np.dot(rmat, ubmat)
        # calc theta
        samp_normal = np.dot(rmat, manip_axisy)
        projected_normal = samp_normal.copy()
        projected_normal[2] = 0.0
        projected_normal = normalise(projected_normal)
        # theta = np.degrees(np.arccos(np.dot(projected_normal, manip_axisy)))
        theta = np.degrees(np.arccos(np.dot(samp_normal, manip_axisy)))
        # print('u vector = ', normalise(np.dot(rubmat, [1, 0, 0])))
        # print('v vector = ', normalise(np.dot(rubmat, [0, 1, 0])))
        inv_rubmat = np.linalg.inv(rubmat)
        self.rubmat = rubmat
        self.inv_rubmat = inv_rubmat
        Ei = self.Ei
        if Ei >= 600:
            HiE = True
        else:
            HiE = False
        try:
            alpha, beta, gamma, r1, r2, rc = CalcPar(HiE, Ei)
        except:
            angle_var = 0.0
            angle_var_vert = 0.0
        else:
            angle_var = math.atan2(13.5, r2)  # half the detector width is 13.5 mm
            grt_height = (
                math.sin(math.radians(90.0 - alpha)) * 150.0
            )  # projection of the grating onto a vertical axis
            angle_var_vert = math.atan2(
                grt_height / 2.0, r1
            )  # half the grating length is 75 mm.
        plambda = p_lambda(Ei)
        flambda = p_lambda(Ei - self.dE)
        # print(Ei,  plambda)
        pmomentum = 2 * np.pi / plambda
        fmomentum = 2 * np.pi / flambda
        self.ki = np.array([1.0, 0.0, 0.0]) * pmomentum
        self.kf = (
            np.array([np.cos(self.scatt_angle), np.sin(self.scatt_angle), 0.0])
            * fmomentum
        )
        # new part for resolution
        all_angles = np.linspace(
            self.scatt_angle - angle_var, self.scatt_angle + angle_var, 3
        )
        all_angles = np.concatenate([all_angles, all_angles, all_angles])
        vert_angles = (
            np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]) * angle_var_vert
        )
        kf_spread = (
            normalise(
                np.column_stack(
                    [np.cos(all_angles), np.sin(all_angles), np.sin(vert_angles)]
                )
            )
            * fmomentum
        )
        all_Q = kf_spread - self.ki
        all_hkl = fast_rotate(inv_rubmat, all_Q / (2 * np.pi))
        min_hkl = all_hkl.min(0)
        max_hkl = all_hkl.max(0)
        #
        self.theta = min(theta, 180.0 - theta)
        self.Q = self.kf - self.ki
        self.Q_perp = np.dot(self.Q, self.v_rspace)
        self.hkl = np.dot(inv_rubmat, self.Q / (2 * np.pi))
        self.hkl_lims = [min_hkl, max_hkl]
        self.Qmod = length(self.Q)
        if self.Qmod > abs(self.Q_perp):
            self.Q_par = math.sqrt(self.Qmod**2 - self.Q_perp**2)
        else:
            self.Q_par = 0.0
        printMatrix(temp_sample.G, "G matrix")
        printMatrix(temp_sample.B, "B matrix")
        printMatrix(temp_sample.U, "U matrix")
        printMatrix(rmat, "R matrix")
        self.hkl_result.emit(
            [
                [self.Qmod, self.Q_perp, self.Q_par],
                self.hkl,
                self.plot_vectors(),
                self.theta,
            ]
        )
        self.hkl_limits.emit([min_hkl, max_hkl])

    # @pyqtSlot()
    #    def old_calc_Rmatrix(self):
    #        # xaxis = manip_axisx.copy().reshape((1,3))
    #        # yaxis = manip_axisy.copy().reshape((1,3))
    #        # zaxis = manip_axisz.copy().reshape((1,3))
    #        # normvec = holder_normal.copy().reshape((1,3))
    #        # rotlist = [xaxis, yaxis, zaxis, normvec]
    #        ang_offsets = tilt_values(abs(90-self.arm_theta))
    #        angles = self.manip_angles
    ##        # print(ang_offsets)
    ##        # now we have temporary objects to work on
    ##        # first we tilt the chamber according to the arm theta
    ##        Cham_rot = arb_rotation(rotlist[1][0], -ang_offsets)
    ##        for n, i in enumerate(rotlist):
    ##            rotlist[n] = rotate(Cham_rot, i)
    ##        # then we rotate everything according to zdir
    ##        Z_rot = arb_rotation(rotlist[3][0], angles[2])
    ##        for n, i in enumerate(rotlist):
    ##            rotlist[n] = rotate(Z_rot, i)
    ##        # then we rotate everything according to rot
    ##        Y_rot = arb_rotation(rotlist[2][0], angles[1])
    ##        for n, i in enumerate(rotlist):
    ##            rotlist[n] = rotate(Y_rot, i)
    ##        # then we rotate everything according to incl
    ##        X_rot = arb_rotation(rotlist[1][0], angles[0])
    ##        for n, i in enumerate(rotlist):
    ##            rotlist[n] = rotate(X_rot, i)
    #        #first the chamber tilt due to arm motion
    #        Cham_rot = arb_rotation(xaxis[0], ang_offsets)
    #        a = rotate(Cham_rot, xaxis)
    #        b = rotate(Cham_rot, yaxis)
    #        c = rotate(Cham_rot, zaxis)
    #        # then we rotate everything according to zdir
    #        Z_rot = arb_rotation(c[0], angles[2])
    #        a = rotate(Z_rot, a)
    #        b = rotate(Z_rot, b)
    #        c = rotate(Z_rot, c)
    #        # then the tilt around the x-direction due to inclination
    #        X_rot = arb_rotation(a[0], angles[0])
    #        a = rotate(X_rot, a)
    #        b = rotate(X_rot, b)
    #        c = rotate(X_rot, c)
    #        # then we rotate everything according to rot
    #        Y_rot = arb_rotation(b[0], angles[1])
    #        a = rotate(Y_rot, a)
    #        b = rotate(Y_rot, b)
    #        c = rotate(Y_rot, c)
    #        self.u_rspace = a[0].copy()
    #        self.v_rspace = b[0].copy()
    #        rotmat = np.dot(Z_rot,  Cham_rot)
    #        rotmat = np.dot(X_rot, rotmat)
    #        rotmat = np.dot(Y_rot, rotmat)
    #        # rotmat = np.dot(rotmat,  X_rot)
    #        self.Rmatrix = rotmat.copy()
    @pyqtSlot()
    def calc_Rmatrix(self):
        self.u_rspace, self.v_rspace, self.Rmatrix = fast_Rmatrix(
            tilt_values(abs(90 - self.arm_theta)),
            self.manip_angles,
            ext_xaxis,
            ext_yaxis,
            ext_zaxis,
        )

    def hkl_to_q(self, vec):
        if self.rubmat is not None:
            return np.dot(self.rubmat, vec)
        else:
            return None

    def plot_vectors(self):
        """
        Each vector should be defined as two points, so that they can be plotted easily.
        This means a pair (start_point(3), end_point(3)), which can be drawn as an arrow.
        It will be up to the plotting function to choose the projection and to cast
        the vectors onto a 2D plane.
        """
        vdict = {}
        # vdict['k$_{i}$'] = (-self.ki, np.zeros(3))
        vdict["-k$_{i}$"] = (self.kf, self.kf - self.ki)
        vdict["k$_{f}$"] = (np.zeros(3), self.kf)
        vdict["u$_{ideal}$"] = (np.zeros(3), np.array([1, 0, 0]))
        vdict["v$_{ideal}$"] = (np.zeros(3), np.array([0, 1, 0]))
        vdict["Q"] = (np.zeros(3), self.Q)
        vdict["u$_{real}$"] = (np.zeros(3), self.u_rspace)
        vdict["v$_{real}$"] = (np.zeros(3), self.v_rspace)
        return vdict

    def specular_pos(self, arm_th):
        zdir = arm_th / 2.0
        incl = tilt_values(abs(90 - arm_th))
        rot = 0 * arm_th
        return [zdir, incl, rot]


class ReverseKernel(QObject):
    motors_result = pyqtSignal(object)
    hkl_result = pyqtSignal(object)
    hkl_limits = pyqtSignal(object)

    def __init__(self, parent):
        if parent is not None:
            super().__init__(parent)
        else:
            super().__init__(parent)
        self.Rmatrix = None
        self.ki = None
        self.kf = None
        self.Q = None
        self.hkl = None
        self.Ei = None
        self.dE = 0.0
        self.Qmod = None
        self.Q_perp = None
        self.Q_par = None
        self.u_rspace = None
        self.v_rspace = None
        self.manip_angles = None
        self.arm_theta = None
        self.dontproceed = True
        self.need_new_values = False
        self.innerCK = ComputeKernel(None)
        self.lock = QMutex()

    @pyqtSlot(object)
    def parse_input(self, all_inputs):
        sval, inval = all_inputs
        self.sample_vals = sval
        self.instrument_vals = inval
        self.latt_const = np.concatenate(
            [
                self.sample_vals[1]["a"],
                self.sample_vals[1]["b"],
                self.sample_vals[1]["c"],
            ]
        )
        self.latt_angs = np.concatenate(
            [
                self.sample_vals[1]["alpha"],
                self.sample_vals[1]["beta"],
                self.sample_vals[1]["gamma"],
            ]
        )
        # print(latt_const)
        # print(latt_angs)
        self.u, self.v = self.sample_vals[1]["u"], self.sample_vals[1]["v"]
        self.gonio_angles = self.sample_vals[1]["misalignment"]
        self.hkl = self.instrument_vals[1]["HKL"]
        self.Ei = self.instrument_vals[1]["Ei"][0]
        self.dE = self.instrument_vals[1]["dE"][0]
        self.inplane_vec = self.instrument_vals[1]["lock"]
        # self.arm_theta = self.rixs_vals[1]['arm_theta'][0]
        # self.scatt_angle = np.radians(self.rixs_vals[1]['arm_theta'])[0]
        self.lock.lock()
        self.need_new_values = True
        self.lock.unlock()

    @pyqtSlot()
    def start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.produce_values)
        self.timer.setInterval(1000)
        self.timer.start()

    @pyqtSlot()
    def produce_values(self):
        result = False
        self.lock.lock()
        result = self.need_new_values == True
        self.lock.unlock()
        if result:
            self.calc_Rmatrix()
            self.calc_hkl()
            self.lock.lock()
            self.need_new_values = False
            self.lock.unlock()

    def calc_Q(self):
        temp_sample = Sample(self.latt_const, self.latt_angs)
        temp_sample.orient(self.u, self.v, 0.0, self.gonio_angles)
        # rmat = self.Rmatrix
        ubmat = temp_sample.themat
        # rubmat = np.dot(rmat, ubmat)
        # print('u vector = ', normalise(np.dot(rubmat, [1, 0, 0])))
        # print('v vector = ', normalise(np.dot(rubmat, [0, 1, 0])))
        inv_ubmat = np.linalg.inv(ubmat)
        Ei = self.Ei
        plambda = p_lambda(Ei)
        flambda = p_lambda(Ei - self.dE)
        # print(Ei,  plambda)
        pmomentum = 2 * np.pi / plambda
        fmomentum = 2 * np.pi / flambda
        # new section
        Qabs = length(np.dot(ubmat, self.hkl) * 2 * np.pi)
        theta = np.degrees(np.arcsin(abs(Qabs / (2.0 * pmomentum))) * 2)
        self.arm_theta = theta
        self.scatt_angle = np.radians(theta)
        self.ki = np.array([1.0, 0.0, 0.0]) * pmomentum
        self.kf = (
            np.array([np.cos(self.scatt_angle), np.sin(self.scatt_angle), 0.0])
            * fmomentum
        )
        self.Q = self.kf - self.ki
        # self.Q_perp = np.dot(self.Q,  self.v_rspace)
        # self.hkl = np.dot(inv_rubmat,  self.Q/(2*np.pi))
        self.Qmod = length(self.Q)
        # if self.Qmod > abs(self.Q_perp):
        #     self.Q_par = math.sqrt(self.Qmod**2 - self.Q_perp**2)
        # else:
        #     self.Q_par = 0.0
        # another new section
        Rhkl = 0.5 / np.pi * np.dot(inv_ubmat, self.Q)
        if length(self.hkl) < 1e-14:
            self.dontproceed = True
            return None
        else:
            self.dontproceed = False
            temp_rotmat = rotation_matrix_from_vectors(self.hkl, Rhkl)
            self.cart_rotmat = temp_rotmat
            # self.cart_rotmat = np.linalg.inv(rotation_matrix_from_vectors(Rhkl, self.hkl))
        rot_x = np.degrees(np.arctan2(temp_rotmat[2, 1], temp_rotmat[2, 2]))
        rot_y = np.degrees(
            np.arctan2(
                -temp_rotmat[2, 0],
                (temp_rotmat[2, 2] ** 2 + temp_rotmat[2, 1] ** 2) ** 0.5,
            )
        )
        rot_z = np.degrees(np.arctan2(temp_rotmat[1, 0], temp_rotmat[0, 0]))
        # print("Absolute rotations around x,y,z:",  rot_x, rot_y, rot_z)

    @pyqtSlot()
    def calc_Rmatrix(self):
        if self.dontproceed:
            return None
        xaxis = manip_axisx.copy().reshape((1, 3))
        yaxis = manip_axisy.copy().reshape((1, 3))
        zaxis = manip_axisz.copy().reshape((1, 3))
        normvec = holder_normal.copy().reshape((1, 3))
        vertvec = np.array([[0, 0, 1.0]])
        ang_offsets = tilt_values(abs(90 - self.arm_theta))
        # first the chamber tilt due to arm motion
        Cham_rot = fast_rotation(xaxis[0], ang_offsets)
        final_zaxis = fast_rotate(Cham_rot, zaxis)  # first axis is KNOWN
        final_yaxis = fast_rotate(self.cart_rotmat, yaxis)  # ROT axis is KNOWN
        final_xaxis = np.cross(final_zaxis, final_yaxis)  #
        #
        zdir = angle_v1v2(final_xaxis, np.array([1.0, 0, 0]))[0]
        #
        Z_rot = fast_rotation(final_zaxis[0], zdir)
        vt = fast_rotate(Z_rot, fast_rotate(Cham_rot, yaxis))
        incl = angle_v1v2(vt, final_yaxis[0])[0]
        #
        X_rot = fast_rotation(final_xaxis[0], incl)
        temp1 = fast_rotate(self.cart_rotmat, vertvec)
        temp2 = fast_rotate(X_rot, fast_rotate(Z_rot, fast_rotate(Cham_rot, vertvec)))
        rot = angle_v1v2(temp1, temp2[0])[0]
        self.motors_result.emit([self.arm_theta, zdir, incl, rot])

    def find_angles_constrained(self):
        if self.dontproceed:
            return None
        angs = [tilt_values(abs(90 - self.arm_theta)), 0.0, self.arm_theta / 2.0]
        rixsvals = [], {"arm_theta": [self.arm_theta], "Ei": self.Ei, "dE": self.dE}
        instvals = [], {
            "incl": [tilt_values(abs(90 - self.arm_theta))],
            "rot": [0],
            "zdir": [self.arm_theta / 2.0],
        }
        self.innerCK.parse_input([self.sample_vals, instvals, rixsvals])
        results = scopt.minimize(
            iter_angleseek_constrained,
            angs,
            args=(self.innerCK, self.hkl, self.inplane_vec),
            method="Nelder-Mead",
        )
        self.answer_angle = results["x"]
        self.motors_result.emit(
            [
                self.arm_theta,
                self.answer_angle[2],
                self.answer_angle[0],
                self.answer_angle[1],
            ]
        )

    def find_angles(self):
        if self.dontproceed:
            return None
        if np.abs(self.inplane_vec).sum() > 1e-10:
            self.find_angles_constrained()
            return None
        angs = [tilt_values(abs(90 - self.arm_theta)), 0.0, self.arm_theta / 2.0]
        rixsvals = [], {"arm_theta": [self.arm_theta], "Ei": self.Ei, "dE": self.dE}
        instvals = [], {
            "incl": [tilt_values(abs(90 - self.arm_theta))],
            "rot": [0],
            "zdir": [self.arm_theta / 2.0],
        }
        self.innerCK.parse_input([self.sample_vals, instvals, rixsvals])
        results = scopt.minimize(
            iter_angleseek, angs, args=(self.innerCK, self.hkl), method="Nelder-Mead"
        )
        self.answer_angle = results["x"]
        self.motors_result.emit(
            [
                self.arm_theta,
                self.answer_angle[2],
                self.answer_angle[0],
                self.answer_angle[1],
            ]
        )

    def plot_vectors(self):
        """
        Each vector should be defined as two points, so that they can be plotted easily.
        This means a pair (start_point(3), end_point(3)), which can be drawn as an arrow.
        It will be up to the plotting function to choose the projection and to cast
        the vectors onto a 2D plane.
        """
        vdict = {}
        # vdict['k$_{i}$'] = (-self.ki, np.zeros(3))
        vdict["-k$_{i}$"] = (self.kf, self.kf - self.ki)
        vdict["k$_{f}$"] = (np.zeros(3), self.kf)
        vdict["u$_{ideal}$"] = (np.zeros(3), np.array([1, 0, 0]))
        vdict["v$_{ideal}$"] = (np.zeros(3), np.array([0, 1, 0]))
        vdict["Q"] = (np.zeros(3), self.Q)
        vdict["u$_{real}$"] = (np.zeros(3), self.u_rspace)
        vdict["v$_{real}$"] = (np.zeros(3), self.v_rspace)
        return vdict

    def verify_hkl(self, motor_dict):
        if self.dontproceed:
            return None
        angs = [tilt_values(abs(90 - self.arm_theta)), 0.0, self.arm_theta / 2.0]
        rixsvals = [], {"arm_theta": [self.arm_theta], "Ei": self.Ei}
        instvals = [], {
            "incl": [tilt_values(abs(90 - self.arm_theta))],
            "rot": [0],
            "zdir": [self.arm_theta / 2.0],
        }
        self.innerCK.parse_input([self.sample_vals, instvals, rixsvals])
        self.innerCK.new_angles(
            np.concatenate(
                [
                    motor_dict["incl"],
                    motor_dict["rot"],
                    motor_dict["zdir"],
                ]
            )
        )
        hkl, hkl_lims = self.innerCK.new_hkl()
        self.hkl_result.emit(hkl)
        self.hkl_limits.emit(hkl_lims)


def iter_angleseek(angs, CK, hkl):
    CK.new_angles(angs)
    newhkl, newhkl_lims = CK.new_hkl()
    return ((newhkl - hkl) ** 2).sum()


def iter_angleseek_constrained(angs, CK, hkl, cvec):
    CK.new_angles(angs)
    newhkl, newhkl_lims = CK.new_hkl()
    inplane = CK.hkl_to_q(cvec)
    return ((newhkl - hkl) ** 2).sum() + abs(np.dot(inplane, [0, 0, 1]))


class TiltedKernel(QObject):
    motors_result = pyqtSignal(object)
    hkl_result = pyqtSignal(object)
    hkl_limits = pyqtSignal(object)

    def __init__(self, parent):
        if parent is not None:
            super().__init__(parent)
        else:
            super().__init__(parent)
        self.Rmatrix = None
        self.ki = None
        self.kf = None
        self.Q = None
        self.hkl = None
        self.Ei = None
        self.dE = 0.0
        self.Qmod = None
        self.Q_perp = None
        self.Q_par = None
        self.u_rspace = None
        self.v_rspace = None
        self.manip_angles = None
        self.arm_theta = None
        self.dontproceed = True
        self.need_new_values = False
        self.innerCK = ComputeKernel(None)
        self.lock = QMutex()

    @pyqtSlot(object)
    def parse_input(self, all_inputs):
        sval, inval = all_inputs
        self.sample_vals = sval
        self.instrument_vals = inval
        self.latt_const = np.concatenate(
            [
                self.sample_vals[1]["a"],
                self.sample_vals[1]["b"],
                self.sample_vals[1]["c"],
            ]
        )
        self.latt_angs = np.concatenate(
            [
                self.sample_vals[1]["alpha"],
                self.sample_vals[1]["beta"],
                self.sample_vals[1]["gamma"],
            ]
        )
        # print(latt_const)
        # print(latt_angs)
        self.u, self.v = self.sample_vals[1]["u"], self.sample_vals[1]["v"]
        self.gonio_angles = self.sample_vals[1]["misalignment"]
        self.arm_theta = self.instrument_vals[1]["arm_theta"][0]
        self.Ei = self.instrument_vals[1]["Ei"][0]
        self.Q_par = self.instrument_vals[1]["Q_par"][0]
        self.inplane_vec = self.instrument_vals[1]["lock"]
        # self.arm_theta = self.rixs_vals[1]['arm_theta'][0]
        # self.scatt_angle = np.radians(self.rixs_vals[1]['arm_theta'])[0]
        self.lock.lock()
        self.need_new_values = True
        self.lock.unlock()

    @pyqtSlot()
    def start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.produce_values)
        self.timer.setInterval(1000)
        self.timer.start()

    @pyqtSlot()
    def produce_values(self):
        result = False
        self.lock.lock()
        result = self.need_new_values == True
        self.lock.unlock()
        if result:
            self.calc_Rmatrix()
            self.calc_hkl()
            self.lock.lock()
            self.need_new_values = False
            self.lock.unlock()

    def calc_Q(self):
        temp_sample = Sample(self.latt_const, self.latt_angs)
        temp_sample.orient(self.u, self.v, 0.0, self.gonio_angles)
        # rmat = self.Rmatrix
        ubmat = temp_sample.themat
        # rubmat = np.dot(rmat, ubmat)
        # print('u vector = ', normalise(np.dot(rubmat, [1, 0, 0])))
        # print('v vector = ', normalise(np.dot(rubmat, [0, 1, 0])))
        inv_ubmat = np.linalg.inv(ubmat)
        Ei = self.Ei
        plambda = p_lambda(Ei)
        flambda = p_lambda(Ei - self.dE)
        # print(Ei,  plambda)
        pmomentum = 2 * np.pi / plambda
        fmomentum = 2 * np.pi / flambda
        # new section
        # Qabs = length(np.dot(ubmat, self.hkl)*2*np.pi)
        # theta = np.degrees(np.arcsin(abs(Qabs / (2.0*pmomentum)))*2)
        # self.arm_theta = theta
        self.scatt_angle = np.radians(self.arm_theta)
        self.ki = np.array([1.0, 0.0, 0.0]) * pmomentum
        self.kf = (
            np.array([np.cos(self.scatt_angle), np.sin(self.scatt_angle), 0.0])
            * fmomentum
        )
        self.Q = self.kf - self.ki
        # self.Q_perp = np.dot(self.Q,  self.v_rspace)
        # self.hkl = np.dot(inv_rubmat,  self.Q/(2*np.pi))
        self.Qmod = length(self.Q)
        if self.Qmod > abs(self.Q_par):
            self.Q_perp = math.sqrt(self.Qmod**2 - self.Q_par**2)
            self.dontproceed = False
        else:
            self.Q_perp = np.nan
            self.dontproceed = True
            return None

    #        # another new section
    #        Rhkl = 0.5/np.pi * np.dot(inv_ubmat, self.Q)
    #        if length(self.hkl) < 1e-14:
    #            self.dontproceed = True
    #            return None
    #        else:
    #            self.dontproceed = False
    #            temp_rotmat = rotation_matrix_from_vectors(self.hkl,  Rhkl)
    #            self.cart_rotmat = temp_rotmat
    #            # self.cart_rotmat = np.linalg.inv(rotation_matrix_from_vectors(Rhkl, self.hkl))
    #        rot_x = np.degrees(np.arctan2(temp_rotmat[2, 1], temp_rotmat[2, 2]))
    #        rot_y = np.degrees(np.arctan2(-temp_rotmat[2, 0], (temp_rotmat[2, 2]**2 + temp_rotmat[2, 1]**2)**0.5))
    #        rot_z = np.degrees(np.arctan2(temp_rotmat[1, 0], temp_rotmat[0, 0]))
    #        # print("Absolute rotations around x,y,z:",  rot_x, rot_y, rot_z)
    @pyqtSlot()
    def calc_Rmatrix(self):
        if self.dontproceed:
            return None
        xaxis = manip_axisx.copy().reshape((1, 3))
        yaxis = manip_axisy.copy().reshape((1, 3))
        zaxis = manip_axisz.copy().reshape((1, 3))
        normvec = holder_normal.copy().reshape((1, 3))
        vertvec = np.array([[0, 0, 1.0]])
        ang_offsets = tilt_values(abs(90 - self.arm_theta))
        # first the chamber tilt due to arm motion
        Cham_rot = fast_rotation(xaxis[0], ang_offsets)
        final_zaxis = fast_rotate(Cham_rot, zaxis)  # first axis is KNOWN
        final_yaxis = fast_rotate(self.cart_rotmat, yaxis)  # ROT axis is KNOWN
        final_xaxis = np.cross(final_zaxis, final_yaxis)  #
        #
        zdir = angle_v1v2(final_xaxis, np.array([1.0, 0, 0]))[0]
        #
        Z_rot = fast_rotation(final_zaxis[0], zdir)
        vt = fast_rotate(Z_rot, fast_rotate(Cham_rot, yaxis))
        incl = angle_v1v2(vt, final_yaxis[0])[0]
        #
        X_rot = fast_rotation(final_xaxis[0], incl)
        temp1 = fast_rotate(self.cart_rotmat, vertvec)
        temp2 = fast_rotate(X_rot, fast_rotate(Z_rot, fast_rotate(Cham_rot, vertvec)))
        rot = angle_v1v2(temp1, temp2[0])[0]
        self.motors_result.emit([self.arm_theta, zdir, incl, rot])

    def find_angles_constrained(self):
        if self.dontproceed:
            return None
        angs = [tilt_values(abs(90 - self.arm_theta)), 0.0, self.arm_theta / 2.0 - 10.0]
        rixsvals = [], {"arm_theta": [self.arm_theta], "Ei": self.Ei, "dE": self.dE}
        instvals = [], {
            "incl": [tilt_values(abs(90 - self.arm_theta))],
            "rot": [0],
            "zdir": [self.arm_theta / 2.0],
        }
        self.innerCK.parse_input([self.sample_vals, instvals, rixsvals])
        results1 = scopt.minimize(
            iter_tilted_angleseek_constrained,
            angs,
            args=(self.innerCK, -self.Q_par, self.Q_perp, self.inplane_vec),
            method="Nelder-Mead",
        )
        self.answer_angle = results1["x"]
        a, b, c = self.answer_angle[2], self.answer_angle[0], self.answer_angle[1]
        off1 = c - self.arm_theta / 2.0
        # second go
        angs = [tilt_values(abs(90 - self.arm_theta)), 0.0, self.arm_theta / 2.0 + 10.0]
        instvals = [], {
            "incl": [tilt_values(abs(90 - self.arm_theta))],
            "rot": [0],
            "zdir": [self.arm_theta / 2.0],
        }
        self.innerCK.parse_input([self.sample_vals, instvals, rixsvals])
        results2 = scopt.minimize(
            iter_tilted_angleseek_constrained,
            angs,
            args=(self.innerCK, self.Q_par, self.Q_perp, self.inplane_vec),
            method="Nelder-Mead",
        )
        self.answer_angle2 = results2["x"]
        d, e, f = self.answer_angle2[2], self.answer_angle2[0], self.answer_angle2[1]
        self.motors_result.emit([a, b, c, d, e, f])

    def find_angles_generalised(self):
        if self.dontproceed:
            return None
        angs = [tilt_values(abs(90 - self.arm_theta)), 0.0, self.arm_theta / 2.0 - 10.0]
        rixsvals = [], {"arm_theta": [self.arm_theta], "Ei": self.Ei, "dE": self.dE}
        instvals = [], {
            "incl": [tilt_values(abs(90 - self.arm_theta))],
            "rot": [0],
            "zdir": [self.arm_theta / 2.0],
        }
        self.innerCK.parse_input([self.sample_vals, instvals, rixsvals])
        results1 = scopt.minimize(
            iter_tilted_angleseek_constrained,
            angs,
            args=(self.innerCK, -self.Q_par, self.Q_perp, self.inplane_vec),
            method="Nelder-Mead",
        )
        self.answer_angle = results1["x"]
        # self.motors_result.emit([a, b, c, d, e, f])

    def find_angles(self):
        if self.dontproceed:
            return None
        if np.abs(self.inplane_vec).sum() > 1e-10:
            self.find_angles_constrained()
            return None
        else:
            return None
        angs = [tilt_values(abs(90 - self.arm_theta)), 0.0, self.arm_theta / 2.0]
        rixsvals = [], {"arm_theta": [self.arm_theta], "Ei": self.Ei, "dE": self.dE}
        instvals = [], {
            "incl": [tilt_values(abs(90 - self.arm_theta))],
            "rot": [0],
            "zdir": [self.arm_theta / 2.0],
        }
        self.innerCK.parse_input([self.sample_vals, instvals, rixsvals])
        results = scopt.minimize(
            iter_tilted_angleseek,
            angs,
            args=(self.innerCK, self.hkl),
            method="Nelder-Mead",
        )
        self.answer_angle = results["x"]
        self.motors_result.emit(
            [
                self.arm_theta,
                self.answer_angle[2],
                self.answer_angle[0],
                self.answer_angle[1],
            ]
        )

    def plot_vectors(self):
        """
        Each vector should be defined as two points, so that they can be plotted easily.
        This means a pair (start_point(3), end_point(3)), which can be drawn as an arrow.
        It will be up to the plotting function to choose the projection and to cast
        the vectors onto a 2D plane.
        """
        vdict = {}
        # vdict['k$_{i}$'] = (-self.ki, np.zeros(3))
        vdict["-k$_{i}$"] = (self.kf, self.kf - self.ki)
        vdict["k$_{f}$"] = (np.zeros(3), self.kf)
        vdict["u$_{ideal}$"] = (np.zeros(3), np.array([1, 0, 0]))
        vdict["v$_{ideal}$"] = (np.zeros(3), np.array([0, 1, 0]))
        vdict["Q"] = (np.zeros(3), self.Q)
        vdict["u$_{real}$"] = (np.zeros(3), self.u_rspace)
        vdict["v$_{real}$"] = (np.zeros(3), self.v_rspace)
        return vdict

    def verify_hkl(self, motor_dict):
        if self.dontproceed:
            return None
        angs = [tilt_values(abs(90 - self.arm_theta)), 0.0, self.arm_theta / 2.0]
        rixsvals = [], {"arm_theta": [self.arm_theta], "Ei": self.Ei}
        instvals = [], {
            "incl": [tilt_values(abs(90 - self.arm_theta))],
            "rot": [0],
            "zdir": [self.arm_theta / 2.0],
        }
        self.innerCK.parse_input([self.sample_vals, instvals, rixsvals])
        self.innerCK.new_angles(
            np.concatenate(
                [
                    motor_dict["incl1"],
                    motor_dict["rot1"],
                    motor_dict["zdir1"],
                ]
            )
        )
        hkl1, hkl_lims1 = self.innerCK.new_hkl()
        self.innerCK.new_angles(
            np.concatenate(
                [
                    motor_dict["incl2"],
                    motor_dict["rot2"],
                    motor_dict["zdir2"],
                ]
            )
        )
        hkl2, hkl_lims2 = self.innerCK.new_hkl()
        extras = [self.innerCK.Qmod, self.innerCK.Q_perp]
        self.hkl_result.emit(np.concatenate([hkl1, hkl2, extras]))
        self.hkl_limits.emit(
            [
                np.concatenate([hkl_lims1[0], hkl_lims2[0]]),
                np.concatenate([hkl_lims1[1], hkl_lims2[1]]),
            ]
        )


tabnames = [
    "CostFunction",
    "ArmTheta",
    "ZDIR",
    "INCL",
    "ROT",
    "H",
    "K",
    "L",
    "Q",
    "Q_par",
    "Q_perp",
    "theta",
    "Ei",
]


class FlexKernel(QObject):
    motors_result = pyqtSignal(object)
    other_results = pyqtSignal(object)
    hkl_result = pyqtSignal(object)
    hkl_limits = pyqtSignal(object)
    table_results = pyqtSignal(object)

    def __init__(self, parent):
        if parent is not None:
            super().__init__(parent)
        else:
            super().__init__(parent)
        self.Rmatrix = None
        self.ki = None
        self.kf = None
        self.Q = None
        self.hkl = None
        self.Ei = None
        self.dE = 0.0
        self.Qmod = None
        self.Q_perp = None
        self.Q_par = None
        self.u_rspace = None
        self.v_rspace = None
        self.manip_angles = None
        self.arm_theta = None
        self.vary_armtheta = True
        self.dontproceed = True
        self.inplane = None
        self.need_new_values = False
        self.innerCK = ComputeKernel(None)
        self.lock = QMutex()
        self.setlimits()

    def setlimits(self):
        self.limits = {}
        self.limits["Ei"] = (200.0, 1200.0)
        self.limits["zdir"] = (0.0, 180.0)
        self.limits["incl"] = (-24.0, 36.0)
        self.limits["rot"] = (-90.0, 135.0)
        self.limits["arm_theta"] = (33.0, 139.0)

    @pyqtSlot(object)
    def parse_input(self, all_inputs):
        sval, inval, inflags = all_inputs
        self.sample_vals = sval
        self.all_vals = {}
        self.all_flags = inflags
        vnames = inval[0]
        vdict = inval[1]
        fdict = inflags[1]
        self.latt_const = np.concatenate(
            [
                self.sample_vals[1]["a"],
                self.sample_vals[1]["b"],
                self.sample_vals[1]["c"],
            ]
        )
        self.latt_angs = np.concatenate(
            [
                self.sample_vals[1]["alpha"],
                self.sample_vals[1]["beta"],
                self.sample_vals[1]["gamma"],
            ]
        )
        # print(latt_const)
        # print(latt_angs)
        fixedpars, outputs = {}, {}
        results = {}
        for nn, nam in enumerate(vnames):
            if fdict[nn]:
                try:
                    vlen = len(vdict[nam])
                except:
                    fixedpars[nam] = vdict[nam]
                else:
                    if vlen == 1:
                        fixedpars[nam] = vdict[nam][0]
                    else:
                        fixedpars[nam] = vdict[nam]
            else:
                outputs[nam] = None
                results[nam] = np.nan
        self.fixedpars = fixedpars
        self.outputs = outputs
        self.vnames = vnames
        self.results = results
        self.fix_keys = [str(xx) for xx in self.fixedpars.keys()]
        self.res_keys = [str(xx) for xx in self.results.keys()]
        self.u, self.v = self.sample_vals[1]["u"], self.sample_vals[1]["v"]
        self.gonio_angles = self.sample_vals[1]["misalignment"]
        # self.arm_theta = self.rixs_vals[1]['arm_theta'][0]
        # self.scatt_angle = np.radians(self.rixs_vals[1]['arm_theta'])[0]
        self.innerCK.new_sample(
            self.latt_const, self.latt_angs, self.u, self.v, self.gonio_angles
        )
        self.some_guesses()
        self.lock.lock()
        self.need_new_values = True
        self.lock.unlock()

    def some_guesses(self):
        # ARM_THETA H K L THETA INPLANE_H INPLANE_K INPLANE_L Q_PAR Q_PERP
        # if a parameter is given, it means that it is fixed. The other parameters have to be derived from the known ones.
        try:
            Ei = self.fixedpars["Ei"]
        except KeyError:
            Ei = None
            kabs = None
        else:
            plambda = p_lambda(Ei)
            kabs = 2 * np.pi / plambda
        self.Ei = Ei
        # now the difficult decision tree
        try:
            arm_theta = self.fixedpars["arm_theta"]
        except KeyError:
            arm_theta = None
        else:
            self.arm_theta = arm_theta
        try:
            Qmod = self.fixedpars["Q"]
        except KeyError:
            Qmod = None
        if arm_theta is None:
            if Qmod is not None:
                if kabs is not None:
                    arm_theta = 180.0 - 2 * np.degrees(np.arccos(Qmod / 2 * kabs))
        # now arm_theta is known
        if arm_theta is None:
            self.vary_armtheta = True
        else:
            self.arm_theta = arm_theta
            self.vary_armtheta = False
        # self.manip_angles = angs
        # self.calc_Rmatrix()

    def oneflip(self, angs, armth, ener):
        self.innerCK.new_params(angs, armth, ener)
        hkl, hkl_lims = self.innerCK.new_hkl_fast()
        # hkl = self.innerCK.hkl
        if "lock" in self.fix_keys:
            self.inplane = self.innerCK.hkl_to_q(self.fixedpars["lock"])
        costfunction = 0
        if "theta" in self.fix_keys:
            costfunction += (self.fixedpars["theta"] - self.innerCK.theta) ** 2
        if "Q" in self.fix_keys:
            costfunction += (self.fixedpars["Q"] - self.innerCK.Qmod) ** 2
        if "Q_perp" in self.fix_keys:
            costfunction += (self.fixedpars["Q_perp"] - self.innerCK.Q_perp) ** 2
        if "Q_par" in self.fix_keys:
            costfunction += (self.fixedpars["Q_par"] - self.innerCK.Q_par) ** 2
        if "H" in self.fix_keys:
            costfunction += (self.fixedpars["H"] - hkl[0]) ** 2
        if "K" in self.fix_keys:
            costfunction += (self.fixedpars["K"] - hkl[1]) ** 2
        if "L" in self.fix_keys:
            costfunction += (self.fixedpars["L"] - hkl[2]) ** 2
        if "lock" in self.fix_keys:
            costfunction += abs(np.dot(self.inplane, [0, 0, 1])) ** 2
        return costfunction

    def output_results(self):
        theta = self.innerCK.theta
        Q = self.innerCK.Q
        Q_perp = self.innerCK.Q_perp
        hkl = self.innerCK.hkl
        Qmod = self.innerCK.Qmod
        Q_par = self.innerCK.Q_par
        if "Q" in self.res_keys:
            self.results["Q"] = Qmod
        if "Q_par" in self.res_keys:
            self.results["Q_par"] = Q_par
        if "Q_perp" in self.res_keys:
            self.results["Q_perp"] = Q_perp
        if "H" in self.res_keys:
            self.results["H"] = hkl[0]
        if "K" in self.res_keys:
            self.results["K"] = hkl[1]
        if "L" in self.res_keys:
            self.results["L"] = hkl[2]
        if "theta" in self.res_keys:
            self.results["theta"] = theta
        # if 'lock' in self.res_keys:
        #     inplane = self.innerCK.hkl_to_q(cvec)
        self.other_results.emit(self.results)

    @pyqtSlot()
    def start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.produce_values)
        self.timer.setInterval(1000)
        self.timer.start()

    @pyqtSlot()
    def produce_values(self):
        result = False
        self.lock.lock()
        result = self.need_new_values == True
        self.lock.unlock()
        if result:
            self.calc_Rmatrix()
            self.calc_hkl()
            self.lock.lock()
            self.need_new_values = False
            self.lock.unlock()

    @pyqtSlot()
    def find_angles(self):
        full_inputs = []
        costfuns = []
        if self.vary_armtheta:
            if self.Ei is None:
                bounds = [
                    self.limits["zdir"],
                    self.limits["incl"],
                    self.limits["rot"],
                    self.limits["arm_theta"],
                    self.limits["Ei"],
                ]
                results = scopt.shgo(
                    optimise_everything,
                    bounds,
                    args=(self,),
                    iters=5,
                    sampling_method="sobol",
                )
                print(results)
                mptr = results["x"][:3]
                armth = results["x"][3]
                Ei = results["x"][4]
                for nres, resargs in enumerate(results["xl"]):
                    full_inputs.append(
                        [resargs[0], resargs[1], resargs[2], resargs[3], resargs[4]]
                    )
                    costfuns.append(results["funl"][nres])
            else:
                bounds = [
                    self.limits["zdir"],
                    self.limits["incl"],
                    self.limits["rot"],
                    self.limits["arm_theta"],
                ]
                results = scopt.shgo(
                    optimise_mptr_arm_angles,
                    bounds,
                    args=(self,),
                    iters=5,
                    sampling_method="sobol",
                )
                print(results)
                mptr = results["x"][:3]
                armth = results["x"][3]
                Ei = self.Ei
                for nres, resargs in enumerate(results["xl"]):
                    full_inputs.append(
                        [resargs[0], resargs[1], resargs[2], resargs[3], Ei]
                    )
                    costfuns.append(results["funl"][nres])
        else:
            if self.Ei is None:
                bounds = [
                    self.limits["zdir"],
                    self.limits["incl"],
                    self.limits["rot"],
                    self.limits["Ei"],
                ]
                results = scopt.shgo(
                    optimise_Ei_mptr_angles,
                    bounds,
                    args=(self,),
                    iters=5,
                    sampling_method="sobol",
                )
                print(results)
                mptr = results["x"][:3]
                armth = self.arm_theta
                Ei = results["x"][3]
                for nres, resargs in enumerate(results["xl"]):
                    full_inputs.append(
                        [resargs[0], resargs[1], resargs[2], armth, resargs[3]]
                    )
                    costfuns.append(results["funl"][nres])
            else:
                bounds = [self.limits["zdir"], self.limits["incl"], self.limits["rot"]]
                results = scopt.shgo(
                    optimise_mptr_angles,
                    bounds,
                    args=(self,),
                    iters=5,
                    sampling_method="sobol",
                )
                print(results)
                mptr = results["x"][:3]
                armth = self.arm_theta
                Ei = self.Ei
                for nres, resargs in enumerate(results["xl"]):
                    full_inputs.append([resargs[0], resargs[1], resargs[2], armth, Ei])
                    costfuns.append(results["funl"][nres])
        cfun = results["fun"]
        if "Ei" in self.res_keys:
            self.results["Ei"] = Ei
        if "arm_theta" in self.res_keys:
            self.results["arm_theta"] = armth
        self.innerCK.new_params(mptr, armth, Ei)
        self.innerCK.new_hkl()
        self.motors_result.emit([mptr, cfun])
        self.output_results()
        ext_output = []
        for nin, alt_x in enumerate(full_inputs):
            resline = {}
            self.innerCK.new_params(alt_x[0:3], alt_x[3], alt_x[4])
            self.innerCK.new_hkl()
            resline["ZDIR"] = alt_x[0]
            resline["INCL"] = alt_x[1]
            resline["ROT"] = alt_x[2]
            resline["ArmTheta"] = alt_x[3]
            resline["Ei"] = alt_x[4]
            theta = self.innerCK.theta
            Q = self.innerCK.Q
            Q_perp = self.innerCK.Q_perp
            hkl = self.innerCK.hkl
            Qmod = self.innerCK.Qmod
            Q_par = self.innerCK.Q_par
            resline["Q"] = Qmod
            resline["Q_par"] = Q_par
            resline["Q_perp"] = Q_perp
            resline["H"] = hkl[0]
            resline["K"] = hkl[1]
            resline["L"] = hkl[2]
            resline["theta"] = theta
            resline["CostFunction"] = costfuns[nin]
            ext_output.append(resline)
        self.table_results.emit(ext_output)

    def plot_vectors(self):
        """
        Each vector should be defined as two points, so that they can be plotted easily.
        This means a pair (start_point(3), end_point(3)), which can be drawn as an arrow.
        It will be up to the plotting function to choose the projection and to cast
        the vectors onto a 2D plane.
        """
        vdict = {}
        # vdict['k$_{i}$'] = (-self.ki, np.zeros(3))
        vdict["-k$_{i}$"] = (self.kf, self.kf - self.ki)
        vdict["k$_{f}$"] = (np.zeros(3), self.kf)
        vdict["u$_{ideal}$"] = (np.zeros(3), np.array([1, 0, 0]))
        vdict["v$_{ideal}$"] = (np.zeros(3), np.array([0, 1, 0]))
        vdict["Q"] = (np.zeros(3), self.Q)
        vdict["u$_{real}$"] = (np.zeros(3), self.u_rspace)
        vdict["v$_{real}$"] = (np.zeros(3), self.v_rspace)
        return vdict

    def verify_hkl(self, motor_dict):
        if self.dontproceed:
            return None
        angs = [tilt_values(abs(90 - self.arm_theta)), 0.0, self.arm_theta / 2.0]
        rixsvals = [], {"arm_theta": [self.arm_theta], "Ei": self.Ei}
        instvals = [], {
            "incl": [tilt_values(abs(90 - self.arm_theta))],
            "rot": [0],
            "zdir": [self.arm_theta / 2.0],
        }
        self.innerCK.parse_input([self.sample_vals, instvals, rixsvals])
        self.innerCK.new_angles(
            np.concatenate(
                [
                    motor_dict["incl1"],
                    motor_dict["rot1"],
                    motor_dict["zdir1"],
                ]
            )
        )
        hkl1, hkl_lims1 = self.innerCK.new_hkl()
        self.innerCK.new_angles(
            np.concatenate(
                [
                    motor_dict["incl2"],
                    motor_dict["rot2"],
                    motor_dict["zdir2"],
                ]
            )
        )
        hkl2, hkl_lims2 = self.innerCK.new_hkl()
        extras = [self.innerCK.Qmod, self.innerCK.Q_perp]
        self.hkl_result.emit(np.concatenate([hkl1, hkl2, extras]))
        self.hkl_limits.emit(
            [
                np.concatenate([hkl_lims1[0], hkl_lims2[0]]),
                np.concatenate([hkl_lims1[1], hkl_lims2[1]]),
            ]
        )


def iter_tilted_angleseek_constrained(angs, CK, Qpar, Qperp, cvec):
    CK.new_angles(angs)
    newhkl, newhkl_lims = CK.new_hkl()
    inplane = CK.hkl_to_q(cvec)
    newQpar = CK.current_Q_par()
    newQperp = CK.current_Q_perp()
    # return ((newhkl-hkl)**2).sum() + abs(np.dot(inplane, [0, 0, 1]) + (Qpar-newQpar)**2)
    return (
        abs(np.dot(inplane, [0, 0, 1])) ** 2
        + (Qpar - newQpar) ** 2
        + (Qperp - newQperp) ** 2
    )
    # return  (Qpar-newQpar)**2


# variables:
# ARM_THETA H K L THETA INPLANE_H INPLANE_K INPLANE_L Q Q_PAR Q_PERP
# flags:
# weight_factor
# my axiom: the Ei is ALWAYS fixed! ZDIR, ROT, INCL, are ALWAYS variable!


def optimise_mptr_angles(angs, CK):
    ath = CK.arm_theta
    ei = CK.Ei
    return CK.oneflip(angs, ath, ei)


def optimise_mptr_arm_angles(angs, CK):
    ath = angs[3]
    ei = CK.Ei
    return CK.oneflip(angs[0:3], ath, ei)


def optimise_Ei_mptr_angles(angs, CK):
    ath = CK.arm_theta
    ei = angs[3]
    return CK.oneflip(angs[0:3], ath, ei)


def optimise_everything(angs, CK):
    ath = angs[3]
    ei = angs[4]
    return CK.oneflip(angs[0:3], ath, ei)
