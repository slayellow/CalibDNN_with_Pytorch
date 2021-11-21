import mathutils
import numpy as np


def convert_6DoF_to_RTMatrix(rotation, translation):
    R = mathutils.Euler((rotation[0], rotation[1], rotation[2]), 'XYZ')
    T = mathutils.Vector((translation[0], translation[1], translation[2]))

    R = R.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(T)

    RT = T @ R
    return np.array(RT)


def convert_RTMatrix_to_6DoF(matrix):
    RT = mathutils.Matrix(matrix)
    RT.invert_safe()
    T_GT, R_GT, _ = RT.decompose()
    R_GT_Euler = R_GT.to_euler()
    return np.array(R_GT_Euler), np.array(T_GT)

