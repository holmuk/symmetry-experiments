import cv2 as cv
import numpy as np


def estimate_affine_transform(k1points, k2points, params: dict):
    if len(k1points) < 2:
        return None

    M, _ = cv.estimateAffinePartial2D(k1points, k2points, **params)
    if M is None:
        return None

    R = M[:2, :2]
    dR = np.linalg.det(R)

    if np.isclose(dR, 0.0):
        dR = 1
    R = R / dR

    M[:2, :2] = R

    return M


def estimate_reflection_naive(k1points, k2points):
    median_points = 0.5 * (k1points + k2points)
    if len(median_points) < 2:
        return None, None, None

    line = cv.fitLine(median_points, cv.DIST_WELSCH, 0, 0.2, 0.01)
    v = np.array([line[0][0], line[1][0]])
    p = np.array([line[2][0], line[3][0]])

    return p, v, median_points


def estimate_reflection_from_transform(symmetry_matrix,
                                       transform_matrix,
                                       w_offset,
                                       normal_vector,
                                       eigenvalue_threshold=0.01):
    R = transform_matrix[0:2, 0:2]
    eigenvals, eigenvectors = np.linalg.eig(symmetry_matrix @ R.T)

    indx = None
    if abs(eigenvals[0] + 1.0) <= eigenvalue_threshold:
        indx = 0
    if abs(eigenvals[1] + 1.0) <= eigenvalue_threshold:
        indx = 1

    if indx is None:
        return None, None

    eigvec = np.real(eigenvectors[:, indx])

    v = np.reshape([-eigvec[1], eigvec[0]], (2, 1))
    p = 0.5 * ((R @ (2 * w_offset * normal_vector)) + transform_matrix[:2, 2].reshape((2, 1)))

    return p, v


def normal_center_from_matlab_point(point):
    vx = point[0][0] - point[1][0]
    vy = point[0][1] - point[1][1]
    c = 0.5 * (point[0] + point[1])
    dv = np.sqrt(vx ** 2 + vy ** 2)
    v = vx / dv, vy / dv
    return v, dv, c


def normal_of_segment(segment):
    vx = segment[0] - segment[2]
    vy = segment[1] - segment[3]
    cx = (segment[0] + segment[2]) / 2
    cy = (segment[1] + segment[3]) / 2
    dv = np.sqrt(vx ** 2 + vy ** 2)
    v = (vx / dv, vy / dv)
    c = (cx, cy)

    return v, dv, c


def angles_equal(v1, v2, eps=0.97):
    return (v1[0] * v2[0] + v1[1] * v2[1]) ** 2 >= eps
