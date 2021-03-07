import cv2 as cv
import numpy as np


def draw_matlab_segments(img, segments):
    for s in segments:
        cv.line(img,
                (s[0], s[1]), (s[2], s[3]),
                (0, 0, 255), 2)


def reflect_image_x(img: np.array):
    h, w = img.shape
    normal = np.array([[1], [0]])
    d = int((w / 2) * normal[0] + (h / 2) * normal[1])
    symmetry = (np.eye(2) - 2 * normal @ normal.T)
    reflected_img = cv.flip(img, flipCode=1)
    return reflected_img, symmetry, normal, d


def show_points_with_lines(input, mirrored_rotated, points_original, points_matched, vrr,
                           bbox_size=50,
                           n_bboxes=3,
                           window_name='points and lines'):
    cv.namedWindow(window_name, cv.WINDOW_GUI_NORMAL)
    cv.resizeWindow(window_name, 4 * input.shape[1], 2 * input.shape[0])
    input_draw = cv.cvtColor(input.copy(), cv.COLOR_GRAY2BGR)
    mirrored_draw = cv.cvtColor(mirrored_rotated.copy(), cv.COLOR_GRAY2BGR)

    np.set_printoptions(precision=2)
    for i, p in enumerate(points_original):
        cv.circle(input_draw, p, 1, (255, 0, 0))
        if i < n_bboxes:
            cv.rectangle(input_draw, p, (p[0] + bbox_size, p[1] + bbox_size), (255, 0, 0))
            cv.putText(input_draw, str(vrr[i] * 1e5), p, cv.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 255, 255))

    for i, pj in enumerate(points_matched):
        cv.circle(mirrored_draw, pj, 1, (0, 255, 0))
        if i < n_bboxes:
            cv.rectangle(mirrored_draw, pj, (pj[0] + bbox_size, pj[1] + bbox_size), (0, 255, 0))

    ulmat = np.hstack([input_draw, mirrored_draw])
    for (p, pj) in zip(points_original, points_matched):
        cv.line(ulmat, p, (pj[0] + input.shape[1], pj[1]), (255, 255, 0))

    cv.imshow(window_name, ulmat)
    cv.waitKey(-1)


def gradient(img):
    """
    Compute image gradient.

    Parameters
    ----------
    img
        Image.

    Returns
    -------
    grad
        Image gradient.
    """
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1)

    grad = 0.5 * (np.abs(sobelx) + np.abs(sobely))
    grad = np.clip(grad, 0, 1)
    return grad
