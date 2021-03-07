import numpy as np
import cv2 as cv


def ransac_matching(input, mirrored, matching_fn,
                    box_size=50,
                    n_samples=400,
                    angle_set_range=range(0, 360, 6),
                    variance_threshold=1e-4,
                    ransac_threshold=0.0):
    h, w = input.shape
    image_center = (w // 2, h // 2)

    np.random.seed(1)

    angle_set = list(angle_set_range)
    angles_len = len(angle_set)
    rmags = np.zeros((angles_len,))
    vs = np.zeros((angles_len, 2))

    for i in range(len(angle_set)):
        angle = angle_set[i]
        rot_matrix = cv.getRotationMatrix2D(image_center, angle, 1.0)
        mirrored_rotated = cv.warpAffine(mirrored, rot_matrix, dsize=(w, h))

        A = np.zeros((2 * h - 1, 2 * w - 1))
        B = np.zeros((2 * h - 1, 2 * w - 1))

        box_r = np.random.uniform(size=(n_samples, 2))

        for box_index in range(n_samples):
            box_x0 = int(box_r[box_index, 0] * (w - box_size))
            box_y0 = int(box_r[box_index, 1] * (h - box_size))

            box = mirrored_rotated[
                  box_y0: box_y0 + box_size,
                  box_x0: box_x0 + box_size
            ]

            vrr = np.var(box / 255)
            if vrr > variance_threshold:
                loc, val = matching_fn(input, box)
                if val > 0.75:
                    row = max(min(loc[1] - box_y0 + h - 1, 2 * h - 2), 0)
                    col = max(min(loc[0] - box_x0 + w - 1, 2 * w - 2), 0)

                    A[row, col] += 1
                    B[row, col] += val

        _, max_val, _, max_loc = cv.minMaxLoc(A)

        rmags[i] = max_val  # B[max_loc[1], max_loc[0]] * max_val
        vs[i] = [max_loc[0] - w + 1, max_loc[1] - h + 1]

    sorted_iangles = rmags.argsort()

    conf = rmags[sorted_iangles[-1]]
    if conf < ransac_threshold:
        return None

    iangle = sorted_iangles[-1]
    v = vs[iangle]
    arad = -angle_set[iangle] * np.pi / 180

    tx = 0.5 * (w * (1 - np.cos(arad)) + h * np.sin(arad)) + v[0]
    ty = 0.5 * (h * (1 - np.cos(arad)) - w * np.sin(arad)) + v[1]

    T = np.array([
        [np.cos(arad), -np.sin(arad), tx],
        [np.sin(arad), np.cos(arad), ty],
        [0, 0, 1]]
    )

    return T
