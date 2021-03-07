import cv2 as cv


def tm_ccorr_matching(image, template, threshold=0.3):
    """
    Template matching with normalized cross-correlation.

    Parameters
    ----------
    image
        Input image.
    template
        Input template.
    threshold
        Max value threshold.

    Returns
    -------
    Location of template or None if there is no matching (depends on threshold).

    """
    result = cv.matchTemplate(image, template, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    return max_loc if max_val > threshold else None


def tm_ccoeff_matching(image, template, threshold=0.75):
    """
    Template matching with normalized correlation coefficient.

    Parameters
    ----------
    image
        Input image.
    template
        Input template.
    threshold
        Max value threshold.

    Returns
    -------
    Location of template or None if there is no matching (depends on threshold).

    """
    result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(result)
    return max_loc, max_val


def tm_sqdiff_matching(image, template, threshold=0.02):
    """
    Template matching with normalized squared difference.

    Parameters
    ----------
    image
        Input image.
    template
        Input template.
    threshold
        Min value threshold.

    Returns
    -------
    Location of template or None if there is no matching (depends on threshold).

    """
    result = cv.matchTemplate(image, template, cv.TM_SQDIFF_NORMED)
    min_val, _, min_loc, _ = cv.minMaxLoc(result)
    return min_loc if min_val < threshold else None
