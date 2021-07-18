import numpy as np
import pycocotools.mask as mask_util

# TODO: move this function to more proper place
def encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code.
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    if isinstance(mask_results, tuple):  # mask scoring
        cls_segms, cls_mask_scores = mask_results
    else:
        cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = [[] for _ in range(num_classes)]
    for i in range(len(cls_segms)):
        for cls_segm in cls_segms[i]:
            encoded_mask_results[i].append(
                mask_util.encode(
                    np.array(
                        cls_segm[:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    if isinstance(mask_results, tuple):
        return encoded_mask_results, cls_mask_scores
    else:
        return encoded_mask_results