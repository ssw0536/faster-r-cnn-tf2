import cv2
import numpy as np


def draw_box(img, box, box_label=None, color=[255, 0, 0], thickness=2):
    # copy image
    img_vis = img.copy()

    # un-normalize the image
    if img_vis.dtype != np.uint8:
        img_vis *= 255
        img_vis = img_vis.astype('uint8')

    # upack box
    ymin = box[0]
    xmin = box[1]
    ymax = box[2]
    xmax = box[3]

    # draw box on the image
    cv2.rectangle(
        img_vis,
        pt1=(int(xmin), int(ymin)),
        pt2=(int(xmax), int(ymax)),
        color=color,
        thickness=thickness,
        )

    if box_label is not None:
        cv2.rectangle(
            img_vis,
            pt1=(int(xmin), int(ymin-20)),
            pt2=(int(xmin+100), int(ymin)),
            color=color,
            thickness=-1,)
        cv2.putText(
            img_vis,
            box_label,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            org=(int(xmin+5), int(ymin-5)),
            color=[255, 255, 255],
            thickness=2,)

    return img_vis
