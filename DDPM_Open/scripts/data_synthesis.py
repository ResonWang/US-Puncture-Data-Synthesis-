import cv2
import numpy as np

def PossionEdit(src_img, dst_img, center):
    """
    :param src_img: 被融合的图像
    :param dst_img: 针尖图像
    :param center: （列，行）
    :return:
    """
    mask = 255 * np.ones(src_img.shape, src_img.dtype)
    output = cv2.seamlessClone(src_img, dst_img, mask, center, cv2.NORMAL_CLONE)
    return output


mean_height = 20
std_dev_height = 3
mean_width = 36
std_dev_width = 5

def Synthesize_one(US_image, Tip_image):

    new_height = int(np.random.normal(mean_height, std_dev_height))
    new_width = int(np.random.normal(mean_width, std_dev_width))
    new_height = max(1, new_height)
    new_width = max(1, new_width)
    resized_tip = cv2.resize(Tip_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    center_col = np.random.randint(int(US_image.shape[1]*0.2), US_image.shape[1] - new_width - int(US_image.shape[1]*0.2))
    center_row = np.random.randint(int(US_image.shape[0]*0.2), US_image.shape[0] - new_height - int(US_image.shape[0]*0.2))

    fused_img = PossionEdit(resized_tip, US_image, (center_col, center_row))

    return fused_img



