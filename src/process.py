import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils.ColorHelper import ColorHelper
from src.utils.DistanceHelper import DistanceHelper
from src.utils import constants


#Used
def get_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bin = ColorHelper.gray2bin(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 2.0)
    canny_threshold_max = find_max_gradient_value(blur)
    canny_threshold_tb_1 = int(canny_threshold_max * 0.1)
    canny_threshold_tb_2 = int(canny_threshold_max * 0.2)
    canny = do_canny_threshold(blur, canny_threshold_tb_1, canny_threshold_tb_2)
    kernel = np.ones((2, 2))
    dial = cv2.dilate(canny, kernel=kernel, iterations=2)
    

    return dial

#Used
def find_corners_set(img, original, draw=False):
    # find the set of contours on the threshed image
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # sort them by highest area
    proper = sorted(contours, key=cv2.contourArea, reverse=True)

    four_corners_set = []

    for cnt in proper:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)

        # only select those contours with a good area
        if area > 10000:
            # find out the number of corners
            approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
            num_corners = len(approx)

            if num_corners == 4:
                # create bounding box around shape
                x, y, w, h = cv2.boundingRect(approx)

                if draw:
                    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # make sure the image is oriented right: top left, bot left, bot right, top right
                l1 = np.array(approx[0]).tolist()
                l2 = np.array(approx[1]).tolist()
                l3 = np.array(approx[2]).tolist()
                l4 = np.array(approx[3]).tolist()

                finalOrder = []

                # sort by X vlaue
                sortedX = sorted([l1, l2, l3, l4], key=lambda x: x[0][0])

                # sortedX[0] and sortedX[1] are the left half
                finalOrder.extend(sorted(sortedX[0:2], key=lambda x: x[0][1]))

                # now sortedX[1] and sortedX[2] are the right half
                # the one with the larger y value goes first
                finalOrder.extend(sorted(sortedX[2:4], key=lambda x: x[0][1], reverse=True))

                four_corners_set.append(finalOrder)

                if draw:
                    for a in approx:
                        cv2.circle(original, (a[0][0], a[0][1]), 6, (255, 0, 0), 3)

    return four_corners_set

#used
def reorder_card_corners(corners):
    """
    corners: [ [ [x,y] ], ... ] formátum (a mostani listád)
    Visszaad: [top_left, bottom_left, bottom_right, top_right]
    """
    pts = np.array([c[0] for c in corners], dtype=np.float32)  # (4,2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]

    # Kívánt sorrend: TL, BL, BR, TR
    return [tl, bl, br, tr]

#Used
def find_flatten_cards(img, set_of_corners, debug=False):
    width, height = 200, 300
    img_outputs = []

    for corners in set_of_corners:
        # Sarokok robusztus újrarendezése
        tl, bl, br, tr = reorder_card_corners(corners)

        # Él hosszok (magasság vs szélesség ellenőrzéshez)
        vertical_left = DistanceHelper.euclidean(tl[0], tl[1], bl[0], bl[1])
        horizontal_top = DistanceHelper.euclidean(tl[0], tl[1], tr[0], tr[1])

        pts1 = np.float32([tl, bl, br, tr])
        pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_output = cv2.warpPerspective(img, matrix, (width, height))

        # Ha a "felső" él hosszabb, akkor a kártya feküdt: forgassuk vissza álló formára
        if horizontal_top > vertical_left:
            img_output = cv2.rotate(img_output, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img_outputs.append(img_output)

        if debug:
            print(f'v={vertical_left:.1f}, h={horizontal_top:.1f}')

    return img_outputs

#Used
def get_corner_snip(flattened_images: list):
    corner_images = []
    for img in flattened_images:
        # crop the image to where the corner might be
        # vertical, horizontal
        crop = img[5:110, 1:38]

        # resize by a factor of 4
        crop = cv2.resize(crop, None, fx=4, fy=4)

        # threshold the corner
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bin_img = ColorHelper.gray2bin(gray)
        bilateral = cv2.bilateralFilter(bin_img, 11, 174, 17)
        canny = cv2.Canny(bilateral, 40, 24)
        kernel = np.ones((1, 1))
        result = cv2.dilate(canny, kernel=kernel, iterations=2)

        # append the thresholded image and the original one
        corner_images.append([result, bin_img])

    return corner_images




#Used
def template_matching(rank, suit, train_ranks, train_suits, show_plt=False) -> tuple[str, str]:
    """Finds best rank and suit matches for the query card. Differences
    the query card rank and suit images with the train rank and suit images.
    The best match is the rank or suit image that has the least difference."""

    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"

    for train_rank in train_ranks:

        diff_img = cv2.absdiff(rank, train_rank.img)

        rank_diff = int(np.sum(diff_img) / 255)

        if rank_diff < best_rank_match_diff:
            best_rank_match_diff = rank_diff
            best_rank_name = train_rank.name

            if show_plt:
                print(f'diff score: {rank_diff}')
                #plt.subplot(1, 2, 1)
                #plt.imshow(diff_img, 'gray')

        #plt.show()

    # Same processing with suit images
    for train_suit in train_suits:

        diff_img = cv2.absdiff(suit, train_suit.img)
        suit_diff = int(np.sum(diff_img) / 255)

        if suit_diff < best_suit_match_diff:
            best_suit_match_diff = suit_diff
            best_suit_name = train_suit.name

            if show_plt:
                print(f'diff score: {suit_diff}')
                #plt.subplot(1, 2, 2)
                #plt.imshow(diff_img, 'gray')

        #plt.show()

    if best_rank_match_diff < 2300:
        best_rank_match_name = best_rank_name

    if best_suit_match_diff < 1000:
        best_suit_match_name = best_suit_name

    #plt.show()

    return best_rank_match_name, best_suit_match_name

def show_text(predictions: list[str], four_corners_set, img):
    for i, prediction in enumerate(predictions):
        # figure out where to place the text
        corners = np.array(four_corners_set[i])
        corners_flat = corners.reshape(-1, corners.shape[-1])
        start_x = corners_flat[0][0] + 0
        half_y = corners_flat[0][1] - 40

        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(img, prediction, (start_x, half_y), font, 2, (50, 205, 50), 2, cv2.LINE_AA)

def find_max_gradient_value(img_in):
    canny_sobel_kernel_size = constants.CANNY_SOBEL_KERNEL_SIZE

    im_dx = cv2.Sobel(img_in, cv2.CV_32FC1, 1, 0, None, canny_sobel_kernel_size)
    im_dy = cv2.Sobel(img_in, cv2.CV_32FC1, 0, 1, None, canny_sobel_kernel_size)
    im_gradient_magnitude = cv2.magnitude(im_dx, im_dy)
    return int(np.amax(im_gradient_magnitude)) + 1

def do_canny_threshold(im_blur, canny_threshold_tb_1, canny_threshold_tb_2):
    
    canny_threshold_1, canny_threshold_2 = get_canny_threshold_values(canny_threshold_tb_1, canny_threshold_tb_2)

    im_edges = cv2.Canny(im_blur, canny_threshold_1, canny_threshold_2, None, 3, True)
    return im_edges

def get_canny_threshold_values(canny_threshold_tb_1, canny_threshold_tb_2):

    if canny_threshold_tb_1 < canny_threshold_tb_2:
        threshold_1, threshold_2 = canny_threshold_tb_1, canny_threshold_tb_2
    else:
        threshold_1, threshold_2 = canny_threshold_tb_2, canny_threshold_tb_1

    return threshold_1, threshold_2  # Values in lower, higher order