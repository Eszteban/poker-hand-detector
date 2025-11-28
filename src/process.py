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
def find_corners_set(img, original, draw=False, debug=False):
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

            if debug:
                print(f"Kontúr terület: {area:.0f}, kerület: {perimeter:.0f}, sarkok száma: {num_corners}")

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

                # Debug: sarokpontok vizualizálása és megállás
                if debug:
                    debug_img = original.copy()
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # TL, BL, BR, TR
                    labels = ['TL', 'BL', 'BR', 'TR']
                    
                    for i, corner in enumerate(finalOrder):
                        cx, cy = corner[0]
                        cv2.circle(debug_img, (cx, cy), 10, colors[i], -1)
                        cv2.putText(debug_img, labels[i], (cx + 15, cy), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
                        print(f"  {labels[i]}: ({cx}, {cy})")
                    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
                    plt.title(f"Sarokpontok - Kártya #{len(four_corners_set)}")
                    plt.axis('off')
                    plt.show()  # Ez megállítja a kódot, amíg be nem zárod az ablakot

    if debug:
        print(f"\nÖsszesen {len(four_corners_set)} kártya található")

    return four_corners_set

#used
def reorder_card_corners(corners):
    """
    corners: [ [ [x,y] ], ... ] formátum (a mostani listád)
    Visszaad: [top_left, bottom_left, bottom_right, top_right]
    """
    pts = np.array([c[0] for c in corners], dtype=np.float32)  # (4,2)
    
    # Középpont kiszámítása
    center = pts.mean(axis=0)
    
    # Sarkok osztályozása a középponthoz képest
    top_pts = []
    bottom_pts = []
    
    for pt in pts:
        if pt[1] < center[1]:  # y kisebb = feljebb van
            top_pts.append(pt)
        else:
            bottom_pts.append(pt)
    
    # Ha nincs pontosan 2-2 pont fent és lent, használjuk az y szerinti rendezést
    if len(top_pts) != 2 or len(bottom_pts) != 2:
        sorted_by_y = pts[np.argsort(pts[:, 1])]
        top_pts = sorted_by_y[:2]
        bottom_pts = sorted_by_y[2:]
    else:
        top_pts = np.array(top_pts)
        bottom_pts = np.array(bottom_pts)
    
    # Felső pontok: bal és jobb (x szerint)
    if top_pts[0][0] < top_pts[1][0]:
        tl, tr = top_pts[0], top_pts[1]
    else:
        tl, tr = top_pts[1], top_pts[0]
    
    # Alsó pontok: bal és jobb (x szerint)
    if bottom_pts[0][0] < bottom_pts[1][0]:
        bl, br = bottom_pts[0], bottom_pts[1]
    else:
        bl, br = bottom_pts[1], bottom_pts[0]
    
    return [tl, bl, br, tr]


#Used
#Used
def find_flatten_cards(img, set_of_corners, debug=False):
    width, height = 200, 300
    img_outputs = []

    for idx, corners in enumerate(set_of_corners):
        # Validáció hozzáadása
        if len(corners) != 4:
            if debug:
                print(f"Skipping: expected 4 corners, got {len(corners)}")
            continue
        
        if debug:
            print(f"\n=== Kártya #{idx + 1} ===")
            print(f"Eredeti sarkok: {corners}")
            
        # Sarokok robusztus újrarendezése
        tl, bl, br, tr = reorder_card_corners(corners)

        if debug:
            print(f"Újrarendezett sarkok:")
            print(f"  TL: {tl}")
            print(f"  BL: {bl}")
            print(f"  BR: {br}")
            print(f"  TR: {tr}")

        # Él hosszok
        vertical_left = DistanceHelper.euclidean(tl[0], tl[1], bl[0], bl[1])
        horizontal_top = DistanceHelper.euclidean(tl[0], tl[1], tr[0], tr[1])

        if debug:
            print(f"Élek: vertical={vertical_left:.1f}, horizontal={horizontal_top:.1f}")
            print(f"Fekvő kártya? {horizontal_top > vertical_left}")

        pts1 = np.float32([tl, bl, br, tr])
        pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        if debug:
            print(f"pts1 (forrás): {pts1}")
            print(f"pts2 (cél): {pts2}")

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_output = cv2.warpPerspective(img, matrix, (width, height))

        # Debug: transzformáció előtt/után megjelenítés
        if debug:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Eredeti kép a sarokpontokkal
            debug_img = img.copy()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            labels = ['TL', 'BL', 'BR', 'TR']
            points = [tl, bl, br, tr]
            for i, (pt, color, label) in enumerate(zip(points, colors, labels)):
                cx, cy = int(pt[0]), int(pt[1])
                cv2.circle(debug_img, (cx, cy), 15, color, -1)
                cv2.putText(debug_img, label, (cx + 20, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            # Vonalak a sarkok között
            pts_int = pts1.astype(int)
            cv2.polylines(debug_img, [pts_int], True, (0, 255, 255), 3)
            
            axes[0].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"Eredeti - Kártya #{idx + 1}")
            axes[0].axis('off')
            
            # Kivágott kép (forgatás előtt)
            axes[1].imshow(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"Kivágva (forgatás előtt)\nv={vertical_left:.0f}, h={horizontal_top:.0f}")
            axes[1].axis('off')

        # Forgatás után resize, hogy konzisztens méret legyen
        if horizontal_top > vertical_left:
            img_output = cv2.rotate(img_output, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_output = cv2.resize(img_output, (width, height))

        if debug:
            # Kivágott kép (forgatás után)
            axes[2].imshow(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB))
            axes[2].set_title("Végeredmény (forgatás után)" if horizontal_top > vertical_left else "Végeredmény (nincs forgatás)")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()

        img_outputs.append(img_output)

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
    best_rank_img = None
    best_suit_img = None

    for train_rank in train_ranks:
        diff_img = cv2.absdiff(rank, train_rank.img)
        rank_diff = int(np.sum(diff_img) / 255)

        if rank_diff < best_rank_match_diff:
            best_rank_match_diff = rank_diff
            best_rank_match_name = train_rank.name
            best_rank_img = train_rank.img

    for train_suit in train_suits:
        diff_img = cv2.absdiff(suit, train_suit.img)
        suit_diff = int(np.sum(diff_img) / 255)

        if suit_diff < best_suit_match_diff:
            best_suit_match_diff = suit_diff
            best_suit_match_name = train_suit.name
            best_suit_img = train_suit.img

    # Debug: template matching vizualizáció
    if show_plt and best_rank_img is not None and best_suit_img is not None:
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        
        # Rank sor
        axes[0, 0].imshow(rank, cmap='gray')
        axes[0, 0].set_title("Input Rank")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(best_rank_img, cmap='gray')
        axes[0, 1].set_title(f"Best Match: {best_rank_match_name}")
        axes[0, 1].axis('off')
        
        rank_diff_img = cv2.absdiff(rank, best_rank_img)
        axes[0, 2].imshow(rank_diff_img, cmap='hot')
        axes[0, 2].set_title(f"Diff: {best_rank_match_diff}")
        axes[0, 2].axis('off')
        
        # Suit sor
        axes[1, 0].imshow(suit, cmap='gray')
        axes[1, 0].set_title("Input Suit")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(best_suit_img, cmap='gray')
        axes[1, 1].set_title(f"Best Match: {best_suit_match_name}")
        axes[1, 1].axis('off')
        
        suit_diff_img = cv2.absdiff(suit, best_suit_img)
        axes[1, 2].imshow(suit_diff_img, cmap='hot')
        axes[1, 2].set_title(f"Diff: {best_suit_match_diff}")
        axes[1, 2].axis('off')
        
        plt.suptitle(f"Template Matching: {best_rank_match_name} of {best_suit_match_name}")
        plt.tight_layout()
        plt.show()

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