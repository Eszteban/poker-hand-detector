import cv2
import matplotlib.pyplot as plt
import numpy as np
from src import process
from src.utils.Loader import Loader

card_path =  'test/20251108_105827.jpg'

# Eredeti kép megjelenítése
original_image = cv2.imread(card_path)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#plt.imshow(original_image_rgb)
#plt.show()
cv2.imshow('Eredeti kép', original_image_rgb)


#Kontúrok keresése
imgResult = original_image_rgb.copy()
imgResult2 = original_image_rgb.copy()

thresh = process.get_thresh(imgResult)

corners_list = process.find_corners_set(thresh, imgResult, draw=True)

four_corners_set = corners_list


#plt.imshow(thresh)
#plt.show()
cv2.imshow('Kontúrok', thresh)

for i, corners in enumerate(corners_list):
    top_left = corners[0][0]
    bottom_left = corners[1][0]
    bottom_right = corners[2][0]
    top_right = corners[3][0]
    
    print(f'top_left: {top_left}')
    print(f'bottom_left: {bottom_left}')
    print(f'bottom_right: {bottom_right}')
    print(f'top_right: {top_right}\n')


#Perspektívikus torzítás
flatten_card_set = process.find_flatten_cards(imgResult2, four_corners_set)

for img_output in flatten_card_set:
    print(img_output.shape)

    #plt.imshow(img_output)
    #plt.show()
    cv2.imshow('Kártya perspektívikus torzítása', img_output)

#Kártya sarkok kivágása
    cropped_images = process.get_corner_snip(flatten_card_set)
for i, pair in enumerate(cropped_images):
    for j, img in enumerate(pair):
        # cv2.imwrite(f'num{i*2+j}.jpg', img)
        plt.subplot(1, len(pair), j+1)
        plt.imshow(img, 'gray')

    cv2.imshow(f'Kártya sarkok kivágása {i}', img)

# Kontúrok keresése a kivágott sarkokon
ranksuit_list: list = list()


plt.figure(figsize=(12, 6))
for i, (img, original) in enumerate(cropped_images):

    drawable = img.copy()
    d2 = original.copy()

    contours, _ = cv2.findContours(drawable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnts_sort = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    cnts_sort = sorted(cnts_sort, key=lambda x: cv2.boundingRect(x)[1])
    
    for cnt in cnts_sort:
        print(f'contour sorts = {cv2.contourArea(cnt)}')

    cv2.drawContours(drawable, cnts_sort, -1, (0, 255, 0), 1)

    cv2.imwrite(f'{i}.jpg', drawable)
    plt.grid(True)
    plt.subplot(1, len(cropped_images), i+1)
    #plt.imshow(img)
    cv2.imshow(f'Kontúrok a kivágott sarkon {i}', drawable)

    ranksuit = list()

    for i, cnt in enumerate(cnts_sort):
        x, y, w, h = cv2.boundingRect(cnt)
        x2, y2 = x+w, y+h

        crop = d2[y:y2, x:x2]
        if(i == 0): # rank: 70, 125
            crop = cv2.resize(crop, (70, 125), 0, 0)
        else: # suit: 70, 100
            crop = cv2.resize(crop, (70, 100), 0, 0)
        # convert to bin image
        _, crop = cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        crop = cv2.bitwise_not(crop)

        # # reverse bin image
        # crop = 

        ranksuit.append(crop)

        # cv2.rectangle(d2, (x, y), (x2, y2), (0, 255, 0), 2)
        

    ranksuit_list.append(ranksuit)
        
#plt.show()
cv2.imshow('Kontúrok a kivágott sarkon', drawable)

# A szám és a szín "kinyerése"
black_img = np.zeros((120, 70))
plt.figure(figsize=(12, 6))
for i, ranksuit in enumerate(ranksuit_list):

    rank = black_img
    suit = black_img
    try:
        rank = ranksuit[0]
        suit = ranksuit[1]
    except:
        pass

    plt.subplot(len(ranksuit_list), 2, i*2+1)

    cv2.imwrite(f"{i}.jpg", suit)
    plt.imshow(rank, 'gray')
    plt.subplot(len(ranksuit_list), 2, i*2+2)
    plt.imshow(suit, 'gray')

cv2.imshow('Kártya szám és szín', black_img)


# Template matching - összehasonlítás
train_ranks = Loader.load_ranks('assets/imgs/ranks')
train_suits = Loader.load_suits('assets/imgs/suits')

print(train_ranks[0].img.shape)
print(train_suits[0].img.shape)

for i, rank in enumerate(train_ranks):
    plt.subplot(1, len(train_ranks), i +1)
    plt.axis('off')
    plt.imshow(rank.img, 'gray')

cv2.imshow('Kép betöltve', rank.img)

for i, suit in enumerate(train_suits):
    plt.subplot(1, len(train_suits), i +1)
    plt.axis('off')
    plt.imshow(suit.img, 'gray')

cv2.imshow('Kép betöltve', suit.img)

for it in ranksuit_list:
    try:
        rank = it[0]
        suit = it[1]
    except:
        continue
    rs = process.template_matching(rank, suit, train_ranks, train_suits, show_plt=True)
    print(rs)

cv2.waitKey(0)
cv2.destroyAllWindows()