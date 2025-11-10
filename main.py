import cv2
import matplotlib.pyplot as plt
import numpy as np

from src import process
from src.utils.Loader import Loader
from src.utils import display
from src.model import Card, CardPack

card_path =  'test/20251109_164730.jpg'

# Eredeti kép megjelenítése
original_image = cv2.imread(card_path)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
plt.figure(num="Eredeti kép")
plt.imshow(original_image_rgb)
plt.show()

#Kontúrok keresése
imgResult = original_image_rgb.copy()
imgResult2 = original_image_rgb.copy()

thresh = process.get_thresh(imgResult)
corners_list = process.find_corners_set(thresh, imgResult, draw=True)
four_corners_set = corners_list

plt.figure(num="Élek detektálása")
plt.imshow(thresh, cmap='gray')
plt.show()

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

# Több kép egy ablakban (egy sorban)
fig, axes = plt.subplots(1, len(flatten_card_set), figsize=(4*len(flatten_card_set), 6))
if len(flatten_card_set) == 1:
    axes = [axes]
for i, img_output in enumerate(flatten_card_set):
    print(img_output.shape)
    axes[i].imshow(img_output)           # színes képekhez
    axes[i].axis('off')
plt.tight_layout()
plt.show()

cropped_images = process.get_corner_snip(flatten_card_set)

#Kártya sarkok kivágása    
for i, pair in enumerate(cropped_images):
    for j, img in enumerate(pair):
        # cv2.imwrite(f'num{i*2+j}.jpg', img)
        
        plt.subplot(1, len(pair), j+1)
        plt.imshow(img, 'gray')

    plt.show()

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
    plt.grid(True)
    plt.subplot(1, len(cropped_images), i+1)
    plt.imshow(img, 'gray')
    ranksuit = list()

    for i, cnt in enumerate(cnts_sort):
        x, y, w, h = cv2.boundingRect(cnt)
        x2, y2 = x+w, y+h
        crop = d2[y:y2, x:x2]
        if(i == 0): # rank: 70, 125
            crop = cv2.resize(crop, (70, 125), 0, 0)
        else: # suit: 70, 100
            crop = cv2.resize(crop, (70, 100), 0, 0)
        # Binárizálás
        _, crop = cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        crop = cv2.bitwise_not(crop)
        ranksuit.append(crop)

    ranksuit_list.append(ranksuit)
        
plt.show()

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
    plt.imshow(rank, 'gray')
    plt.subplot(len(ranksuit_list), 2, i*2+2)
    plt.imshow(suit, 'gray')

plt.show()

# Template matching - összehasonlítás
train_ranks = Loader.load_ranks('assets/imgs/ranks')
train_suits = Loader.load_suits('assets/imgs/suits')

print(train_ranks[0].img.shape)
print(train_suits[0].img.shape)

cardPack = CardPack()

for it in ranksuit_list:
    try:
        rank = it[0]
        suit = it[1]
    except:
        continue
    r, s = process.template_matching(rank, suit, train_ranks, train_suits, show_plt=True)
    print(f'Predicted rank: {r}, suit: {s}\n')
    cardPack.addCard(Card(r, s))
    
print("Hand:", cardPack.cards)
print("Result:", cardPack.checkHand())

# Annotált kép készítése (eredeti BGR kell a putText-hez)

# Szövegek listája (pl. 'AH', '7D', stb.)
pred_texts = [f"{i.rank} {i.suit}" for i in cardPack.cards]

# Saroklista (four_corners_set) már korábban elkészült
process.show_text(pred_texts, four_corners_set, original_image)

# Megjelenítés matplotlib-ben (RGB konverzió)
plt.figure(num="Felismert kártyák")
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(str(cardPack.cards) + " -> " + str(cardPack.checkHand()))
plt.show()
