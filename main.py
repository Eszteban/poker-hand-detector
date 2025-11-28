import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

from src import process
from src.utils.Loader import Loader
from src.model import Card, CardPack


class PokerDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Poker Kártya Detektor")
        self.root.geometry("800x650")
        
        self.image_path = None
        self.original_image = None
        
        self._create_widgets()
        
    def _create_widgets(self):
        # Fő keret
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Felső rész: Kép betöltés ===
        load_frame = ttk.LabelFrame(main_frame, text="Kép betöltése", padding="10")
        load_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.path_var = tk.StringVar(value="Nincs kép kiválasztva")
        path_label = ttk.Label(load_frame, textvariable=self.path_var, wraplength=600)
        path_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        load_btn = ttk.Button(load_frame, text="Kép tallózása...", command=self._load_image)
        load_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # === Középső rész: Beállítások ===
        settings_frame = ttk.LabelFrame(main_frame, text="Debug beállítások", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Debug pipák - két oszlopban
        left_col = ttk.Frame(settings_frame)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        right_col = ttk.Frame(settings_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Bal oszlop
        self.debug_corners_var = tk.BooleanVar(value=False)
        self.debug_flatten_var = tk.BooleanVar(value=False)
        
        corners_check = ttk.Checkbutton(
            left_col, 
            text="Sarokpont detektálás", 
            variable=self.debug_corners_var
        )
        corners_check.pack(anchor=tk.W)
        
        flatten_check = ttk.Checkbutton(
            left_col, 
            text="Perspektívikus kivágás", 
            variable=self.debug_flatten_var
        )
        flatten_check.pack(anchor=tk.W)
        
        # Jobb oszlop
        # Jobb oszlop
        self.debug_cropped_var = tk.BooleanVar(value=False)
        self.debug_template_var = tk.BooleanVar(value=False)
        self.debug_result_var = tk.BooleanVar(value=True)
        self.debug_thresh_var = tk.BooleanVar(value=False)
        
        cropped_check = ttk.Checkbutton(
            right_col, 
            text="Kivágott sarkok (rank/suit)", 
            variable=self.debug_cropped_var
        )
        cropped_check.pack(anchor=tk.W)
        
        template_check = ttk.Checkbutton(
            right_col, 
            text="Template matching", 
            variable=self.debug_template_var
        )
        template_check.pack(anchor=tk.W)
        
        thresh_check = ttk.Checkbutton(
            right_col, 
            text="Élkép (Canny/threshold)", 
            variable=self.debug_thresh_var
        )
        thresh_check.pack(anchor=tk.W)
        
        result_check = ttk.Checkbutton(
            left_col, 
            text="Végeredmény megjelenítése", 
            variable=self.debug_result_var
        )
        result_check.pack(anchor=tk.W)
        
        # === Kép előnézet ===
        preview_frame = ttk.LabelFrame(main_frame, text="Előnézet", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.preview_label = ttk.Label(preview_frame, text="Tölts be egy képet!", anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # === Alsó rész: Futtatás ===
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        run_btn = ttk.Button(
            button_frame, 
            text="🎴 Felismerés indítása", 
            command=self._run_detection,
            style="Accent.TButton"
        )
        run_btn.pack(side=tk.RIGHT)
        
        # Eredmény label
        self.result_var = tk.StringVar(value="")
        result_label = ttk.Label(button_frame, textvariable=self.result_var, font=("Arial", 12, "bold"))
        result_label.pack(side=tk.LEFT)
        
    def _load_image(self):
        """Kép betöltése fájlböngészővel"""
        filetypes = [
            ("Képfájlok", "*.jpg *.jpeg *.png *.bmp"),
            ("Minden fájl", "*.*")
        ]
        
        path = filedialog.askopenfilename(
            title="Válassz egy képet",
            filetypes=filetypes,
            initialdir="test"
        )
        
        if path:
            self.image_path = path
            self.path_var.set(path)
            # Ékezetes útvonal kezelése
            self.original_image = self._imread_unicode(path)
            if self.original_image is None:
                messagebox.showerror("Hiba", f"Nem sikerült betölteni a képet:\n{path}")
                return
            self._show_preview()
    
    def _imread_unicode(self, path):
        """Kép betöltése unicode útvonalról (ékezetes karakterek támogatása)"""
        try:
            # numpy buffer-ként olvassuk be a fájlt
            with open(path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
            # OpenCV dekódolja a képet
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Hiba a kép betöltésekor: {e}")
            return None
            
    def _show_preview(self):
        """Előnézeti kép megjelenítése"""
        if self.original_image is None:
            return
            
        # Átméretezés az előnézethez
        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Maximális méret
        max_width, max_height = 700, 350
        h, w = img_rgb.shape[:2]
        
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        # PIL Image -> Tkinter PhotoImage
        pil_img = Image.fromarray(img_resized)
        self.preview_photo = ImageTk.PhotoImage(pil_img)
        
        self.preview_label.config(image=self.preview_photo, text="")
        
    def _run_detection(self):
        """Kártya felismerés futtatása"""
        if self.original_image is None:
            messagebox.showwarning("Figyelmeztetés", "Először tölts be egy képet!")
            return
            
        try:
            self._process_image()
        except Exception as e:
            messagebox.showerror("Hiba", f"Hiba történt a feldolgozás során:\n{str(e)}")
            
    def _process_image(self):
        """A fő feldolgozási logika"""
        debug_corners = self.debug_corners_var.get()
        debug_flatten = self.debug_flatten_var.get()
        debug_cropped = self.debug_cropped_var.get()
        debug_template = self.debug_template_var.get()
        debug_result = self.debug_result_var.get()
        debug_thresh = self.debug_thresh_var.get()
        
        # Eredeti kép másolatok
        original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        imgResult = original_image_rgb.copy()
        imgResult2 = original_image_rgb.copy()
        
        # Kontúrok keresése
        thresh = process.get_thresh(imgResult)
        
        # Debug: élkép megjelenítése
        if debug_thresh:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].imshow(cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2RGB))
            axes[0].set_title("Eredeti kép")
            axes[0].axis('off')
            
            axes[1].imshow(thresh, cmap='gray')
            axes[1].set_title("Élkép (Canny + dilatáció)")
            axes[1].axis('off')
            
            plt.suptitle("Éldetektálás eredménye")
            plt.tight_layout()
            plt.show()
        
        corners_list = process.find_corners_set(thresh, imgResult, draw=True, debug=debug_corners)
        four_corners_set = corners_list
        
        if not corners_list:
            messagebox.showinfo("Eredmény", "Nem találtam kártyát a képen!")
            return
        
        # Perspektívikus torzítás
        flatten_card_set = process.find_flatten_cards(imgResult2, four_corners_set, debug=debug_flatten)
        
        if not flatten_card_set:
            messagebox.showinfo("Eredmény", "Nem sikerült kivágni a kártyákat!")
            return
        
        # Kártya sarkok kivágása
        cropped_images = process.get_corner_snip(flatten_card_set)
        
        # Debug: kivágott sarkok megjelenítése
        if debug_cropped:
            n_cards = len(cropped_images)
            fig, axes = plt.subplots(n_cards, 2, figsize=(6, 3 * n_cards))
            if n_cards == 1:
                axes = [axes]
            for i, (img, original) in enumerate(cropped_images):
                axes[i][0].imshow(img, cmap='gray')
                axes[i][0].set_title(f"Kártya #{i+1} - Thresh")
                axes[i][0].axis('off')
                axes[i][1].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                axes[i][1].set_title(f"Kártya #{i+1} - Original")
                axes[i][1].axis('off')
            plt.suptitle("Kivágott kártya sarkok")
            plt.tight_layout()
            plt.show()
        
        # Kontúrok keresése a kivágott sarkokon
        ranksuit_list = []
        for i, (img, original) in enumerate(cropped_images):
            drawable = img.copy()
            d2 = original.copy()

            contours, _ = cv2.findContours(drawable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts_sort = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            cnts_sort = sorted(cnts_sort, key=lambda x: cv2.boundingRect(x)[1])
            
            ranksuit = []
            for j, cnt in enumerate(cnts_sort):
                x, y, w, h = cv2.boundingRect(cnt)
                x2, y2 = x + w, y + h
                crop = d2[y:y2, x:x2]
                if j == 0:  # rank: 70, 125
                    crop = cv2.resize(crop, (70, 125), 0, 0)
                else:  # suit: 70, 100
                    crop = cv2.resize(crop, (70, 100), 0, 0)
                _, crop = cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                crop = cv2.bitwise_not(crop)
                ranksuit.append(crop)

            ranksuit_list.append(ranksuit)
        
        # Template matching
        train_ranks = Loader.load_ranks('assets/imgs/ranks')
        train_suits = Loader.load_suits('assets/imgs/suits')
        
        cardPack = CardPack()
        
        for it in ranksuit_list:
            try:
                rank = it[0]
                suit = it[1]
            except:
                continue
            r, s = process.template_matching(rank, suit, train_ranks, train_suits, show_plt=debug_template)
            cardPack.addCard(Card(r, s))
        
        # Eredmény megjelenítése
        result_text = f"Kártyák: {cardPack.cards} → {cardPack.checkHand()}"
        self.result_var.set(result_text)
        
        # Annotált kép készítése és megjelenítése (opcionális)
        if debug_result:
            annotated_image = self.original_image.copy()
            pred_texts = [f"{card.rank} {card.suit}" for card in cardPack.cards]
            process.show_text(pred_texts, four_corners_set, annotated_image)
            
            # Matplotlib ablak az eredménnyel
            plt.figure(num="Felismert kártyák", figsize=(12, 8))
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"{cardPack.cards} → {cardPack.checkHand()}")
            plt.show()


def main():
    root = tk.Tk()
    
    # Stílus beállítások
    style = ttk.Style()
    style.configure("Accent.TButton", font=("Arial", 11, "bold"))
    
    app = PokerDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()