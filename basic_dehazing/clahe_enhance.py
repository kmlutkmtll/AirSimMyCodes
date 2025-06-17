import cv2
import numpy as np

# Giriş ve çıkış dosyası yolları
input_path = "img.png"  # Sisli olan orijinal görüntü
output_path = "clahe_sharpened.jpg"

# Görüntüyü yükle
image = cv2.imread(input_path)

# BGR → LAB dönüşümü
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# CLAHE uygulaması (Kontrastı artırır)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l)

# LAB görüntüsünü geri birleştir
limg = cv2.merge((cl, a, b))
enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Hafif keskinleştirme filtresi (sharpen)
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened = cv2.filter2D(enhanced, -1, kernel)

# Sonucu kaydet
cv2.imwrite(output_path, sharpened)
print(f"[+] İyileştirilmiş görüntü kaydedildi: {output_path}")
