import cv2
import matplotlib.pyplot as plt

def main():
    # 1. Učitavanje slika (prilagodi putanju ako su u nekoj mapi, npr. 'LV5/imageL.bmp')
    img_L = cv2.imread('imageL.bmp')
    img_R = cv2.imread('imageR.bmp')

    # Provjera jesu li slike uspješno učitane
    if img_L is None or img_R is None:
        print("Greška: Nije moguće učitati slike! Provjeri putanje.")
        return

    # 2. Pretvorba iz BGR u RGB format (za ispravan prikaz u Matplotlibu)
    img_L_rgb = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
    img_R_rgb = cv2.cvtColor(img_R, cv2.COLOR_BGR2RGB)

    # 3. Prikaz slika jednu pokraj druge
    plt.figure(figsize=(12, 6))

    # Lijeva slika
    plt.subplot(1, 2, 1)
    plt.imshow(img_L_rgb)
    plt.title('Lijeva slika (imageL)')
    plt.axis('off')  # Isključuje prikaz koordinatnih osi (piksela)

    # Desna slika
    plt.subplot(1, 2, 2)
    plt.imshow(img_R_rgb)
    plt.title('Desna slika (imageR)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()