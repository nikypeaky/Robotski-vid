import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
from convert_2d_points_to_3d_points import convert_2d_points_to_3d_points

def main():
    # 1. Učitavanje slika (prilagodi putanju ako su u nekoj mapi, npr. 'LV5/imageL.bmp')
    img_L = cv2.imread('./LV2/test2L.jpeg')
    img_R = cv2.imread('./LV2/test2R.jpeg')

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

    sift = cv2.SIFT_create()

    # 5. Pronalaženje ključnih točaka (kp) i računanje njihovih deskriptora (des)
    kp_L, des_L = sift.detectAndCompute(img_L, None)
    kp_R, des_R = sift.detectAndCompute(img_R, None)

    # 6. Inicijalno sparivanje značajki pomoću Brute-Force Matcher-a (BFMatcher)
    # Koristimo cv2.NORM_L2 jer je to standardna mjera udaljenosti za SIFT deskriptore
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_L, des_R)

    # Sortiramo sparivanja po udaljenosti (od najboljih/najsličnijih prema lošijima)
    matches = sorted(matches, key=lambda x: x.distance)

    # 7. Isrisavanje prvih 50 najboljih sparivanja radi provjere
    img_matches = cv2.drawMatches(img_L_rgb, kp_L, img_R_rgb, kp_R, matches[:50], None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 8. Prikaz spojenih slika s linijama sparivanja
    plt.figure(figsize=(15, 8))
    plt.imshow(img_matches)
    plt.title('Inicijalna SIFT sparivanja (Top 50)')
    plt.axis('off')
    plt.show()

    # 9. Priprema koordinata točaka za RANSAC
    # Iz svakog sparivanja (match) uzimamo indeks ključne točke i izvlačimo njezine (x, y) koordinate
    pts_L = np.float32([kp_L[m.queryIdx].pt for m in matches])
    pts_R = np.float32([kp_R[m.trainIdx].pt for m in matches])

    # 10. Estimacija Fundamentalne matrice (F) pomoću RANSAC metode
    # Parametar 1.0 je RANSAC prag (udaljenost u pikselima od epipolarne linije)
    # Parametar 0.99 je željena pouzdanost (confidence)
    F_matrix, mask = cv2.findFundamentalMat(pts_L, pts_R, cv2.FM_RANSAC, 1.0, 0.99)

    # mask nam vraća 1 za točna sparivanja (inliers) i 0 za pogrešna (outliers)
    # Pretvaramo masku u ravnu listu bajtova radi lakšeg filtriranja
    matches_mask = mask.ravel().tolist()

    print(f"Ukupno inicijalnih sparivanja: {len(matches)}")
    print(f"Broj točnih sparivanja nakon RANSAC-a (inliers): {sum(matches_mask)}")
    print("Fundamentalna matrica F je uspješno izračunata.")

    # 11. Prikaz SAMO točnih sparivanja koje je RANSAC odobrio
    # Koristimo matchesMask parametar u drawMatches kako bismo sakrili "uljeze"
    draw_params = dict(matchColor=(0, 255, 0),  # Točna sparivanja crtamo zelenom bojom
                       singlePointColor=None,
                       matchesMask=matches_mask,  # Prikazuje samo točke gdje je maska == 1
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_ransac_matches = cv2.drawMatches(img_L_rgb, kp_L, img_R_rgb, kp_R, matches, None, **draw_params)

    # Prikaz filtriranih sparivanja
    plt.figure(figsize=(15, 8))
    plt.imshow(img_ransac_matches)
    plt.title('Sparivanja nakon RANSAC filtriranja (Samo Inliers)')
    plt.axis('off')
    plt.show()

    # 12. Fizičko odbacivanje krivo sparenih točaka na temelju RANSAC maske
    # Zadržavamo samo točke kod kojih je mask[i] == 1 (zadovoljavaju epipolarno ograničenje)
    pts_L_final = []
    pts_R_final = []

    for i in range(len(matches)):
        if matches_mask[i] == 1:
            # Izvlačimo originalne KeyPoint objekte koji su prošli provjeru
            pts_L_final.append(kp_L[matches[i].queryIdx])
            pts_R_final.append(kp_R[matches[i].trainIdx])

    print(f"Broj preostalih parova značajki nakon filtriranja maskom: {len(pts_L_final)}")
    
    # Sada imamo dvije čiste liste (pts_L_final i pts_R_final) koje sadrže 
    # OpenCV KeyPoint objekte spremne za profesoričinu funkciju!

    # 13. Ponovno označavanje i povezivanje pravcima samo točnih parova značajki (Inliers)
    # Postavljamo parametre za crtanje tako da se prikazuju samo parovi odobreni maskom
    draw_params = dict(
        matchColor=(0, 255, 0),       # Točne parove povezujemo zelenim pravcima
        singlePointColor=None,        # Ne iscrtavamo ključne točke koje nemaju svoj par
        matchesMask=matches_mask,     # Primjena RANSAC maske za filtriranje prikaza
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Crta linije između preostalih (točnih) značajki na obje slike
    img_filtered_matches = cv2.drawMatches(
        img_L_rgb, kp_L, 
        img_R_rgb, kp_R, 
        matches, None, 
        **draw_params
    )

    # 14. Prikaz rezultata ponovnog označavanja i povezivanja
    plt.figure(figsize=(15, 8))
    plt.imshow(img_filtered_matches)
    plt.title('Ponovno označeni i pravcima povezani točni parovi značajki (RANSAC Inliers)')
    plt.axis('off')
    plt.show()

    # 15. Učitavanje kalibracijske/projekcijske matrice kamere (P) iz JSON datoteke
    with open('./LV2/camera_params_LV5.json', 'r') as f:
        camera_data = json.load(f)
    
    # Pretvaramo pročitane podatke u standardno NumPy polje float tipa
    P_matrix = np.array(camera_data['camera_params'], dtype=np.float64)

    # 16. Određivanje esencijalne matrice E prema formuli: E = P.T * F * P
    # Koristimo .dot() za matrično množenje i .T za transponiranje matrice P
    E_matrix = P_matrix.T.dot(F_matrix).dot(P_matrix)

    # Ispis rezultata u konzoli radi provjere
    print("\n--- Kalibracijska matrica kamere (P) ---")
    print(P_matrix)
    print("\n--- Izračunata Esencijalna matrica (E) ---")
    print(E_matrix)
    print("------------------------------------------")

    # 17. Pozivanje profesoričine funkcije za estimaciju 3D točaka (Triangulacija)
    # Prosljeđujemo filtrirane parove točaka, esencijalnu matricu (E) i matricu kamere (P)
    E_za_funkciju = np.matrix(E_matrix.reshape(3, 3))
    P_za_funkciju = np.matrix(P_matrix.reshape(3, 3))

    print("\nPokretanje trodimenzionalne rekonstrukcije...")
    points_3d = convert_2d_points_to_3d_points(pts_L_final, pts_R_final, E_za_funkciju, P_za_funkciju)
    
    print(f"Uspješno rekonstruirano {len(points_3d)} 3D točaka u prostoru.")

    # 18. Spremanje izračunatih 3D koordinata u JSON datoteku
    # Budući da points_3d_path u tvom kodu za crtanje glasi 'LV5/points_3d.json', 
    # osigurat ćemo da mapa postoji prije nego spremimo datoteku
    import os
    output_dir = 'LV2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    points_3d_path = os.path.join(output_dir, 'points_3d.json')

    # NumPy polje pretvaramo u običnu listu (.tolist()) jer json.dump ne podržava direktno NumPy polja
    with open(points_3d_path, 'w') as f:
        json.dump(points_3d.tolist(), f, indent=4)

    print(f"3D koordinate su uspješno spremljene u datoteku: {points_3d_path}")
    print("Sada možeš pokrenuti drugi dio koda za vizualizaciju 3D modela!")

if __name__ == '__main__':
    main()