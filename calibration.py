import cv2
import numpy as np
import json
import math

## u paintu smanjiti rezoluciju slika na 1280 x 720

max_images = 5
board_width = 8
board_height = 6
square_length = 36

corners = []
image_points = []
board_size = (board_width, board_height)

successes = 0
found = False

print('Učitavanje slika s diska i pokretanje detekcije kuteva...')

for i in range(max_images):
    putanja_slike = f'./slike za kalibraciju/calib_{i}.jpg'
    
    # Učitavanje slike s diska
    frame = cv2.imread(putanja_slike)
    
    if frame is None:
        print(f"Greška: Nije moguće pronaći sliku {putanja_slike}")
        continue

    img_clone = frame.copy()
    img_gray = cv2.cvtColor(img_clone, cv2.COLOR_BGR2GRAY)

    # Detekcija kuteva (Profesoricina logika)
    found, corners = cv2.findChessboardCorners(img_gray, board_size, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if found:
        corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), 
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # Prikazujemo kako je algoritam pronašao kuteve na toj slici
        cv2.drawChessboardCorners(img_clone, board_size, corners2, found)

        ime_prozora = f"Detekcija kuteva - Slika {i}"
        
        cv2.namedWindow(ime_prozora, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ime_prozora, 900, 600)
        cv2.imshow(ime_prozora, img_clone)

        while True:
            tipka = cv2.waitKey(100) & 0xFF
            # Ako je pritisnuta bilo koja tipka (osim 255 što znači da ništa nije pritisnuto)
            if tipka != 255:
                break
                
        cv2.destroyWindow(ime_prozora)  # Zatvara prozor tek nakon što si stisnuo tipku

        image_points.append(corners2.reshape(-1,2))
        successes += 1
    else:
        print(f"Upozorenje: Šahovnica nije pronađena na slici calib_{i}.jpg")

cv2.destroyAllWindows()

# Ako su sve slike uspješno obrađene, pokreće se matematika za kalibraciju
if successes == max_images:
    print('Pokrećem kalibraciju (cv2.calibrateCamera)...')
    total_avg_error = 0

    object_points = []

    for i in range(board_size[1]):
        for j in range(board_size[0]):
            object_points.append(np.array([j*square_length, i*square_length, 0]))

    object_points = np.array([object_points] * len(image_points), dtype=np.float32)

    # Glavna funkcija za kalibraciju
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, img_gray.shape[::-1], None, None
    )

    print('Re-projection error reported by calibrateCamera: ', rms)

    ok = cv2.checkRange(camera_matrix) and cv2.checkRange(dist_coeffs)

    if ok:
        print('Calibration succeeded (Kalibracija uspješna)!')
        print(f'Re-projection error (RMS): {rms:.4f}')
        
        # Spremanje u JSON datoteku
        out_dict = {'camera_matrix': camera_matrix.tolist(), 'dist_coeffs': dist_coeffs.tolist()}
        with open('camera_params.json', 'w') as f:
            json.dump(out_dict, f)
        
        putanja_referentne = './slike za detekciju/slika_0.jpg' 
        img_ref = cv2.imread(putanja_referentne)

        if img_ref is None:
            print(f"Greška: Nije moguće pronaći referentnu sliku: {putanja_referentne}")
        else:
            img_ref_ispravljena = cv2.undistort(img_ref, camera_matrix, dist_coeffs)

            # 2. Označavanje objekta mišem (ROI) - OVO RADIMO SAMO JEDNOM
            print("\n>>> OTVARA SE PROZOR ZA OZNAČAVANJE OBJEKTA <<<")
            print("UPUTE: Označite objekt mišem na prvoj slici i pritisnite ENTER.")
            
            ime_oznacavanja = "Oznaci objekt na referentnoj slici i stisni ENTER"
            
            cv2.namedWindow(ime_oznacavanja, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(ime_oznacavanja, 900, 600)

            roi = cv2.selectROI(ime_oznacavanja, img_ref_ispravljena, fromCenter=False, showCrosshair=True)
            x, y, w, h = roi
            cv2.destroyWindow(ime_oznacavanja)

            if w > 0 and h > 0:
                print(f"Objekt uspješno označen! Dimenzije: {w}x{h} px.")
                referentni_objekt = img_ref_ispravljena[y:y+h, x:x+w]
                sivi_objekt = cv2.cvtColor(referentni_objekt, cv2.COLOR_BGR2GRAY)
                sivi_objekt_za_prikaz = cv2.cvtColor(sivi_objekt, cv2.COLOR_GRAY2BGR)
                
                sift = cv2.SIFT_create()
                kp_model, des_model = sift.detectAndCompute(sivi_objekt, None)
                print(f" -> Detektirano {len(kp_model)} SIFT značajki na modelu.")

                # Inicijalizacija Flann Matchera za brzo sparivanje značajki
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)

                print("\nPokrećem sparivanje i povezivanje značajki pravcima...")

                # 3. PETLJA KROZ OSTALIH 5 SLIKA ZA DETEKCIJU (od 1 do 5)
                print("\nPokrećem prolazak kroz ostalih 5 slika za detekciju...")
                
                for k in range(1, 6): # Ide od slika_1.jpg do slika_5.jpg
                    putanja_trenutne = f'./slike za detekciju/slika_{k}.jpg'
                    img_trenutna = cv2.imread(putanja_trenutne)
                    
                    if img_trenutna is None:
                        print(f"Upozorenje: Nije moguće pronaći sliku {putanja_trenutne}")
                        continue
                        
                    # Ispravljanje distorzije za trenutnu sliku iz mape
                    img_trenutna_ispravljena = cv2.undistort(img_trenutna, camera_matrix, dist_coeffs)
                    siva_trenutna = cv2.cvtColor(img_trenutna_ispravljena, cv2.COLOR_BGR2GRAY)
                    siva_trenutna_za_prikaz = cv2.cvtColor(siva_trenutna, cv2.COLOR_GRAY2BGR)

                    kp_scena, des_scena = sift.detectAndCompute(siva_trenutna, None)

                    if des_model is not None and des_scena is not None:
                        # Pronalazimo top 2 najbolja pogotka za svaku točku (k=2)
                        matches = matcher.knnMatch(des_model, des_scena, k=2)
                        
                        # Filtriranje dobrih pogodaka (Lowe's ratio test)
                        good_matches = []
                        for par in matches:
                            if len(par) == 2:
                                m, n = par
                                if m.distance < 0.7 * n.distance:
                                    good_matches.append(m)
                        
                        print(f"Slika {k}: Pronađeno {len(good_matches)} ispravno sparenih točaka.")
                        
                        if len(good_matches) >= 4:
                            src_pts = np.float32([kp_model[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp_scena[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            
                            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            
                            if M is not None:
                                h_obj, w_obj = sivi_objekt.shape
                                kutevi_modela = np.float32([[0, 0], [0, h_obj - 1], [w_obj - 1, h_obj - 1], [w_obj - 1, 0]]).reshape(-1, 1, 2)
                                kutevi_scene = cv2.perspectiveTransform(kutevi_modela, M)
                                
                                # --- OVDJE JE PROMIJENJENA BOJA U BIJELU ---
                                # (255, 255, 255) predstavlja bijelu boju u BGR formatu, debljina je postavljena na 4
                                cv2.polylines(siva_trenutna_za_prikaz, [np.int32(kutevi_scene)], True, (255, 255, 255), 4, cv2.LINE_AA)
                                
                                kutevi_clp = kutevi_scene.reshape(4, 2)
                                centar_x = float(np.mean(kutevi_clp[:, 0]))
                                centar_y = float(np.mean(kutevi_clp[:, 1]))
                                
                                # Računanje kuta orijentacije (orijentacija gornjeg ruba kutije u slici)
                                dx = kutevi_clp[1, 0] - kutevi_clp[0, 0]
                                dy = kutevi_clp[1, 1] - kutevi_clp[0, 1]
                                kut_deg = math.degrees(math.atan2(dy, dx))
                                
                                # Iscrtaj plavu točku u središtu objekta
                                cv2.circle(siva_trenutna_za_prikaz, (int(centar_x), int(centar_y)), 7, (255, 0, 0), -1)
                                
                                print(f"\n========== REZULTATI ZA SLIKU {k} ==========")
                                print(f"Središte u slici: X = {centar_x:.1f} px, Y = {centar_y:.1f} px")
                                print(f"Orijentacija u slici: Kut = {kut_deg:.1f}°")
                                print("==========================================")

                        # --- KLJUČNA FUNKCIJA ZA TVOJ ZADATAK ---
                        # Spaja referentni_objekt i img_trenutna_ispravljena u jedan prozor i vuče linije
                        zelena_boja = (0, 255, 0)

                        prikaz_pravaca = cv2.drawMatches(
                            sivi_objekt_za_prikaz, kp_model, 
                            siva_trenutna_za_prikaz, kp_scena, 
                            good_matches, None, 
                            matchColor= zelena_boja,
                            singlePointColor=None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                        )

                        # Prikaz rezultata na ekranu
                        ime_prozora_spajanja = f"Sparene znacajke - Slika {k}"
                       
                        cv2.namedWindow(ime_prozora_spajanja, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(ime_prozora_spajanja, 1200, 700)
                        cv2.imshow(ime_prozora_spajanja, prikaz_pravaca)                        
                        print(f"Pravci su iscrtani za sliku {k}. Pritisni bilo koju tipku za iduću sliku...")
                        
                        cv2.waitKey(0)
                        cv2.destroyWindow(ime_prozora_spajanja)

                print("\nPogledaj plavu točku!")
            else:
                print("Niste označili ispravan ROI.")

    else:
        print('Calibration failed (Kalibracija neuspješna)!')
    print("-" * 40)
else:
    print(f"Kalibracija prekinuta. Uspješno obrađeno samo {successes}/{max_images} slika.")
