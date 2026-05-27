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

#----------------------------------------------------------------------
#učitavanje slika i traženje kuteva na šahovnici
print('Učitavanje slika...')

for i in range(max_images):
    putanja_slike = f'./LV1/slike za kalibraciju/calib_{i}.jpg'
    
    frame = cv2.imread(putanja_slike)
    
    if frame is None:
        print(f"Nije moguće pronaći sliku {putanja_slike}")
        continue

    img_clone = frame.copy()
    img_gray = cv2.cvtColor(img_clone, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(img_gray, board_size, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if found:
        corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), 
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        cv2.drawChessboardCorners(img_clone, board_size, corners2, found)

        ime_prozora = f"Šahovnica na slici calib_{i}"
        
        cv2.namedWindow(ime_prozora, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ime_prozora, 900, 600)
        cv2.imshow(ime_prozora, img_clone)

        #cekaj dok se ne stisne tipka na tipkovnici pa zatvorit prozor
        while True:
            tipka = cv2.waitKey(100) & 0xFF
            if tipka != 255:
                break
                
        cv2.destroyWindow(ime_prozora) 

        image_points.append(corners2.reshape(-1,2))
        successes += 1
    else:
        print(f"Šahovnica nije pronađena na slici calib_{i}.jpg")

cv2.destroyAllWindows()
#---------------------------------------------------------------------
# Apokreni kalibraciju
if successes == max_images:
    print('Pokrećem kalibraciju...')
    total_avg_error = 0

    object_points = []

    for i in range(board_size[1]):
        for j in range(board_size[0]):
            object_points.append(np.array([j*square_length, i*square_length, 0]))

    object_points = np.array([object_points] * len(image_points), dtype=np.float32)

    print(object_points)
    print(image_points)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, img_gray.shape[::-1], None, None
    )

    print('Re-projection error reported by calibrateCamera: ', rms)

    ok = cv2.checkRange(camera_matrix) and cv2.checkRange(dist_coeffs)

    if ok:
        print('Calibration succeeded!')

        out_dict = {'camera_matrix': camera_matrix.tolist(), 'dist_coeffs': dist_coeffs.tolist()}
        with open('camera_params.json', 'w') as f:
            json.dump(out_dict, f)
        
        putanja_referentne = './LV1/slike za detekciju/slika_0.jpg' 
        img_ref = cv2.imread(putanja_referentne)

        if img_ref is None:
            print(f"Nije moguće pronaći referentnu sliku: {putanja_referentne}")
        else:
            img_ref_ispravljena = cv2.undistort(img_ref, camera_matrix, dist_coeffs)

            #-------------------------------------------------------------------------------------------------
            #određivanje koordinardnog sustava mm papira

            points_clicked = []

            def click(event, x_m, y_m, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    points_clicked.append((x_m, y_m))
                    cv2.circle(img_papir_prikaz, (x_m, y_m), 5, (0, 0, 255), -1) # Crvena točka
                    cv2.imshow(ime_papira, img_papir_prikaz)
                    
                    red_klika = len(points_clicked)
                    if red_klika == 1:
                        print("1. Kliknuto Ishodište (0,0)")
                    elif red_klika == 2:
                        print("2. Kliknut kraj X-osi (277,0)")
                    elif red_klika == 3:
                        print("3. Kliknut kraj Y-osi (0,180)")
                    elif red_klika == 4:
                        print("4. Kliknuta Dijagonala (277,180)")

            img_papir_prikaz = img_ref_ispravljena.copy()
            ime_papira = "Postavi koordinatni sustav papira (klikni 4 kuta)"
            
            cv2.namedWindow(ime_papira, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(ime_papira, 900, 600)
            cv2.setMouseCallback(ime_papira, click)


            cv2.imshow(ime_papira, img_papir_prikaz)

            while len(points_clicked) < 4:
                cv2.waitKey(100)

            cv2.waitKey(0)
            cv2.destroyWindow(ime_papira)

            # Računanje homografije za pretvorbu piksela u stvarne milimetre (A4 okvir unutar crta)
            pts_izvorne = np.float32(points_clicked)
            pts_stvarne = np.float32([[0, 0], [277, 0], [0, 180], [277, 180]])
            H_papir, _ = cv2.findHomography(pts_izvorne, pts_stvarne)

            #---------------------------------------------------------------------------------------------
            #ROI označavanje            
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
                #---------------------------------------------------------------------------------------------------
                #SIFT na prvoj slici
                sift = cv2.SIFT_create()
                kp_model, des_model = sift.detectAndCompute(sivi_objekt, None)
                print(f"Detektirano {len(kp_model)} SIFT značajki na modelu.")

                # FLANN za sparivanj znacajki kod sifta
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)

                print("\nPokrećem sparivanje i povezivanje značajki pravcima...")

                #-------------------------------------------------------------------------------------------
                #SIFT za ostale slike
                for k in range(1, 6): # Ide od slika_1.jpg do slika_5.jpg
                    putanja_trenutne = f'./LV1/slike za detekciju/slika_{k}.jpg'
                    img_trenutna = cv2.imread(putanja_trenutne)
                    
                    if img_trenutna is None:
                        print(f"Nije moguće pronaći sliku {putanja_trenutne}")
                        continue
                        
                    # za svaku sliku radimo undistort i trazimo sift znacajke
                    img_trenutna_ispravljena = cv2.undistort(img_trenutna, camera_matrix, dist_coeffs)
                    siva_trenutna = cv2.cvtColor(img_trenutna_ispravljena, cv2.COLOR_BGR2GRAY)

                    kp_scena, des_scena = sift.detectAndCompute(siva_trenutna, None)

                    if des_model is not None and des_scena is not None:
                        #pomocu knn algortima trazimo dva najbolja pogotka sa des_model i des_scena 
                        matches = matcher.knnMatch(des_model, des_scena, k=2)
                        
                        #filtriranje da ne bi krivo uzeli točke pozadine koje su slične
                        #distance je mjera sličnosti točaka tj njihovih deskritpora
                        #m.distance mora biti značajno manji, tj. značajno bolja točka na sceni od n.distance
                        #njihov omjer mora biti manji od 0.7
                        good_matches = []
                        for par in matches:
                            if len(par) == 2:
                                m, n = par
                                if m.distance < 0.7 * n.distance:
                                    good_matches.append(m)
                        
                        print(f"Slika {k}: Pronađeno {len(good_matches)} ispravno sparenih točaka.")
                        
                        #provjeravamo imamo li minimalno 4 sprarene točke, ako da onda možemo rekonstruirati ravninu 
                        if len(good_matches) >= 4:
                            src_pts = np.float32([kp_model[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp_scena[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            
                            #tražimo matricu transformacije da bi mogli naći točke modela na novoj sceni
                            #koristimo RANSAC za odbacivanje outliera
                            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            
                            if M is not None:
                                #uzimamo točke koje smo dobili iz roia i pomocu matrice m i trazimo gdje se nalaze kutevi na novoj sceni
                                h_obj, w_obj = sivi_objekt.shape
                                kutevi_modela = np.float32([[0, 0], [0, h_obj - 1], [w_obj - 1, h_obj - 1], [w_obj - 1, 0]]).reshape(-1, 1, 2)
                                kutevi_scene = cv2.perspectiveTransform(kutevi_modela, M)
                                
                               #-----------------------------------------------------------------------------------------------------------
                               #crta bijeli okvir oko modela na novoj sceni pomocu kuteva
                                cv2.polylines(img_trenutna_ispravljena, [np.int32(kutevi_scene)], True, (255, 255, 255), 4, cv2.LINE_AA)
                                
                                #računamo središte i kut zakrenutosti objekta
                                kutevi_clp = kutevi_scene.reshape(4, 2)
                                centar_x = float(np.mean(kutevi_clp[:, 0])) #tražimo srednju vrijednost x koordinata kuteva
                                centar_y = float(np.mean(kutevi_clp[:, 1]))# i y isto
                                
                                # Računanje kuta orijentacije (od lijevog gornjeg kuta  do lijevog donjeg kuta)
                                dx = kutevi_clp[1, 0] - kutevi_clp[0, 0]
                                dy = kutevi_clp[1, 1] - kutevi_clp[0, 1]
                                kut_deg = math.degrees(math.atan2(dy, dx))

                                #racunanje koordinata u mm pomocu matrice papira
                                tocka_px = np.array([[[centar_x, centar_y]]], dtype=np.float32)
                                tocka_mm = cv2.perspectiveTransform(tocka_px, H_papir)
                                mm_x = tocka_mm[0][0][0]
                                mm_y = tocka_mm[0][0][1]
             
                                cv2.circle(img_trenutna_ispravljena, (int(centar_x), int(centar_y)), 7, (255, 0, 0), -1)

                                print(f"Središte u slici: X= {mm_x:.1f} mm, Y= {mm_y:.1f} mm")
                                print(f"Orijentacija u slici: Kut = {kut_deg:.1f}°")

                        # Spaja referentni_objekt i img_trenutna_ispravljena u jedan prozor i vuče linije prema SIFT 

                        prikaz_pravaca = cv2.drawMatches(
                            referentni_objekt, kp_model, 
                            img_trenutna_ispravljena, kp_scena, 
                            good_matches, None, 
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
                        
            else:
                print("Niste označili ispravan ROI.")

    else:
        print('Calibration failed (Kalibracija neuspješna)!')
    print("-" * 40)
else:
    print(f"Kalibracija prekinuta. Uspješno obrađeno samo {successes}/{max_images} slika.")
