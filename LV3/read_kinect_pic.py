import numpy as np
import cv2
import random
import time
import os
from openni import openni2
from openni import _openni2

IMG_DIR = "output-images"

def image_capture():

    if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

    try:
        # ZA WINDOWS: Ostavljamo prazno ili prosljeđujemo Windows putanju ako driver zapne
        openni2.initialize("putanja") 
        print("OpenNI2 initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        exit()

    dev = openni2.Device.open_any()

    color_stream = dev.create_color_stream()
    color_stream.set_video_mode(_openni2.OniVideoMode(
        pixelFormat=_openni2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, 
        resolutionX=320, resolutionY=240, fps=30))
    color_stream.start()

    depth_stream = dev.create_depth_stream()
    depth_stream.set_video_mode(_openni2.OniVideoMode(
        pixelFormat=_openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, 
        resolutionX=320, resolutionY=240, fps=30))
    depth_stream.start()

    if dev.is_image_registration_mode_supported(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR):
        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        print("Registration (Alignment) enabled!")

    dev.set_depth_color_sync_enabled(True)

    print("Warming up sensors...")
    time.sleep(3)
    for _ in range(30):
        color_stream.read_frame()
        depth_stream.read_frame()

    for i in range(1, 11):
        IMG_ID = f"{i:05d}"  # Formatira broj u "00001", "00002" itd.
        print(f"\nPriprema za sliku [{i}/10].")

        while True:
            color_frame = color_stream.read_frame()
            depth_frame = depth_stream.read_frame()

            color_data = color_frame.get_buffer_as_uint8()
            img = np.frombuffer(color_data, dtype=np.uint8).reshape(H_res, W_res, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

            cv2.imshow("Stisni biolo šta za slikanje", img)
            key = cv2.waitKey(1)& 0xFF

            if key != 255: 
                depth_data = depth_frame.get_buffer_as_uint16()
                depth_array = np.frombuffer(depth_data, dtype=np.uint16).reshape(H_res, W_res)

                print(f"Saving images to {IMG_DIR}...")

                # 1. Spremamo kao .bmp (kako RANSAC očekuje)
                cv2.imwrite(f"./{IMG_DIR}/sl-{IMG_ID}.bmp", img)

                np.savetxt(f"./{IMG_DIR}/sl-{IMG_ID}-D.txt", depth_array, fmt='%d', delimiter=' ') 
                print(f"Saved depth as sl-{IMG_ID}-D.txt")

                break

    # Gasi streamove
    color_stream.stop()
    depth_stream.stop()
    openni2.unload()
    print("Done!")


def read_kinect_pic(depth_path, image_shape):

    H, W = image_shape[0], image_shape[1]
    
    depth_map = np.zeros((H, W))
    depth_image = np.zeros((H, W), dtype=np.uint8)

    #depth_map = np.zeros(image_shape)
    #depth_image = np.zeros(image_shape[:2], dtype=np.uint8)

    point_3d_array = []

    d_min = 2047
    d_max = 0    

    with open(depth_path, 'r') as f:
        depth_data = f.read().strip().split('\n')
        depth_data = [row.strip().split(' ') for row in depth_data]
        for v, row in enumerate(depth_data):
            for u, d in enumerate(row):
                d = int(d)
                if d == 2047:
                    d = -1
                else:
                    if d < d_min:
                        d_min = d
                    if d > d_max:
                        d_max = d

                    point_3d_array.append([u, v, d])

                depth_map[v, u] = d

        for v, row in enumerate(depth_data):
            for u, d in enumerate(row):
                d_val = depth_map[v, u]
                if d_val != -1:

                    if d_max != d_min:
                        scaled_d = (d_val - d_min) * 254 // (d_max - d_min) + 1
                    else:
                        scaled_d = 1

                    depth_image[v, u] = np.clip(scaled_d, 0, 255)
                #d = int(d)
                #if d != -1:
                #    d = (d - d_min) * 254 // (d_max - d_min) + 1
                #    depth_image[v, u] = d

    n_3d_points = len(point_3d_array)

    return depth_image, point_3d_array, n_3d_points

def find_dominant_plane(point_3d_array, iterations=1000, epsilon=5.0):

    # Prebacujemo listu u numpy polje radi drastičnog ubrzanja rada s indeksima
    S = np.array(point_3d_array, dtype=np.float64) # Oblak točaka: stupci su [u, v, d]
    
    T_star = []
    R_star = None
    max_inliers = -1
    
    num_points = S.shape[0]
    if num_points < 3:
        print("Nedovoljno točaka za izvođenje RANSAC-a!")
        return None, []

    for it in range(iterations):
        well_conditioned = False
        a, b, c = 0, 0, 0
        
        # 2. Ponavljaj dok se ne dobije dobro kondicionirana ravnina
        while not well_conditioned:
            # (a) Nasumično izaberi tri točke (njihove indekse)
            idx = random.sample(range(num_points), 3)
            pts = S[idx] # Uzima 3 točke, svaka ima [u, v, d]
            
            # Formiranje matrice sustava (2)
            # [u1, v1, 1]
            # [u2, v2, 1]
            # [u3, v3, 1]
            A = np.array([
                [pts[0][0], pts[0][1], 1.0],
                [pts[1][0], pts[1][1], 1.0],
                [pts[2][0], pts[2][1], 1.0]
            ])
            
            B = np.array([pts[0][2], pts[1][2], pts[2][2]]) # [d1, d2, d3]
            
            # Provjera je li matrica regularna (determinanta != 0) -> uvjet kondicioniranosti
            if np.abs(np.linalg.det(A)) > 1e-5:
                try:
                    # (b) Rješavanje sustava jednadžbi za dobivanje a, b, c
                    params = np.linalg.solve(A, B)
                    a, b, c = params[0], params[1], params[2]
                    well_conditioned = True
                except np.linalg.LinAlgError:
                    # Ako unatoč det krene po zlu, probaj ponovno
                    continue

        # 3. Odredi skup točaka T koje leže na ravnini R prema uvjetu |d' - (au' + bv' + c)| <= epsilon
        # Iskoristit ćemo NumPy vektorizaciju za vrhunsku brzinu umjesto petlje kroz sve točke!
        u_coords = S[:, 0]
        v_coords = S[:, 1]
        d_coords = S[:, 2]
        
        # Izračunaj modelirana d za sve točke odjednom
        d_predicted = a * u_coords + b * v_coords + c
        
        # Provjera uvjeta |d' - d_predicted| <= epsilon
        distances = np.abs(d_coords - d_predicted)
        inlier_mask = distances <= epsilon
        
        # Broj točaka koje zadovoljavaju uvjet (|T|)
        num_inliers = np.sum(inlier_mask)
        
        # 4. Ako je |T| > |T*|
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            # 5. T* <- T (spremamo masku ili indekse točaka)
            T_star = inlier_mask
            # 6. R* <- R
            R_star = (a, b, c)

    # 7. Rezultat je dominantna ravnina R* i pripadajući inlieri
    # Vraćamo parametre ravnine i niz True/False vrijednosti koji označava koje su točke dio ravnine
    return R_star, T_star

def segment_all_planes(point_3d_array, max_planes=5, iterations=1000, epsilon=3.0, min_inliers=5000):
    """
    Segmentira sliku na više ravninskih površina.
    Vraća listu u kojoj se za svaku ravninu nalaze njezini inlieri (točke) i pripadajuća boja.
    """
    # Pretvaramo u numpy polje
    S = np.array(point_3d_array, dtype=np.float64)
    
    # Lista u koju ćemo spremati rezultate: svaka stavka je rječnik s točkama i bojom
    detected_planes = []
    
    # Radimo kopiju oblaka točaka jer ćemo iz njega brisati točke
    remaining_points = S.copy()
    
    for plane_idx in range(max_planes):
        num_points = remaining_points.shape[0]
        if num_points < 3:
            break
            
        best_inliers_mask = None
        max_inliers = -1
        best_abc = None
        
        # Klasični RANSAC nad preostalim točkama
        for it in range(iterations):
            if num_points < 3:
                break
            
            # Nasumični odabir 3 točke iz preostalih točaka
            idx = random.sample(range(num_points), 3)
            pts = remaining_points[idx]
            
            A = np.array([
                [pts[0][0], pts[0][1], 1.0],
                [pts[1][0], pts[1][1], 1.0],
                [pts[2][0], pts[2][1], 1.0]
            ])
            B = np.array([pts[0][2], pts[1][2], pts[2][2]])
            
            if np.abs(np.linalg.det(A)) > 1e-5:
                try:
                    params = np.linalg.solve(A, B)
                    a, b, c = params[0], params[1], params[2]
                except np.linalg.LinAlgError:
                    continue
                
                # Provjera inlijera na trenutno preostalim točkama
                u_coords = remaining_points[:, 0]
                v_coords = remaining_points[:, 1]
                d_coords = remaining_points[:, 2]
                
                d_predicted = a * u_coords + b * v_coords + c

                distance = np.abs(d_coords - d_predicted)
                inliers_mask = distance <= epsilon
                
                num_inliers = np.sum(inliers_mask)
                
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_inliers_mask = inliers_mask
                    best_abc = (a, b, c)
        
        # Ako je najbolja pronađena ravnina prebolna (ima više od min_inliers točaka)
        if max_inliers > min_inliers:
            print(f"Pronađena ravnina {plane_idx + 1} s {max_inliers} točaka.")
            
            # Izdvoji točke koje pripadaju ovoj ravnini
            plane_points = remaining_points[best_inliers_mask]
            
            # Generiraj nasumičnu boju za ovu ravninu (B, G, R)
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            
            # Spremi točke i boju
            detected_planes.append({
                'points': plane_points.astype(int),
                'color': color
            })
            
            # KLJUČNI KORAK: Zadrži samo one točke koje NISU inlieri (outliere) za iduću iteraciju
            outlier_index = np.where(best_inliers_mask == False)[0]
            remaining_points = remaining_points[outlier_index]
        else:
            # Ako iduća najveća ravnina ima premalo točaka, prekidamo petlju
            print(f"Nema više značajnih ravnina (sljedeća ima samo {max_inliers} točaka).")
            break
            
    return detected_planes

if __name__ == "__main__":
  
    #image_capture()

    rgb_path = f'./LV3/lv3-images/sl-00133.bmp'
    depth_path = f'./LV3/lv3-images/sl-00133-D.txt'
    
    # 1. Učitaj RGB sliku pomoću OpenCV-a
    rgb_image = cv2.imread(rgb_path)
    
    if rgb_image is None:
        print(f"Greška: Ne mogu učitati sliku na putanji {rgb_path}")
    else:
        image_shape = rgb_image.shape
        print(f"Uspješno učitana RGB slika. Dimenzije: {image_shape}")
        
        print("Učitavam dubinske podatke (ovo može potrajati)...")
        depth_image, point_3d_array, n_3d_points = read_kinect_pic(depth_path, image_shape)
        print(f"Učitano {n_3d_points} važećih 3D točaka.")

        print("Pokrećem RANSAC...")

        R_star, T_star = find_dominant_plane(point_3d_array, iterations=2000, epsilon=3.0)

        if R_star is not None:
            a, b, c = R_star
            print(f"Pronađena dominantna ravnina:")

            output_image = rgb_image.copy()
            
            # Pretvaramo point_3d_array u numpy array kako bismo izvukli u i v koordinatne inlijere
            S = np.array(point_3d_array)
            inlier_points = S[T_star] # Uzmi samo one točke gdje je T_star == True
            
            # Oboji inliere na slici u crvenu boju [B, G, R] -> [0, 0, 255]
            for pt in inlier_points:
                u, v, _ = pt
                output_image[v, u] = [0, 0, 255] # v je redak (y), u je stupac (x)
                
            # Prikaz rezultata
            cv2.imshow("Originalna RGB slika", rgb_image)
            cv2.imshow("Detektirana dominantna ravnina (Crveno)", output_image)
        else:
            print("RANSAC nije uspio pronaći ravninu.")
        
        print("Pritisni bilo koju tipku za izlaz...")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Pokrećem višestruku RANSAC segmentaciju...")
        # max_planes=5 (tražimo do 5 ravnina), min_inliers=3000 (ravnina mora imati bar 3000 točaka)
        planes = segment_all_planes(point_3d_array, max_planes=5, iterations=1500, epsilon=3.0, min_inliers=3000)
        
        # Kopiramo originalnu sliku na kojoj ćemo bojati segmente
        segmented_image = rgb_image.copy()
        
        # Prolazimo kroz sve pronađene ravnine i bojamo ih na slici
        for i, plane in enumerate(planes):
            pts = plane['points']
            color = plane['color']
            
            for pt in pts:
                u, v, _ = pt
                segmented_image[v, u] = color
        
        # Prikaz rezultata
        cv2.imshow("Originalna slika", rgb_image)
        cv2.imshow("Segmentirane ravnine (Razlicite boje)", segmented_image)
        
        print("Pritisni bilo koju tipku za izlaz...")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
