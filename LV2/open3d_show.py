import json
import numpy as np
import open3d as o3d

def main():
    # Putanja do generirane JSON datoteke s 3D točkama
    points_3d_path = './LV2/points_3d.json'

    # 1. Učitavanje 3D točaka iz datoteke
    try:
        with open(points_3d_path, 'r') as f:
            points_data = json.load(f)
        points_3d = np.array(points_data)
    except FileNotFoundError:
        print(f"Greška: Datoteka {points_3d_path} ne postoji. Prvo pokreni glavnu skriptu da generiraš točke!")
        return

    print(f"Uspješno učitano {len(points_3d)} točaka za Open3D vizualizaciju.")

    # 2. Kreiranje Open3D PointCloud objekta
    pcd = o3d.geometry.PointCloud()
    
    # Budući da kamere često imaju invertiranu Z-os (gledaju "unutra"), 
    # ako primijetiš da je model naopak, ovdje ga možemo okrenuti dodavanjem minusa (-points_3d)
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # 3. Bojanje točaka u upečatljivu boju (npr. svijetlo plava) radi bolje vidljivosti
    pcd.paint_uniform_color([0.2, 0.6, 0.8])

    # 4. Izračun normala (pomaže grafičkom engineu da bolje osjenča točke i prikaže dubinu)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Zadatak - 3D Oblak Točaka (Open3D)",
        width=1024,
        height=768,
        left=50,
        top=50
    )

if __name__ == '__main__':
    main()