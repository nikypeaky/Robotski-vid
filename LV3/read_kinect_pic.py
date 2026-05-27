import numpy as np

def read_kinect_pic(depth_path, image_shape):
    depth_map = np.zeros(image_shape)
    depth_image = np.zeros(image_shape[:2], dtype=np.uint8)

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
                d = int(d)
                if d != -1:
                    d = (d - d_min) * 254 // (d_max - d_min) + 1
                    depth_image[v, u] = d

    n_3d_points = len(point_3d_array)

    return depth_image, point_3d_array, n_3d_points