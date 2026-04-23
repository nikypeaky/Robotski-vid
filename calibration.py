import cv2
import numpy as np
import json

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

max_images = 5
board_width = 8
board_height = 6
square_length = 36

corners = []
image_points = []
board_size = (board_width, board_height)

successes = 0



found = False

while True:
    c = cv2.waitKey(15)

    ret, frame = cap.read()
    cv2.imshow('Input', frame)
    image_size = frame.size

    if c == ord('c'):
        cv2.imwrite('./LV3/image.jpg', frame)

    if c == ord('p'):
        if successes >= max_images:
            break
        img_clone = frame.copy()
        img_gray = cv2.cvtColor(img_clone, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(img_gray, board_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if found:
            corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv2.drawChessboardCorners(img_clone, board_size, corners2, found)
            image_points.append(corners2.reshape(-1,2))

            successes += 1

        cv2.imshow("Calibration", img_clone)

    if c == 27:
        break

cv2.destroyAllWindows()

if successes == max_images:
    print('Calibrating...')
    total_avg_error  = 0

    object_points = []

    for i in range(board_size[1]):
        for j in range(board_size[0]):
            object_points.append(np.array([j*square_length, i*square_length, 0]))

    object_points = np.array([object_points] * len(image_points), dtype=np.float32)#.reshape(1, -1)

    print(object_points)
    print(image_points)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_gray.shape[::-1], None, None)

    print('Re-projection error reported by calibrateCamera: ', rms)

    ok = cv2.checkRange(camera_matrix) and cv2.checkRange(dist_coeffs)

    if ok:
        print('Calibration succeeded')
    else:
        print('Calibration failed')

    out_dict = {'camera_matrix': camera_matrix.tolist(), 'dist_coeffs': dist_coeffs.tolist()}

    with open('camera_params.json', 'w') as f:
        json.dump(out_dict, f)

    if cap.isOpened():
        c = 0

        cv2.namedWindow('Original view', 1)
        while True:
            c = cv2.waitKey(15)

            ret, frame = cap.read()

            cv2.imshow('Original view', frame)

            img_clone = frame.copy()

            img_undistorted = cv2.undistort(img_clone, camera_matrix, dist_coeffs, None)

            cv2.imshow('Undistorted view', img_undistorted)
            if c == 27:
                break
            
    cv2.destroyAllWindows()

cap.release()