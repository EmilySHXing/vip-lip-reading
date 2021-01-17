import numpy as np
import cv2
import dlib
import skvideo.io

# initialize file paths
filename = "TODAY/val/TODAY_0000"
ext = ".mp4"
outfile = "data/Lips/Today/val/TODAY_0000"

# initialize dlib face detector and landmark predictors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# loop through videos in directory
for i in range(1,10):
    name = filename + str(i) + ext
    out = outfile + str(i) + ext
    # read a video
    try:
        reader = skvideo.io.vread(name)
        video_shape = reader.shape
        frames = np.zeros((video_shape[0], video_shape[1], video_shape[2]))
        # turn each frame into gray scale
        for j in range(video_shape[0]):
            frames[j] = cv2.cvtColor(reader[j], cv2.COLOR_BGR2GRAY)

        # initialize coordinates of center of lip to (0,0)s
        centers = np.zeros((video_shape[0], 2))
        width_max = 0
        for m in range(frames.shape[0]):
            frame = frames[m].astype('uint8')
            # detect face in current frame
            rects = detector(frame, 1)

            # initialize coordinates for landmarks around lip area
            coords = np.zeros((2, 20))

            # predict facial landmarks
            shape = predictor(frame, rects[0])
            k = 0

            # save coordinates of landmarks around lip
            for n in range(48, 68):
                X = shape.part(n)
                coords[0, k] = X.x
                coords[1, k] = X.y
                k += 1

            # calculate minimum and maximum x, y coordinate for lip
            x_min = int(np.amin(coords, axis=1)[0])
            x_max = int(np.amax(coords, axis=1)[0])
            y_min = int(np.amin(coords, axis=1)[1])
            y_max = int(np.amax(coords, axis=1)[1])

            # calculate width of lip
            width = max(x_max - x_min, y_max - y_min)
            width = int(width / 2) + 3

            # calculate center coordinate of lip
            x_center = int((x_max + x_min) / 2)
            y_center = int((y_max + y_min) / 2)
            centers[m] = [y_center, x_center]

            # save the maximum width
            width_max = max(width, width_max)
    except Exception as e:
        continue

    width = width_max
    # crop mouth from each frame, resize to 32x32 and arrange into 3d array
    mouth = np.zeros((frames.shape[0], 32, 32))
    for m in range(frames.shape[0]):
        mouth[m] = cv2.resize(frames[m, int(centers[m, 0] - width):int(centers[m, 0] + width),
                              int(centers[m, 1] - width):int(centers[m, 1] + width)], (32, 32))
    # save to file
    skvideo.io.vwrite(out, mouth)