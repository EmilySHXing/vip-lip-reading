import numpy as np
import cv2
import argparse
import dlib
from imutils import face_utils
import skvideo.io

# https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

# use an argument parser
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
#ap.add_argument("-i", "--input", required=True, help="path to input video")
#args = vars(ap.parse_args())

# read an image and convert to grayscale using opencv
# img = cv2.imread(args["image"])
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# read input video
reader = skvideo.io.vread("s.MOV")
skvideo.io.vwrite('allow.gif', reader)
video_shape = reader.shape

# convert each frame to gray scale
frames = np.zeros((video_shape[0],video_shape[1],video_shape[2]))
for i in range(video_shape[0]):
    frames[i] = cv2.cvtColor(reader[i],cv2.COLOR_BGR2GRAY)

# print(frames)

# https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
# face_cascade = cv2.CascadeClassifier('
# /Users/macbookpro/xing35/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# initialize lip center info
centers = np.zeros((video_shape[0],2))

new = np.zeros(frames.shape)

for m in range(frames.shape[0]):
    frame = frames[m].astype('uint8')
    # faces = face_cascade.detectMultiScale(frame, 1.3, 5)  # detect faces using opencv
    rects = detector(frame, 1)  # detect faces using dlib for each frame in video

# Draw rectangle around face
# for (x, y, w, h) in faces:
#    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#    roi_gray = gray[y:y+h, x:x+w]
#    roi_color = img[y:y+h, x:x+w]

# save modified image
# cv2.imwrite('img.jpg',img)


    coords = np.zeros((2,20))
    width_max = 0

# loop over the face detections
# https://github.com/astorfi/lip-reading-deeplearning/blob/master/code/lip_tracking/VisualizeLip.py
    for (i, rect) in enumerate(rects):
        shape = predictor(frame, rect)  # detect facial landmark positions
        k = 0
    # find mouth related coordinates
        for j in range(48, 68):
            X = shape.part(j)
            coords[0, k] = X.x
            coords[1, k] = X.y
            k += 1

    # find mouth region
        x_min = int(np.amin(coords, axis=1)[0])
        x_max = int(np.amax(coords, axis=1)[0])
        y_min = int(np.amin(coords, axis=1)[1])
        y_max = int(np.amax(coords, axis=1)[1])
        # print(x_min,x_max,y_min,y_max)

        width = max(x_max-x_min, y_max-y_min)
        width = int(width / 2) + 5
        x_center = int((x_max + x_min) / 2)
        y_center = int((y_max + y_min) / 2)
        centers[m] = [y_center, x_center]
        width = max(width, width_max)
        width_max = width

        # mouth = frame[(y_center-width):(y_center+width),(x_center-width):(x_center+width)]
        # cv2.imwrite('mouth' + str(m) + '.jpg',mouth)

        shape = face_utils.shape_to_np(shape)  # convert to numPy array

        (x, y, w, h) = face_utils.rect_to_bb(rect)  # convert to coordinate location
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw rectangle around face

    # show the face number
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    new[m] = frame

mouth = np.zeros((frames.shape[0],width*2,width*2))

# crop mouth from each frame
for m in range(frames.shape[0]):
        mouth[m] = frames[m,int(centers[m,0]-width):int(centers[m,0]+width),int(centers[m,1]-width):int(centers[m,1]+width)]

skvideo.io.vwrite('mouth.gif',mouth)
#skvideo.io.vwrite('new.gif',new)


# mouth related range 48-68
