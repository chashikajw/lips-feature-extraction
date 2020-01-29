from collections import OrderedDict
import numpy as np
import cv2
import argparse
import dlib
import imutils
import csv
import os
import math

facial_features_cordinates = {}

# defined facial landmarks to face regions
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 35)),
    ("Jaw", (0, 17))
])




shapepredictorPara = "shape_predictor_68_face_landmarks.dat"

def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates

def generateMouthParameters(mouth_cordinates):
    mouth_paramaters = {} #49-68  
    mouth_paramaters["w"] = abs((mouth_cordinates[0]+ mouth_cordinates[12]) - (mouth_cordinates[6]+ mouth_cordinates[16]))/2   #(49+61)/2 - (55+65)/2
    mouth_paramaters["h0"] = abs((mouth_cordinates[9]+ mouth_cordinates[18]) - (mouth_cordinates[3]+ mouth_cordinates[14]))/2  #(58+67)/2 - (52+63)/2
    mouth_paramaters["h1"] = abs((mouth_cordinates[19]+ mouth_cordinates[10]) - (mouth_cordinates[2]+ mouth_cordinates[13]))/2  #(68+59)/2 - (51 + 62)/2  
    mouth_paramaters["h2"] = abs((mouth_cordinates[17]+ mouth_cordinates[8]) - (mouth_cordinates[4]+ mouth_cordinates[15]))/2  #(66+57)/2 - (53+64)/2 

    #updae values as one value
    mouth_paramaters["w"] = math.sqrt(mouth_paramaters["w"][0]**2 + mouth_paramaters["w"][1]**2)
    mouth_paramaters["h0"] = math.sqrt(mouth_paramaters["h0"][0]**2 + mouth_paramaters["h0"][1]**2)
    mouth_paramaters["h1"] = math.sqrt(mouth_paramaters["h1"][0]**2 + mouth_paramaters["h1"][1]**2)
    mouth_paramaters["h2"] = math.sqrt(mouth_paramaters["h2"][0]**2 + mouth_paramaters["h2"][1]**2)
   
    return mouth_paramaters

# 0  1   2  3  4  5  6 7  8  9  10 11 12 13 14 15 16 17 18 19
# 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts

        # check if are supposed to draw the jawline
        if name == "Jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    print(facial_features_cordinates)
    print("*********only mouth*************")
    print(facial_features_cordinates["Mouth"][0][0])
    parameters = generateMouthParameters(facial_features_cordinates["Mouth"])
    #return output
    return parameters

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor(shapepredictorPara)


# include the images directory
directory = os.fsencode("vowels")

with open('mouth_parameters.csv', mode='w+') as csv_file:
    fieldnames = ['phoneme', 'w', 'h0','h1','h2']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    #writer.writerow({'w': ([96.5,  4. ]), 'h2': ([ 1.5, 44.5]), 'h0': ([ 0.5, 43.5]), 'h1': ([ 1.5, 43. ])})
    #writer.writerow({'w': ([96.5,  4. ]), 'h2': ([ 1.5, 44.5]), 'h0': ([ 0.5, 43.5]), 'h1': ([ 1.5, 43. ])})
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        dirname = filename.split(".")[0]
        imagePara = "vowels/" + filename
        
        # load the input image, resize it, and convert it to grayscale
        #image = cv2.imread(args["image"])
        image = cv2.imread(imagePara)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)
        mouth_parameters = {}
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = shape_to_numpy_array(shape)

            mouth_parameters = visualize_facial_landmarks(image, shape)
            #output = visualize_facial_landmarks(image, shape)
            #cv2.imshow("Image", output)
            #cv2.waitKey(0)
            mouth_parameters["phoneme"] = filename.split(".")[0]
            print(mouth_parameters)
            writer.writerow(mouth_parameters)
        
                

                
                



