import cv2
import os


directory = os.fsencode("Vowels")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    dirname = filename.split(".")[0]
    os.mkdir(dirname)
    print(filename)
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()
    count = 0
    #print(file)
    while success:
        #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file  
        cv2.imwrite(os.path.join(dirname, "frame%d.jpg" % count), image)    
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        success,image = vidcap.read()
        count += 1
        #print(filename)
    vidcap.release()
    cv2.destroyAllWindows()
    