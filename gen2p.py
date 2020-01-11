import cv2
import os



dirname = "uu"
os.mkdir(dirname)

vidcap = cv2.VideoCapture("Vowels/uu.mp4")
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