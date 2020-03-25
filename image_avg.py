import cv2
import sys
import argparse
from PIL import Image
from PIL import ImageChops
from os import makedirs
from os.path import exists,join
import numpy as np


# Compute diff of 2 images
def diff(a,b):
    a1  = a.astype(np.int64)
    b1  = b.astype(np.int64)
    d = np.abs(a1 - b1)
    return d.astype(np.uint8)

# Translate int index into 4 digit string
def five_digit(index):
	if index < 10:
		return ("0000"+str(index))
	elif index < 100:
		return ("000"+str(index))
	elif index < 1000:
		return ("00"+str(index))
	elif index < 10000:
		return ("0"+str(index))
	else:
		return (str(index))

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="video file name")
ap.add_argument("-m", "--max", required=True,
	help="max number of diff images")
ap.add_argument("-d", "--diff", required=True,
	help="difference between successive images")
ap.add_argument("-l", "--location", required=True,
    help="location to save")
args = vars(ap.parse_args())

mp4 = args["video"]
max = int(args["max"])
diff_skip = int(args["diff"])
diff_save_location = "images/"+args["location"]

if max == 0:
	max = float('inf')

if not exists(mp4):
	print( mp4,'does not exist')
	quit()

# Capture frames from video
vidcap = cv2.VideoCapture(mp4)
success, image =  vidcap.read()

# Compute average frame
sum = image.astype(np.float64)
count = 1
frame = 0
while (count < 100):
    print('\r',"Computing Average Image: {0}/{1} ({2}%)".format(count, 100, round(100*count/100),1), end='')
    frame += diff_skip
    vidcap.set(1, frame)	# Skip ahead <diff_skip> frames in video
    success, image = vidcap.read()
    if not (success): break
    sum += image.astype(np.float64)
    count += 1
avg = (sum/count).astype(np.uint8)
cv2.imwrite( diff_save_location + 'AVG.jpg', avg )


# Iterate through to create diffs
count = 0
frame = 0
vidcap.set(1, 0)
while (count < max):
    success, image = vidcap.read()
    if not (success): break
    print('\r',count,type(image),image.shape,image.dtype,end='')
    imagepath = join(diff_save_location + 'DIFF'+five_digit(count) + "_FRAME"+five_digit(frame) + ".jpg")
    new_img = diff(image, avg)
    cv2.imwrite( imagepath, new_img )
    frame += diff_skip
    vidcap.set(1, frame)	# Skip ahead <diff_skip> frames in video
    count += 1

print("\n***"+str(count)+" images printed! ***")
quit()
