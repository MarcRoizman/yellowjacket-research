import cv2
import sys
import argparse
from PIL import Image
from PIL import ImageChops
from os import makedirs
from os.path import exists,join
import numpy as np


# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="video file name")
ap.add_argument("-m", "--max", required=True,
	help="max number of diff images")
ap.add_argument("-d", "--diff", required=True,
	help="diff difference (Ending File Number - Starting File Number)")
ap.add_argument("-l", "--location", required=True,
    help="location to save")
args = vars(ap.parse_args())

mp4 = args["video"]
if not exists(mp4):
	print( mp4,'does not exist')
	quit()
max = int(args["max"])
if max == 0: 
	max = float('inf')
diff_skip = int(args["diff"])
diff_save_location = "diff/"+args["location"]

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

#folder = mp4[:-4]
#if not exists(folder):
#	print('Creating folder',folder)
#	makedirs(folder)

vidcap = cv2.VideoCapture(mp4)
success,image = vidcap.read()	# Access first frame of video
count = 0
frame = 0
while (count < max):
	frame += diff_skip
	vidcap.set(1, frame)	# Skip ahead <diff_skip> frames in video
	success2, image2 = vidcap.read()
	if not (success and success2): break
	print('\r',count,type(image),image.shape,image.dtype,end='')
	imagepath = join(diff_save_location + 'DIFF'+five_digit(count) + "_START"+five_digit(frame-diff_skip) + "_END"+five_digit(frame) + ".jpg")
	cv2.imwrite( imagepath, diff(image, image2) )
	count += 1
	success, image = success2, image2

print("\n***"+str(count)+" images printed! ***")
quit()
