import os
from tensorflow.python.platform import gfile
import numpy as np
import sys

# image_dir = './train'
# file_list = []
# file_glob = os.path.join(image_dir, '*.jpg')
# file_list.extend(gfile.Glob(file_glob))
# print (file_list)

print (sys.argv)
for a in sys.argv:
	print (a)