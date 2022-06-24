import math
import numpy as np
import numpy.matlib
from PIL import Image

nr_cycles = 20
A = 0.5
B = 0.5
f = 1
samp_freq = 100
start_time = 0
end_time = nr_cycles*1/f
samp_period = 1/samp_freq

t = np.arange(start_time, end_time+samp_period, samp_period)
y1 = []

for idx in t:
    x = B+A * math.cos(2*math.pi*f*idx)
    y1.append(x)

for idx in range(0, len(y1)):
    if y1[idx] < 0.5:
        y1[idx] = 0
    else:
        y1[idx] = 255

Image_Height = 2160
im1 = np.matlib.repmat(y1, Image_Height, 1)
img = Image.fromarray(im1.astype(np.uint8))

img_resize = img.resize((4096, 2160))
img_resize.save("vertical_pattern_20.png")
img_resize.show()



