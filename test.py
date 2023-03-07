from PIL import Image
import numpy as np

# read wolf_2.png
img = Image.open("wolf_2.png")

# convert to grey scale
img = img.convert("L")

# convert to numpy array
img = np.array(img)

# make all pixel above 200 white
img[img > 128] = 255
# make all pixel below 200 black
img[img < 128] = 0


# convert back to image
img = Image.fromarray(img)

# save
img.save("wolf_2_.png")
