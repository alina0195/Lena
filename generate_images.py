
from PIL import Image
import numpy as np


host_image = Image.open("./Lena.tiff")
# Resize the image to 512x512 pixels
resized_host_image = host_image.resize((512, 512), Image.ANTIALIAS)

# Save the resized image
resized_host_image.save("Lena_512x512.png")
