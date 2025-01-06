
from PIL import Image
import numpy as np

# Generate a random 32x32x3 color image (RGB)
logo_color_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

# Create an image from the array
logo_color_image = Image.fromarray(logo_color_image, mode='RGB')
logo_color_image.save("logo.png")


host_image = Image.open("./Lena.tiff")
# Resize the image to 512x512 pixels
resized_host_image = host_image.resize((512, 512), Image.ANTIALIAS)

# Save the resized image
resized_host_image.save("Lena_512x512.png")
