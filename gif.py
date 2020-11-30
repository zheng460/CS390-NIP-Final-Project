

# Reading an animated GIF file using Python Image Processing Library - Pillow

from PIL import Image

from PIL import GifImagePlugin

 

imageObject = Image.open("./test.gif")

print(imageObject.is_animated)

print(imageObject.n_frames)

 

# Display individual frames from the loaded animated GIF file
i = 0
for frame in range(0,imageObject.n_frames):
    
    i = i+1
    imageObject.seek(frame)
    imageObject.save(f"{i}.png")
    imageObject.show() 
