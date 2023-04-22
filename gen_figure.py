from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def show_image(data,name):
  # plt.imshow(data, cmap='gray')
  # plt.show()
  # plt.savefig(f'images/{name}.png',dpi=1000)

  Image.fromarray(data).save(f'images/{name}.png')



   
def gen_image_1():
	image_array = np.zeros((16, 16), dtype=np.uint8)
	image_array[0:8, 0:2] = 1
	image_array[8:16, 8:16] = 1
	image_array[0:2, 8:12] = 1
	return image_array

def gen_image_2():
  image_array = np.zeros((16, 16), dtype=np.uint8)
  image_array[0:2, 0:8] = 1
  image_array[2:9, 6:8] = 1
  image_array[7:9, 0:8] = 1
  image_array[4:7, 0:2] = 1
  image_array[14:16, 0:5] = 1
  image_array[0:2, 12:16] = 1
  image_array[5:12, 12:16] = 1
  image_array[14:16, 8:16] = 1
  image_array[12:14, 8:10] = 1
  return image_array

def gen_image_3():
  image_array = np.zeros((16, 16), dtype=np.uint8)
  image_array[0:4, 0:4] = 1
  image_array[0:4, 12:16] = 1
  image_array[0:7, 7:9] = 1
  image_array[8:9, 0:5] = 1
  image_array[5:9, 5:6] = 1
  image_array[5:6, 4:5] = 1
  image_array[5:9, 10:11] = 1
  image_array[5:6, 11:12] = 1
  image_array[8:9, 11:16] = 1
  image_array[10:16, 2:3] = 1
  image_array[10:11, 3:4] = 1
  image_array[10:16, 13:14] = 1
  image_array[10:16, 5:11] = 1
  return image_array



# show_image(gen_image_1(),'image1')
# show_image(gen_image_2(),'image2')
# show_image(gen_image_3(),'image3')
# show_image(gen_image_4(),'image4')
def convert_image_to_binary(path):
  image = Image.open(path)
  image.convert('1').save('images/image4.png')

convert_image_to_binary('images/bolhas.png')