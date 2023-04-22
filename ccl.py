"""
	Connected Component Labelling algorithm.
	Jack Lawrence-Jones, July 2016

	For blob/connected component detection in binary images. Labels each pixel in a given connected component 
	with the same label.

	2 pass implementation using disjoint-set data structure with Union-Find algorithms to record 
	label equivalences.

	O(n) for image containing n pixels. 

	Usage:
		Python:
			>>> image = Image.open("./binary_image.png")
			>>> bool_image = image_to_2d_bool_array(image)
			>>> result = connected_component_labelling(bool_image, 4)

		Terminal (second parameter is connectivity type):
			$  python ccl.py path/to/image.png 4
"""

import numpy as np
from PIL import Image
from union_find import UnionFind
import matplotlib.pyplot as plt
import cv2


CONNECTIVITY_4 = 4
CONNECTIVITY_8 = 8


def connected_component_labelling(bool_input_image, connectivity_type=CONNECTIVITY_8):
  """
          2 pass algorithm using disjoint-set data structure with Union-Find algorithms to maintain 
          record of label equivalences.

          Input: binary image as 2D boolean array.
          Output: 2D integer array of labelled pixels.

          1st pass: label image and record label equivalence classes.
          2nd pass: replace labels with their root labels.

          (optional 3rd pass: Flatten labels so they are consecutive integers starting from 1.)

  """
  if connectivity_type != 4 and connectivity_type != 8:
    raise ValueError("Invalid connectivity type (choose 4 or 8)")

  image_width = len(bool_input_image[0])
  image_height = len(bool_input_image)

  # initialise efficient 2D int array with numpy
  # N.B. numpy matrix addressing syntax: array[y,x]
  labelled_image = np.zeros((image_height, image_width), dtype=np.int16)
  uf = UnionFind()  # initialise union find data structure
  current_label = 1  # initialise label counter

  # 1st Pass: label image and record label equivalences
  for y, row in enumerate(bool_input_image):
    for x, pixel in enumerate(row):

      if pixel == False:
        # Background pixel - leave output pixel value as 0
        pass
      else:
        # Foreground pixel - work out what its label should be

        # Get set of neighbour's labels
        labels = neighbouring_labels(labelled_image, connectivity_type, x, y)

        if not labels:
          # If no neighbouring foreground pixels, new label -> use current_label
          labelled_image[y, x] = current_label
          uf.MakeSet(current_label)  # record label in disjoint set
          current_label = current_label + 1  # increment for next time

        else:
          # Pixel is definitely part of a connected component: get smallest label of
          # neighbours
          smallest_label = min(labels)
          labelled_image[y, x] = smallest_label

          if len(labels) > 1:  # More than one type of label in component -> add
              # equivalence class
            for label in labels:
              uf.Union(uf.GetNode(smallest_label), uf.GetNode(label))

  # 2nd Pass: replace labels with their root labels
  final_labels = {}
  new_label_number = 1

  for y, row in enumerate(labelled_image):
    for x, pixel_value in enumerate(row):

      if pixel_value > 0:  # Foreground pixel
        # Get element's set's representative value and use as the pixel's new label
        new_label = uf.Find(uf.GetNode(pixel_value)).value
        labelled_image[y, x] = new_label

        # Add label to list of labels used, for 3rd pass (flattening label list)
        if new_label not in final_labels:
          final_labels[new_label] = new_label_number
          new_label_number = new_label_number + 1

  # 3rd Pass: flatten label list so labels are consecutive integers starting from 1 (in order
  # of top to bottom, left to right)
  # Different implementation of disjoint-set may remove the need for 3rd pass?
  for y, row in enumerate(labelled_image):
    for x, pixel_value in enumerate(row):
      if pixel_value > 0:  # Foreground pixel
        labelled_image[y, x] = final_labels[pixel_value]
  return labelled_image


# Private functions ############################################################################
def neighbouring_labels(image, connectivity_type, x, y):
  """
          Gets the set of neighbouring labels of pixel(x,y), depending on the connectivity type.

          Labelling kernel (only includes neighbouring pixels that have already been labelled - 
          row above and column to the left):

                  Connectivity 4:
                              n
                           w  x  

                  Connectivity 8:
                          nw  n  ne
                           w  x   
  """

  labels = set()

  if (connectivity_type == CONNECTIVITY_4) or (connectivity_type == CONNECTIVITY_8):
    # West neighbour
    if x > 0:  # Pixel is not on left edge of image
      west_neighbour = image[y, x-1]
      if west_neighbour > 0:  # It's a labelled pixel
        labels.add(west_neighbour)

    # North neighbour
    if y > 0:  # Pixel is not on top edge of image
      north_neighbour = image[y-1, x]
      if north_neighbour > 0:  # It's a labelled pixel
        labels.add(north_neighbour)

    if connectivity_type == CONNECTIVITY_8:
      # North-West neighbour
      if x > 0 and y > 0:  # pixel is not on left or top edges of image
        northwest_neighbour = image[y-1, x-1]
        if northwest_neighbour > 0:  # it's a labelled pixel
          labels.add(northwest_neighbour)

      # North-East neighbour
      # Pixel is not on top or right edges of image
      if y > 0 and x < len(image[y]) - 1:
        northeast_neighbour = image[y-1, x+1]
        if northeast_neighbour > 0:  # It's a labelled pixel
          labels.add(northeast_neighbour)
  else:
    print("Connectivity type not found.")

  return labels


def print_image(image):
  """ 
          Prints a 2D array nicely. For debugging.
  """
  for y, row in enumerate(image):
    print(row)


def image_to_2d_bool_array(image):
  im2 = image.convert('L')
  arr = np.asarray(im2)
  arr = arr != 255
  return arr


def show_pretty_image(labels, bitwise=False):
  b = np.max(labels) if np.max(labels) > 0 else 1
  label_hue = np.uint8(179*labels/b)
  blank_ch = 255*np.ones_like(label_hue)
  labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
  labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
  labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2RGB)
  _, labeled_img = cv2.threshold(labeled_img, 70, 255, 0)
  labeled_img[label_hue == 0] = 0
  if bitwise:
    labeled_img = cv2.bitwise_not(labeled_img)
  return labeled_img


def get_cmap(result):
  cmap = plt.cm.get_cmap('hot', np.max(result) + 1)
  cmap.set_gamma(0.3)
  return cmap


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


def connected_component_label_cv(path, image=None):
  if image is None:
    img = cv2.imread(path, 0)
  else:
    img = image
  # img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
  _, labels = cv2.connectedComponents(img)
  return labels


if __name__ == "__main__":
  import sys
  if len(sys.argv) > 1:  # At least 1 command line parameter
    image_path = str(sys.argv[1])
    if(len(sys.argv) > 2):  # At least 2
      connectivity_type = int(sys.argv[2])
    else:
      connectivity_type = CONNECTIVITY_8
    image = Image.open(image_path)

    fig, axs = plt.subplots(1, 3, constrained_layout=True)

    bool_image = image_to_2d_bool_array(image)
    # bool_image = gen_image_3()
    # bool_image = cv2.imread(image_path, 0)

    axs[0].imshow(bool_image, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('Original')

    cll_result = connected_component_labelling(bool_image, connectivity_type)
    # connected_component_label(image_path)
    cv_result = connected_component_label_cv("", bool_image.astype(np.uint8))
    # print_image(result)

    axs[1].imshow(cll_result, cmap=get_cmap(cll_result))
    axs[1].axis('off')
    axs[1].set_title('UnionFind')

    axs[2].imshow(cv_result, cmap=get_cmap(cv_result))
    axs[2].axis('off')
    axs[2].set_title("Spaghetti")

    # for i in range(labels.shape[0]):
    # 		for j in range(labels.shape[0]):
    # 			c1 = cll_result[j, i]
    # 			c2 = cv-result[j, i]
    # 			axs[2].text(i, j, str(c2), va='center', ha='center')
    # 			axs[1].text(i, j, str(c1), va='center', ha='center')

    # plt.savefig('result/result7.png', dpi=1000)
    plt.show()


# Run in Python
# image = Image.open("./images/second_pass.png")
# bool_image = image_to_2d_bool_array(image)
# output = connected_component_labelling(bool_image, CONNECTIVITY_4)
# print(output)
