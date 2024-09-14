import cv2
import os

def play_images(image_dir, delay=30):
  """
  reads the pictures in the specified directory and plays them one by one

  Args:
    image_dir: the path to the directory where the image is stored
    delay: the number of milliseconds to wait after each image playback to control the playback speed
  """

  # get a list of images and sort them by file name
  image_list = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.png', '.bmp'))])

  # cycle through each image
  for image_path in image_list:
    img = cv2.imread(image_path)
    cv2.imshow('Image Viewer', img)

    # press the `q` key to exit
    if cv2.waitKey(delay) & 0xFF == ord('q'):
      break

  cv2.destroyAllWindows()

if __name__ == '__main__':
  import sys
  if len(sys.argv) != 3:
    print("Usage: python script.py <image_directory> <delay>")
    sys.exit(1)

  image_dir = sys.argv[1]
  delay = int(sys.argv[2])
  play_images(image_dir, delay)