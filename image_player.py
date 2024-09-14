import cv2
import os

def play_images(image_dir, delay=30):
  """
  读取指定目录下的图片，并逐张播放。

  Args:
    image_dir: 存放图片的目录路径。
    delay: 每次播放图片后等待的毫秒数，用于控制播放速度。
  """

  # 获取图片列表，并按文件名排序
  image_list = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.png', '.bmp'))])

  # 循环播放每张图片
  for image_path in image_list:
    img = cv2.imread(image_path)
    cv2.imshow('Image Viewer', img)

    # 按下 'q' 键退出
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