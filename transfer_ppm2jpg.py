import os
import cv2
from PIL import Image
from mtcnn import MTCNN
from tf_utils import allow_memory_growth
import tqdm
# transfer colorferet img from ppm to jpg
allow_memory_growth()
people_name_list = os.listdir('./data/colorferet_158')
jpg_save_path = './data/colorferet_jpg'

# for people_name in people_name_list:
#     img_source_path = os.path.join('./data/colorferet_158', people_name)
#     img_save_path = os.path.join(jpg_save_path, people_name)
#     if not os.path.exists(img_save_path):
#         os.mkdir(img_save_path)
#     img_name_list = os.listdir(img_source_path)
#     for img_name in img_name_list:
#         filename, extension = os.path.splitext(img_name)
#         img = Image.open(img_source_path + f'/{filename}.ppm')
#         img.save(img_save_path + f'/{filename}.jpg')


# resize the colorferet jpg img
people_name_list_jpg = os.listdir(jpg_save_path)
jpg_crop_save_path = './data/colorferet_jpg_crop'
detector = MTCNN()
for people_name in tqdm.tqdm(people_name_list_jpg):
    img_source_path = os.path.join(jpg_save_path, people_name)
    img_save_path = os.path.join(jpg_crop_save_path, people_name)
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
    img_name_list = os.listdir(img_source_path)
    for img_name in img_name_list:
        filename, extension = os.path.splitext(img_name)
        # img = Image.open(img_source_path + f'/{filename}.jpg')
        # mtcnn detect and resize
        # img = cv2.cvtColor(cv2.imread(img_source_path + f'/{filename}.jpg'), cv2.COLOR_BGR2RGB)
        img = cv2.imread(img_source_path + f'/{filename}.jpg')
        result = detector.detect_faces(img)
        if result is None:
            continue
        if len(result) ==0:
            continue
        try:
            x, y, width, height = bounding_box = result[0]['box']  # s [x, y, width, height] under the key ‘box’.
            x, y, width, height = x - 60, y - 60, width + 120, height + 120
            keypoints = result[0]['keypoints']
            imgnew = img[y:y + height, x:x + width]
            cv2.imwrite(img_save_path + f'/{filename}.jpg', imgnew)
        except:
            print(img_source_path + f'/{filename}.jpg','error')
            pass

