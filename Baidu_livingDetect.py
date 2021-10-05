# encoding:utf-8

import requests
import os
import pickle
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdkfacebody.request.v20191230.DetectLivingFaceRequest import DetectLivingFaceRequest
from viapi.fileutils import FileUtils
'''
在线活体检测
'''



client = AcsClient('LTAI5tNR2KC6TgCZDoBbvaAP', 'Ds8BPUqWwDJWgsMwLW4mWCMVnX55mM', 'cn-shanghai')
file_utils = FileUtils("LTAI5tNR2KC6TgCZDoBbvaAP", "Ds8BPUqWwDJWgsMwLW4mWCMVnX55mM")

result_path1 = './data/lfw_results0_classify'
result_path2 = './data/lfw_results1_classify'
result_path3 = './data/lfw_results2_classify'


result_path = [result_path1, result_path2, result_path3]

request = DetectLivingFaceRequest()
request.set_accept_format('json')
pass_num = 0
normal_num = 0
review_num = 0
block_num = 0
liveness_num = 0
num = 0
no_spoofing_num = 0

result_list = []
liveness_scores = []
spoofing_scores = []

request_url = "https://aip.baidubce.com/rest/2.0/face/v3/faceverify"


for path in result_path:
  names_list = os.listdir(path)
  for name in names_list:
    pred_img_path = os.path.join(path, name, 'pred_img')
    pred_img_name = os.listdir(pred_img_path)
    img = os.path.join(pred_img_path, pred_img_name[0])
    oss_url = file_utils.get_oss_url(img, "png", True)
    print(oss_url)
    params = '[{"image":"'+oss_url+'","image_type":"URL","face_field":"spoofing"}]'
    access_token = '24.c7fae325e59b166cd9fa3d287852ba76.2592000.1635993926.282335-24943734'
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/json'}
    response = requests.post(request_url, data=params, headers=headers)
    liveness_score = response.json()["result"]['face_liveness']
    liveness_scores.append(liveness_score)
    spoofing_score = response.json()["result"]['face_list'][0]['spoofing']
    spoofing_scores.append(spoofing_score)
    num = num + 1
    if liveness_score >= 0.3:
      liveness_num = liveness_num + 1
    if spoofing_score < 0.00048:
      no_spoofing_num = no_spoofing_num + 1
    if response:
        print ('liveness_score = ', liveness_score)
        print('spoofing_score = ', spoofing_score)
        print('total_num = ', num)
print('liveness_num = ', liveness_num)
print('no_spoofing_num = ', no_spoofing_num)

file = open("./data/living_detect/lfw_Baidu_liveness_score.pickle", "wb")
pickle.dump(liveness_scores, file)
file.close()

f = open("./data/living_detect/lfw_Baidu_spoofing_score.pickle", "wb")
pickle.dump(spoofing_scores, f)
f.close()