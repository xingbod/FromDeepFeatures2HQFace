#coding=utf-8

import os
import pickle
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdkfacebody.request.v20191230.DetectLivingFaceRequest import DetectLivingFaceRequest
from viapi.fileutils import FileUtils


client = AcsClient('', '', 'cn-shanghai')
file_utils = FileUtils("", "")
#
# result_path1 = './data/lfw_results0_classify'
# result_path2 = './data/lfw_results1_classify'
# result_path3 = './data/lfw_results2_classify'
#
#
# result_path = [result_path1, result_path2, result_path3]
#
request = DetectLivingFaceRequest()
request.set_accept_format('json')
pass_num = 0
normal_num = 0
review_num = 0
block_num = 0
liveness_num = 0
num = 0

result_list = []
#
# for path in result_path:
#   names_list = os.listdir(path)
#   for name in names_list:
#     pred_img_path = os.path.join(path, name, 'pred_img')
#     pred_img_name = os.listdir(pred_img_path)
#     img = os.path.join(pred_img_path, pred_img_name[0])
#     oss_url = file_utils.get_oss_url(img, "png", True)
#     print(oss_url)
#
#
#     request.set_Taskss([
#       {
#         "ImageURL": oss_url
#       }
#     ])
#
#     response = client.do_action_with_exception(request)
#     # python2:  print(response)
#     # print(str(response, encoding='utf-8'))
#     Ali_result = eval(str(response, encoding='utf-8'))
#     result = Ali_result['Data']['Elements'][0]['Results'][0]
#     result_list.append(result)
#     if result['Suggestion'] == 'pass':
#       pass_num = pass_num + 1
#     elif result['Suggestion'] == 'review':
#       review_num = review_num + 1
#     elif result['Suggestion'] == 'block':
#      block_num = block_num + 1
#     if result['Label'] == 'normal':
#       normal_num = normal_num + 1
#     elif result['Label'] == 'liveness':
#       liveness_num = liveness_num + 1
#     num = num + 1
#
# file = open("./data/living_detect/lfw_Ali.pickle", "wb")
# pickle.dump(result_list, file)
# file.close()
#
# print("total_num", num)
# print(f'pass_num = {pass_num}, review_num = {review_num}, block_num = {block_num}, normal_num = {normal_num},liveness_num = {liveness_num}')


mai_img_path = './data/maiguangcan'
img_names = os.listdir(mai_img_path)
for name in img_names:
  img = os.path.join(mai_img_path, name)
  oss_url = file_utils.get_oss_url(img, "png", True)
  print(oss_url)
  request.set_Taskss([
    {
      "ImageURL": oss_url
    }
  ])

  response = client.do_action_with_exception(request)
  # python2:  print(response)
  # print(str(response, encoding='utf-8'))
  Ali_result = eval(str(response, encoding='utf-8'))
  result = Ali_result['Data']['Elements'][0]['Results'][0]
  result_list.append(result)
  if result['Suggestion'] == 'pass':
    pass_num = pass_num + 1
  elif result['Suggestion'] == 'review':
    review_num = review_num + 1
  elif result['Suggestion'] == 'block':
   block_num = block_num + 1
  if result['Label'] == 'normal':
    normal_num = normal_num + 1
  elif result['Label'] == 'liveness':
    liveness_num = liveness_num + 1
  num = num + 1
print("total_num", num)
print(f'pass_num = {pass_num}, review_num = {review_num}, block_num = {block_num}, normal_num = {normal_num},liveness_num = {liveness_num}')