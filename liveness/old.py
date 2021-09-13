'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
#!/usr/bin/env python
#coding=utf-8

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdkfacebody.request.v20191230.DetectLivingFaceRequest import DetectLivingFaceRequest
# pip install aliyun-python-sdk-facebody==1.2.21
import json
import base64

#
client = AcsClient('LTAISsjYyR8wFDps', 'cLzw9UVuOADqjOZ614HMufVffl9Kyc', 'cn-shanghai')

request = DetectLivingFaceRequest()
request.set_accept_format('json')




request.set_Taskss([
  {
    "ImageURL": "https://img-blog.csdnimg.cn/20210313131228678.png",

  },
    {
    "ImageURL": "https://img-blog.csdnimg.cn/20210313131228678.png",
    }
])


response = client.do_action_with_exception(request)
# python2:  print(response)
# parse json file
# response = '{"RequestId":"F82E41B0-FC96-43B5-8E22-F63C0044C72D","Data":{"Elements":[{"TaskId":"img7$FCTGfO4DI6vNvkORerdz-1unV0p","Results":[{"Suggestion":"pass","Rate":30.050003,"Label":"normal"}],"ImageURL":"https://img-blog.csdnimg.cn/20210313131228678.png"},{"TaskId":"img5Qfzjr76k7L7wVPy0Ve17Y-1unV0p","Results":[{"Suggestion":"pass","Rate":30.050003,"Label":"normal"}],"ImageURL":"https://img-blog.csdnimg.cn/20210313131228678.png"}]}}'
pythonObj = json.loads(str(response, encoding='utf-8'))
# pythonObj = json.loads(response)
print(str(response, encoding='utf-8'))
samples = len(pythonObj['Data']['Elements'])
for i in range(samples):
    res_obj = pythonObj['Data']['Elements'][i]#{'TaskId': 'img7$FCTGfO4DI6vNvkORerdz-1unV0p', 'Results': [{'Suggestion': 'pass', 'Rate': 30.050003, 'Label': 'normal'}], 'ImageURL': 'https://img-blog.csdnimg.cn/20210313131228678.png'}
    rate_i = res_obj['Rate']