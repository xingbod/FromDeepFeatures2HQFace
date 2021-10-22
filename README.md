# Reconstructing High-Quality Face Image from Deep Features

this is a tensorflow implementation of the paper " Reconstructing High-Quality Face Image from Deep Features "

<img src="https://github.com/charlesLucky/auto_decoder_encoder_tf_2/blob/main/data/reconstruction.png" >


## Requirements
Before running the program, make sure that there are dependent packages required by the program in your environment：python3, tensorflow2, numpy, shutil...



## Pretrained Models
You can download styleGan2 pretrained model from google drive..., and down load the feature extractor pretrained model Resnet-50 from google drive...
If you have download limit by google drive, you can also download styleGan2 pretrained model from baidu drive..., and down load the feature extractor pretrained model Resnet-50 from baidu drive...


##  Usage（Test with our images）：
step1.Download pretrained models：StyleGan2 and Resnet50
step2.Run `python test_ga_result2.py`
step3. check results in './data/our_results'


##  Usage（Test with your images）：
if you Download this pretrain models in Usage1, skip step1

step1.Download pretrained models：StyleGan2 and Resnet50.
step2.Create a new folder and put you pictures in it and create a new folder to save the results.
step3. Modify here in **demo.py**
```python
img_dir = 'create a new folder, put your pictures in it'
save_dir = 'the path of you save results'
```
step4.Run `python demo.py` and check results in the folder that you create to saving the results
