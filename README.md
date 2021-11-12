# Reconstructing High-Quality Face Image from Deep Features  

this is a tensorflow implementation of the paper " Reconstructing High-Quality Face Image from Deep Features "  

## Abstract
Face recognition based on the deep convolutional neural networks (CNN) shows superior accuracy performance attributed to the high discriminative features extracted. Yet, the security and privacy of the extracted features from deep learning models (deep features) have been overlooked, especially reconstructing the high-quality face images from the deep features for malicious activities. In this paper, we formulate the reconstruction of high-quality face images from deep features as a constrained optimization problem. Such optimization aims to minimize the distance between the features extracted from the original face image and the reconstructed high-quality face image. Instead of directly solving the optimization problem in the image space, we reformulate and relax the problem into looking for a latent vector of a GAN generator, then use it to generate the face image. The GAN generator serves a dual role in the proposed framework, i.e., face distribution constraint of the optimization goal and a high-quality face generator. On top of the optimization task, we also propose an attack pipeline to impersonate the target user based on the generated face image. Our results show that the generated face images can achieve a TAR of 98.0\% on LFW under type-I attacks @ FAR of 0.1\%. Our work sheds light on the biometric deployment to meet the privacy-preserving and security policies.

<img src="https://github.com/charlesLucky/auto_decoder_encoder_tf_2/blob/main/data/reconstruction.png" >  

<img src = "https://github.com/charlesLucky/FromDeepFeatures2HQFace/blob/main/data/demo.gif"><img src = "https://github.com/charlesLucky/FromDeepFeatures2HQFace/blob/main/data/demo2%20(1)%20(1).gif">

## Requirements  

Before running the program, make sure that there are dependent packages required by the program in your environment：python3, tensorflow2, numpy, shutil...  



## Pretrained Models  

You can download [styleGan2 pretrained model](https://drive.google.com/drive/folders/1CfeLX2ckWq9NJwm8M0B00_hBdze7NOWq?usp=sharing) from google drive, and down load the feature extractor [pretrained model Resnet-50](https://drive.google.com/drive/folders/1lgBv19VKILyVYrmaLBEpFV5UKJEkilc8?usp=sharing) from google drive  

If you have download limit by google drive, you can also download [styleGan2 pretrained model](https://pan.baidu.com/s/1vOD1gmO5T2aL-WL0ZkgWMg) from baidu drive（Extraction code is cons）, and down load the feature extractor [pretrained model Resnet-50](https://pan.baidu.com/s/1X_7-uxwXX2XRP6JOASOC8g) from baidu drive(Extraction code is cons)


##  Usage（Test with our images）：  

step1.Download pretrained models：StyleGan2 and Resnet50  

step2.Run `python test_ga_result2.py`   

step3. check results in './data/our_results  


##  Usage（Test with your images）：  

if you Download this pretrain models in Usage1, skip step1  

step1.Download pretrained models：StyleGan2 and Resnet50  

step2.Create a new folder and put you pictures in it and create a new folder to save the results  

step3. Modify here in **demo.py**  

```python
img_dir = 'create a new folder, put your pictures in it'
save_dir = 'the path of you save results'
```
step4.Run `python demo.py` and check results in the folder that you create to saving the results
