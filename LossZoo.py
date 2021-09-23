from tensorflow.keras import losses
import tensorflow as tf
from tensorflow.keras import backend as K

def perceptual_loss(y_true, y_pred):
    #     print("Note:Need to remove vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 to C:\\Users\\user_name\\.keras\\models\\")
    vgg_inp = tf.keras.Input([112, 112, 3])
    vgg = tf.keras.applications.VGG19(include_top=False, input_tensor=vgg_inp)
    for l in vgg.layers: l.trainable = False
    vgg_out_layer = vgg.get_layer(index=5).output
    vgg_content = tf.keras.Model(vgg_inp, vgg_out_layer)
    #
    # y_true = tf.tile(y_true, [1, 1, 1, 3])
    # y_pred = tf.tile(y_pred, [1, 1, 1, 3])
    #     print("***************",y_true)

    y_t = vgg_content(y_true)
    y_p = vgg_content(y_pred)
    loss = tf.keras.losses.mean_squared_error(y_t, y_p)
    return tf.reduce_mean(loss)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true)) / 255 ** 2) / K.log(10.)

'''
20210817 xingbo add new loss on HSV
'''
def HSVLoss(y_true, y_pred):
    '''
    input must be in range [0,1]
output[..., 0] contains hue, output[..., 1] contains saturation, and output[..., 2] contains value.
All HSV values are in [0,1]. A hue of 0 corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.
    '''
    hsv_y_true = tf.image.rgb_to_hsv(y_true)
    #Outputs a tensor of the same shape as the images tensor, containing the HSV value of the pixels. The output is only well defined if the value in images are in [0,1].
    hsv_y_pred = tf.image.rgb_to_hsv(y_pred)
    return K.mean(K.square(hsv_y_true - hsv_y_pred))

# def PSNRLoss(y_true, y_pred):
#     return 10. * K.log(255**2/mean_squared_error(y_pred , y_true))/ K.log(10.)

# def customized_loss(y_true, y_pred, lamada=0.0):
#     return losses.mean_absolute_error(y_pred, y_true) + lamada * perceptual_loss(y_pred, y_true)
def customized_loss(lamda = 0.0):
    """softmax loss"""
    print('using lamda',lamda)
    def softmax_loss(y_true, y_pred):
        return losses.mean_absolute_error(y_pred, y_true) + lamda * perceptual_loss(y_pred, y_true)
    return softmax_loss


img1 = tf.random.uniform(shape=[2,512,512,3])
img2 = tf.random.uniform(shape=[2,512,512,3])
# perceptual_loss(img1, img2)
lossv = HSVLoss(img1, img2)
print(lossv)
