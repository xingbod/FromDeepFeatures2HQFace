import os
import numpy as np
import tensorflow as tf

from PIL import Image
from stylegan2.utils import postprocess_images
from load_models import load_generator
from copy_official_weights import convert_official_weights_together
import tqdm


def test_generator(g_clone):


    return


def main():
    from tf_utils import allow_memory_growth

    allow_memory_growth()

    # common variables
    ckpt_dir_base = './official-converted'

    # saving phase
    for use_custom_cuda in [False]:
    # for use_custom_cuda in [True, False]:
        ckpt_dir = os.path.join(ckpt_dir_base, 'cuda') if use_custom_cuda else os.path.join(ckpt_dir_base, 'ref')
        convert_official_weights_together(ckpt_dir, use_custom_cuda)

    # inference phase
    ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')
    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir_cuda, custom_cuda=False)
    seed = 6600
    rnd = np.random.RandomState(seed)
    image_out = np.zeros((50*4, 1024, 1024, 3))
    # 2. inference cuda saved weight from ref model
    for i in tqdm.tqdm(range(50)):
        # test

        latents = rnd.randn(4, 512)
        latents = latents.astype(np.float32)
        image_out1 = g_clone([latents, []], training=False, truncation_psi=0.5)
        image_out[i*4:(i+1)*4,:,:,:] = postprocess_images(image_out1)
    return


if __name__ == '__main__':
    main()
