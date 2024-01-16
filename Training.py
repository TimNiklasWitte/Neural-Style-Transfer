import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
import datetime

from StyleTransferModel import *
from Utils import *


def load_img(path):
    """
    Load the image from a file and rescale it.
    Besides, a batch dimension is added.
    The pixel values are in the range of [0, 1].
    
    @param path: Path to the image
    """
    
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # Rescaling
    max_dim = 512
    shape = tf.shape(img)[:-1] # ignore color channels
    shape = tf.cast(shape, tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)

    # Add batch dim
    img = tf.expand_dims(img, axis=0)
    return img


def main():

    #
    # Logging
    #
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/init_random/{current_time}"
    summary_writer = tf.summary.create_file_writer(file_path)

    content_img = load_img("./images/Garden.jpg")
    style_img = load_img("./images/Night Alley by Leonid Afremov.jpg")

    styleTransferModel = StyleTransferModel()

    #
    # Get targets
    #
    content_target, _ = styleTransferModel(content_img)
    _, style_target = styleTransferModel(style_img)

    style_target = [gram_matrix(layer_activation) for layer_activation in style_target]
    

    image = tf.Variable(tf.random.normal(shape=content_img.shape))
    #image = tf.Variable(content_img)

    #
    # Inital losses
    #
    
    content, style = styleTransferModel(content_img)
 
    content_loss = get_content_loss(content, content_target) 
    style_loss = get_style_loss(style, style_target)
 
    loss = content_loss + style_loss
    with summary_writer.as_default():
         
        tf.summary.scalar(name="Content loss", data=content_loss, step=0)
        tf.summary.scalar(name="Style loss", data=style_loss, step=0)
        tf.summary.scalar(name="Loss", data=loss, step=0)

        tf.summary.image(name="Image", data = image, step=0, max_outputs=1)
    
    #
    # Training: Update image
    #
    NUM_STEPS = 20000
    LOG_INTERVAL = 100
    for step in tqdm.tqdm(range(1, NUM_STEPS + 1), position=0, leave=True):
    
        content_loss, style_loss = styleTransferModel.train_step(image, content_target, style_target)

        # Monitor: Style loss, content loss, total loss and current image
        if step % LOG_INTERVAL == 0:
            with summary_writer.as_default():
                loss = content_loss + style_loss
                tf.summary.scalar(name="Content loss", data=content_loss, step=step)
                tf.summary.scalar(name="Style loss", data=style_loss, step=step)
                tf.summary.scalar(name="Loss", data=loss, step=step)

                tf.summary.image(name="Image", data = image, step=step, max_outputs=1)
 
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")