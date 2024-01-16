import tensorflow as tf

def gram_matrix(x):
    """
    Compute the gram matrix of x

    @param x: Tensor of shape (batch_size, feature_map_height, feature_map_width, num_channels)
    @return: Gram matrix of x (shape=batch_size, num_channels*num_channels)
    """
    batch_size = tf.shape(x)[0]
    feature_map_height = tf.shape(x)[1]
    feature_map_width = tf.shape(x)[2]
    feature_map_size = feature_map_height * feature_map_width
    num_channels = tf.shape(x)[-1]

    x = tf.reshape(x, shape=(batch_size, feature_map_size, num_channels))
    
    feature_map_size = tf.cast(feature_map_size, tf.float32)
    return (tf.transpose(x, perm=(0,2,1)) @ x) / feature_map_size


def clip_0_1(img):
    """
    Clip the pixel values of the image in the range of [0, 1] 

    @param img: Image those pixel values shall be clipped
    @return: Image with clipped pixel values
    """

    return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)


def get_content_loss(content, content_target):
    """
    Get the content loss given the content and the target content

    @param content: content of the image (activations of the network)
    @param content_target: target content of the image (activations of the network)
    @return: content loss
    """

    loss = [tf.reduce_mean((layer_activation_current - layer_activation_target)**2)
                        for layer_activation_current, layer_activation_target in zip(content, content_target)]

    loss = tf.reduce_mean(loss)
    
    return loss
    

def get_style_loss(style, style_target):
    """
    Get the style loss given the style and the target style

    @param style: style of the image (gram matrices)
    @param style_target: target style of the image (gram matrices)
    @return: style loss
    """
    
    style = [gram_matrix(img) for img in style]
        
    loss = [tf.reduce_mean((gram_matrix_current - gram_matrix_target)**2) 
                for gram_matrix_current, gram_matrix_target in zip(style, style_target)]
        
    loss = tf.reduce_mean(loss)
        
    return loss
