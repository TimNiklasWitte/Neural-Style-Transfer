import tensorflow as tf

from Utils import *

class StyleTransferModel(tf.keras.Model):

    def __init__(self):
        """
        Create the StyleTransferModel.
        """

        super(StyleTransferModel, self).__init__()

        # VGG19
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        # Content
        content_layers_name = ['block5_conv2'] 

        self.num_content_layers = len(content_layers_name)

        # Style
        style_layers_name = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1', 
                             'block4_conv1', 
                             'block5_conv1']

        layer_names = content_layers_name + style_layers_name


        outputs = [vgg.get_layer(layer_name).output for layer_name in layer_names]
        
        self.extractor = tf.keras.Model([vgg.input], outputs)


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    @tf.function
    def call(self, x):
        """
        Forward pass

        @param x: Image
        @return: content, style tensors (selected activations of VGG)
        """
        
        x = tf.keras.applications.vgg19.preprocess_input(x*255)

        y = self.extractor(x)
        
        content = y[:self.num_content_layers]
        style = y[self.num_content_layers:]

        return content, style
   

    @tf.function()
    def train_step(self, image, content_target, style_target):
        """
        Update the image

        @param image: Image to be updated based on content_target and style_target
        @param content_target: Target activations of the content image
        @param style_target: Target activations of the style image
        @return: content_loss, style_loss
        """
        with tf.GradientTape() as tape:
            content, style = self(image)

            content_loss = get_content_loss(content, content_target) 
            style_loss = get_style_loss(style, style_target)
            
            loss = content_loss + style_loss


        grad = tape.gradient(loss, image)
        self.optimizer.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

        return content_loss, style_loss

    
   
    
 


