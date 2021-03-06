#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
import numpy as np
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # Load preptrained model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path) # TODO Implement 
    # Load graph from pretrained model
    graph = tf.get_default_graph()    
    layer1_out = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return layer1_out, keep, layer3_out, layer4_out, layer7_out 
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Hyperparameters used for tuning
    init_stddev = 0.1
    l2_reg_scale = 0.001
    
    # Freeze output layers of underlying VGG16 pre-trained model
    #vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
    #vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
    #vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
    
    """ 
    VGG layer 7 level 
    """
    # Feed in layer 7 from VGG with a 1x1 convolution
    layer7_fc_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 
                                     1, 
                                     strides=(1,1),
                                     padding='same', 
                                     kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale))
    
    """ 
    VGG layer 4 level 
    """
    # Upsample calculated layer 7 to level 4 size
    layer4_fc_1 = tf.layers.conv2d_transpose(layer7_fc_out, num_classes, 
                                             4, 
                                             strides=(2,2), 
                                             padding='same', 
                                             kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale))
    # Feed in layer 4 from VGG with a 1x1 convolution
    layer4_fc_2 = tf.layers.conv2d(vgg_layer4_out, num_classes,
                                   1,
                                   strides=(1,1),
                                   padding='same',
                                   kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale))
    # Add skip-connection (element-wise addition)
    layer4_fc_out = tf.add(layer4_fc_1, layer4_fc_2)
    
    """ 
    VGG layer 3 level 
    """
    # Upsample calculated layer to dimension of VGG layer 3
    layer3_fc_1 = tf.layers.conv2d_transpose(layer4_fc_out, num_classes,
                                             4,
                                             strides=(2,2),
                                             padding='same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale))
    # Feed in layer 3 from VGG with a 1x1 convolution
    layer3_fc_2 = tf.layers.conv2d(vgg_layer3_out, num_classes,
                                   1,
                                   strides=(1,1),
                                   padding='same',
                                   kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale))   
    # Add skip-connection (element-wise addition)
    layer3_fc_out = tf.add(layer3_fc_1, layer3_fc_2)
    
    """ 
    Upsample to generate very last layer
    """
    layer0_fc_out = tf.layers.conv2d_transpose(layer3_fc_out, num_classes,
                                               16,
                                               strides=(8,8),
                                               padding='same',
                                               kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale))
    # Print debugging output
    #tf.Print(output, [tf.shape(output)[:]])
    # Return final layer
    return layer0_fc_out
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Reshape logits and labels to 2D size
    logits_reshaped = tf.reshape(nn_last_layer, (-1, num_classes))
    labels_reshaped = tf.reshape(correct_label, (-1, num_classes))
    # Define loss function    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_reshaped, 
                                                                                labels=labels_reshaped))
    # Include losses from L2 regularization
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.001
    cross_entropy_loss += reg_constant * sum(reg_losses)
    
    # Use Adam optimizer and cross entropy loss for training and optimization
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    
    return logits_reshaped, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    
    print()
    print('Training for {} epochs is about to start... '.format(epochs))
    for epoch in range(epochs):
        print("Epoch {} ...".format(epoch+1), end='')
        # Create empty array to hold temporary losses
        epoch_losses = []
        for image, label in get_batches_fn(batch_size):
            # feed dictionary, label, keep prob...
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, 
                                          correct_label: label, 
                                          keep_prob: 0.8, 
                                          learning_rate: 0.00025})
            epoch_losses.append(float(loss))
            print('.', end='', flush=True)
        print("\rEpoch {} ".format(epoch+1), end='')
#        print("")
#        print("Epoch {} ".format(epoch+1), end='')
        print("average loss: = {:.3f}".format(np.mean(epoch_losses))) 
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = '/data'
    runs_dir = './runs'
    models_dir = './models'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        
        # Hyper-parameters for training
        epochs = 60
        batch_size = 5
        
        # Create TensorFlow placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # Extract layers from VGG
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = \
            load_vgg(sess, vgg_path)
        
        # Create new layers
        nn_last_layer = \
            layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)  # Final layer we just built
        
        # Create function for loss & optimizer
        logits, train_op, cross_entropy_loss = \
            optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        
        # Saver operation to save and restore all variables
        saver = tf.train.Saver()
        
        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)
#        model_fullpath = models_dir + '/testmodel.
#        save_path = saver.save(sess,models_dir+/testmodel)
#        print("Model saved as file: %s" %save_path)
        
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        
        # Final print statement
        print("Overall training and image output finished!!!")
        print("Script ended.")


if __name__ == '__main__':
    run()
