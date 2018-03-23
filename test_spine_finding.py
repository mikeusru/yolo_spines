import argparse

import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def _main():
    anchors = YOLO_ANCHORS
    class_names = os.path.normpath('data/spine_classes.txt')
    model_body = create_model_body(anchors, class_names)
    image = np.array(PIL.Image.open('spines//spine_image000057.tif'), dtype = np.uint8)
    image = np.resize(image,(416,416))
    image_data = np.array([image])
    draw(model_body,
         class_names,
         anchors,
         image_data,
         out_path = 'images/spines/out',
         weights_name='h5_training_files/3_21_allSpines/trained_stage_3_best.h5',
         save_all=False)

def create_model_body(anchors, class_names, load_pretrained=True, freeze_body=True):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)
    return model_body
    #
    # # Place model loss on CPU to reduce GPU memory usage.
    # with tf.device('/cpu:0'):
    #     # TODO: Replace Lambda with custom Keras layer for loss.
    #     model_loss = Lambda(
    #         yolo_loss,
    #         output_shape=(1, ),
    #         name='yolo_loss',
    #         arguments={'anchors': anchors,
    #                    'num_classes': len(class_names)})([
    #                        model_body.output, boxes_input,
    #                        detectors_mask_input, matching_boxes_input
    #                    ])
    #
    # model = Model(
    #     [model_body.input, boxes_input, detectors_mask_input,
    #      matching_boxes_input], model_loss)
    #
    # return model_body

def draw(model_body, class_names, anchors,
         image_data, weights_name='trained_stage_3_best.h5',
         out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0.0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))

        # To display (pauses the program):
        plt.imshow(image_with_boxes, interpolation='nearest')
        plt.show()

if __name__ == '__main__':
    _main()
