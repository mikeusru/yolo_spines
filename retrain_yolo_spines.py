"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse

import os

# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from data_generator import DataGenerator

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes
import h5py

# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'",
    default=os.path.join('..', 'DATA', 'underwater_data.npz'))

argparser.add_argument(
    '-t',
    '--train',
    help="set training to (default) 'on' or 'off'",
    default='on')

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default=os.path.join('..', 'DATA', 'underwater_classes.txt'))

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))


def _main(args):
    # data_path = os.path.expanduser(args.data_path)
    # data_path = "spine_preprocessing//spine_images_and_boxes.pkl"
    images_path = "spine_preprocessing//spine_images.hdf5"
    boxes_path = "spine_preprocessing//spine_boxes.hdf5"
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    training_on = args.train == 'on'

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    hdf5_file = h5py.File(images_path, "r")
    data_len = hdf5_file["images"].shape[0]
    partition = dict(train=np.array(range(int(0.9 * data_len))),
                     validation=np.array(range(int(0.9 * data_len), data_len)))
    hdf5_file.close()
    anchors = YOLO_ANCHORS

    model_body, model = create_model(anchors, class_names)
    if training_on:
        train(
            model,
            class_names,
            anchors,
            partition,
            images_path,
            boxes_path
        )


    draw(model_body,
         class_names,
         anchors,
         partition,
         images_path,
         image_set='validation',  # assumes training/validation split is 0.9
         weights_name='trained_stage_3_best.h5',
         save_all=False)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS


def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
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
    final_layer = Conv2D(len(anchors) * (5 + len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1,),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
            model_body.output, boxes_input,
            detectors_mask_input, matching_boxes_input
        ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model


def train(model, class_names, anchors, partition, images_path, boxes_path):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''

    def make_data_generators(params):
        training_generator = DataGenerator(partition['train'],
                                           anchors=anchors,
                                           boxes_path=boxes_path,
                                           images_path=images_path,
                                           **params)
        validation_generator = DataGenerator(partition['validation'],
                                             anchors=anchors,
                                             boxes_path=boxes_path,
                                             images_path=images_path,
                                             **params)
        return training_generator, validation_generator

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    params = {'dim': (416, 416),
              'batch_size': 32,
              'n_classes': 1,
              'n_channels': 3,
              'shuffle': True}

    training_generator, validation_generator = make_data_generators(params)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers = 6,
                        epochs=5,
                        callbacks=[logging])

    # model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
    #           np.zeros(len(image_data)),
    #           validation_split=validation_split,
    #           batch_size=32,
    #           epochs=5,
    #           callbacks=[logging])
    model.save_weights('trained_stage_1.h5')

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False)

    model.load_weights('trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    params = {'dim': (416, 416),
              'batch_size': 8,
              'n_classes': 1,
              'n_channels': 3,
              'shuffle': True}

    training_generator, validation_generator = make_data_generators(params)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=4,
                        epochs=30,
                        callbacks=[logging])

    model.save_weights('trained_stage_2.h5')

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=4,
                        epochs=30,
                        callbacks=[logging,checkpoint,early_stopping])

    model.save_weights('trained_stage_3.h5')


def draw(model_body, class_names, anchors, partition, images_path,
         image_set='validation', weights_name='trained_stage_3_best.h5',
         out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''

    # load validation data
    hdf5_file_images = h5py.File(images_path, "r")
    image_data = hdf5_file_images["images"][partition[image_set],...]
    hdf5_file_images.close()

    image_data = np.expand_dims(image_data, axis=1)

    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2,))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.5, iou_threshold=0.5)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if not os.path.exists(out_path):
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
        image_with_boxes = draw_boxes(image_data[i][0].astype(np.uint8)*255, out_boxes, out_classes,
                                      class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path, str(i) + '.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()


if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
