"""
This is a class for training and evaluating yadk2
"""
import argparse
import os
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from data_generator import DataGenerator

from yad2k.models.keras_yolo import (yolo_body,
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
    '-s',
    '--starting_weights',
    help="path for starting weights",
    default='trained_stage_3_best.h5')

argparser.add_argument(
    '-f',
    '--from_scratch',
    help="start training model using old yolo training data (on/off)",
    default='off')

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

class SpineYolo(object):

    def __init__(self, _args):
        self.images_path = "spine_preprocessing//spine_images.hdf5"
        self.boxes_path = "spine_preprocessing//spine_boxes.hdf5"
        self.classes_path = os.path.expanduser(_args.classes_path)
        self.anchors_path = os.path.expanduser(_args.anchors_path)
        self.starting_weights = os.path.expanduser(_args.starting_weights)
        self.from_scratch = _args.from_scratch == 'on'
        self.training_on = _args.train == 'on'
        self.class_names = self.get_classes()
        self.anchors = self.get_anchors()
        self.partition = self.get_partition()
        self.detectors_mask_shape = (13, 13, 5, 1)
        self.matching_boxes_shape = (13, 13, 5, 5)
        self.model_body = None
        self.model = None
        if self.training_on:
            self.create_model()
            self.train()
            self.draw(image_set='validation',  # assumes training/validation split is 0.9
                      weights_name='trained_stage_3_best.h5',
                      save_all=False)
        else:
            self.draw(test_model_path='model_data//yolo_spine_model_testing.h5',
                      image_set='validation',  # assumes training/validation split is 0.9
                      save_all=False)

    def get_partition(self):
        hdf5_file = h5py.File(self.images_path, "r")
        data_len = hdf5_file["images"].shape[0]
        partition = dict(train=np.array(range(int(0.9 * data_len))),
                         validation=np.array(range(int(0.9 * data_len), data_len)))
        hdf5_file.close()
        return partition

    def get_classes(self):
        """loads the classes"""
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        """loads the anchors from a file"""
        if os.path.isfile(self.anchors_path):
            with open(self.anchors_path) as f:
                anchors = f.readline()
                anchors = [float(x) for x in anchors.split(',')]
                return np.array(anchors).reshape(-1, 2)
        else:
            Warning("Could not open anchors file, using default.")
            return YOLO_ANCHORS

    def create_model(self, load_pretrained=True, freeze_body=True):
        """
        returns the body of the model and the model

        # Params:

        load_pretrained: whether or not to load the pretrained model or initialize all weights

        freeze_body: whether or not to freeze all weights except for the last layer's

        # Returns:

        model_body: YOLOv2 with new output layer

        model: YOLOv2 with custom loss Lambda layer

        """

        # Create model input layers.
        image_input = Input(shape=(416, 416, 3))
        boxes_input = Input(shape=(None, 5))
        detectors_mask_input = Input(shape=self.detectors_mask_shape)
        matching_boxes_input = Input(shape=self.matching_boxes_shape)

        # Create model body.
        yolo_model = yolo_body(image_input, len(self.anchors), len(self.class_names))
        topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

        if load_pretrained:
            # Save topless yolo:
            topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
            if not os.path.exists(topless_yolo_path):
                print("CREATING TOPLESS WEIGHTS FILE")
                yolo_path = os.path.join('model_data', 'yolo.h5')
                self.model_body = load_model(yolo_path)
                self.model_body = Model(self.model_body .inputs, self.model_body .layers[-2].output)
                self.model_body.save_weights(topless_yolo_path)
            topless_yolo.load_weights(topless_yolo_path)

        if freeze_body:
            for layer in topless_yolo.layers:
                layer.trainable = False
        final_layer = Conv2D(len(self.anchors) * (5 + len(self.class_names)), (1, 1), activation='linear')(
            topless_yolo.output)

        model_body = Model(image_input, final_layer)

        # Place model loss on CPU to reduce GPU memory usage.
        with tf.device('/cpu:0'):
            # TODO: Replace Lambda with custom Keras layer for loss.
            model_loss = Lambda(
                yolo_loss,
                output_shape=(1,),
                name='yolo_loss',
                arguments={'anchors': self.anchors,
                           'num_classes': len(self.class_names)})([
                            model_body.output, boxes_input,
                            detectors_mask_input, matching_boxes_input])

        self.model = Model(
            [model_body.input, boxes_input, detectors_mask_input,
             matching_boxes_input], model_loss)



    def train(self):
        """
        retrain/fine-tune the model

        logs training with tensorboard

        saves training weights in current directory

        best weights according to val_loss is saved as trained_stage_3_best.h5
        """

        self.model.compile(
            optimizer='adam', loss={
                'yolo_loss': lambda y_true, y_pred: y_pred
            })  # This is a hack to use the custom loss function in the last layer.

        logging = TensorBoard()
        checkpoint_final_best = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                     save_weights_only=True, save_best_only=True)
        checkpoint = ModelCheckpoint("trained_checkpoint.h5", monitor='val_loss',
                                     save_weights_only=True, save_best_only=True)

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

        params = {'dim': (416, 416),
                  'batch_size': 32,
                  'n_classes': 1,
                  'n_channels': 3,
                  'shuffle': True}

        training_generator, validation_generator = self.make_data_generators(params)

        if not self.from_scratch:
            self.model.load_weights(self.starting_weights)

        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 use_multiprocessing=True,
                                 workers=6,
                                 epochs=5,
                                 callbacks=[logging, checkpoint])

        self.model.save_weights('trained_stage_1.h5')

        self.draw(image_set='validation', weights_name='trained_stage_1.h5',
                  out_path="output_images_stage_1", save_all=False)

        self.create_model(load_pretrained=False, freeze_body=False)

        self.model.load_weights('trained_stage_1.h5')

        self.model.compile(
            optimizer='adam', loss={
                'yolo_loss': lambda y_true, y_pred: y_pred
            })  # This is a hack to use the custom loss function in the last layer.

        params = {'dim': (416, 416),
                  'batch_size': 16,
                  'n_classes': 1,
                  'n_channels': 3,
                  'shuffle': True}

        training_generator, validation_generator = self.make_data_generators(params)

        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 use_multiprocessing=True,
                                 workers=4,
                                 epochs=30,
                                 callbacks=[logging, checkpoint])

        self.model.save_weights('trained_stage_2.h5')

        self.draw(image_set='validation', weights_name='trained_stage_2.h5',
                  out_path="output_images_stage_2", save_all=False)

        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 use_multiprocessing=True,
                                 workers=4,
                                 epochs=30,
                                 callbacks=[logging, checkpoint_final_best, early_stopping])

        self.model.save_weights('trained_stage_3.h5')
        self.model_body.load_weights('trained_stage_3_best.h5')
        self.model_body.save('model_data//yolo_spine_model_testing.h5')

    def make_data_generators(self, params):
        training_generator = DataGenerator(self.partition['train'],
                                           anchors=self.anchors,
                                           boxes_path=self.boxes_path,
                                           images_path=self.images_path,
                                           **params)
        validation_generator = DataGenerator(self.partition['validation'],
                                             anchors=self.anchors,
                                             boxes_path=self.boxes_path,
                                             images_path=self.images_path,
                                             **params)
        return training_generator, validation_generator

    def draw(self, test_model_path=None, image_set='validation',
             weights_name='trained_stage_3_best.h5',
             out_path="output_images", save_all=True):
        """
        Draw bounding boxes on image data
        """

        if test_model_path is None:
            self.model_body.load_weights(weights_name)
        else:
            self.model_body = load_model(test_model_path)

        # load validation data
        hdf5_file_images = h5py.File(self.images_path, "r")
        image_data = hdf5_file_images["images"][self.partition[image_set], ...]
        hdf5_file_images.close()

        image_data = np.expand_dims(image_data, axis=1)

        # Create output variables for prediction.
        yolo_outputs = yolo_head(self.model_body.output, self.anchors, len(self.class_names))
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
                    self.model_body.input: image_data[i],
                    input_image_shape: [image_data.shape[2], image_data.shape[3]],
                    K.learning_phase(): 0
                })
            print('Found {} boxes for image.'.format(len(out_boxes)))
            print(out_boxes)

            # Plot image with predicted boxes.
            image_uint8 = (image_data[i][0] * 255).astype(np.uint8)
            image_with_boxes = draw_boxes(image_uint8, out_boxes, out_classes,
                                          self.class_names, out_scores)
            # Save the image:
            if save_all or (len(out_boxes) > 0):
                image = PIL.Image.fromarray(image_with_boxes)
                image.save(os.path.join(out_path, str(i) + '.png'))


if __name__ == '__main__':
    args = argparser.parse_args()
    app = SpineYolo(args)
