import h5py
import numpy as np
import keras
from yad2k.models.keras_yolo import preprocess_true_boxes

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, anchors, boxes_path, images_path, batch_size=32, dim=(32,32), n_channels=3,
                 n_classes=1, shuffle=True):
        'Initialization'
        self.anchors = anchors
        self.images_path = images_path
        self.dim = dim
        self.batch_size = batch_size
        self.boxes_path = boxes_path
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype = np.object)

        list_IDs_temp = [x for x in np.sort(np.array(list_IDs_temp))]

        # Generate data
        hdf5_file_images = h5py.File(self.images_path, "r")
        image_data = hdf5_file_images["images"][list_IDs_temp,...]
        hdf5_file_images.close()
        hdf5_file_boxes = h5py.File(self.boxes_path, "r")
        boxes = hdf5_file_boxes['boxes'][list_IDs_temp,...]
        hdf5_file_boxes.close()

        # image_data = image_data.astype(np.float) / 255

        detectors_mask, matching_true_boxes = self.get_detector_mask(boxes, self.anchors)
        return [image_data, boxes, detectors_mask, matching_true_boxes], np.zeros(len(image_data))

    def get_detector_mask(self, boxes, anchors):
        '''
        Precompute detectors_mask and matching_true_boxes for training.
        Detectors mask is 1 for each spatial position in the final conv layer and
        anchor that should be active for the given boxes and 0 otherwise.
        Matching true boxes gives the regression targets for the ground truth box
        that caused a detector to be active or 0 otherwise.
        '''
        detectors_mask = [0 for i in range(len(boxes))]
        matching_true_boxes = [0 for i in range(len(boxes))]
        for i, box in enumerate(boxes):
            detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, self.anchors, self.dim)
        return np.array(detectors_mask), np.array(matching_true_boxes)