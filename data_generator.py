import h5py
import numpy as np
import keras
from yad2k.models.keras_yolo import preprocess_true_boxes


def random_rotation_with_boxes(x, boxes, rg, row_axis=1, col_axis=2, channel_axis=0,
                               fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor. Also rotates the corresponding bounding boxes

   # Arguments
       x: Input tensor. Must be 3D.
       boxes: a numpy array of bounding boxes [x_center, y_center, box_width, box_height], values in [0,1].
       rg: Rotation range, in degrees.
       row_axis: Index of axis for rows in the input tensor.
       col_axis: Index of axis for columns in the input tensor.
       channel_axis: Index of axis for channels in the input tensor.
       fill_mode: Points outside the boundaries of the input
           are filled according to the given mode
           (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
       cval: Value used for points outside the boundaries
           of the input if `mode='constant'`.

   # Returns
       Rotated Numpy image tensor.
       And rotated bounding boxes
   """

    # sample parameter for augmentation
    theta = np.pi / 180 * np.random.uniform(-rg, rg)

    # apply to image
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)

    # apply to vertices
    # vertices = boxes_to_vertices(boxes)
    # vertices = vertices.reshape((-1, 2))

    # apply offset to have pivot point at [0.5, 0.5]
    # vertices -= [0.5, 0.5]

    # apply rotation, we only need the rotation part of the matrix
    # vertices = np.dot(vertices, rotation_matrix[:2, :2])
    # vertices += [0.5, 0.5]

    # boxes = vertices_to_boxes(vertices)

    return x, boxes#, vertices

def rotate_coordinates(origin,points,angle):
    # points is an x,y matrix of shape (n,2)
    ox,oy = origin
    ox,oy = points[:,0], points[:,1]


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.

   # Arguments
       x: 2D numpy array, single image.
       transform_matrix: Numpy array specifying the geometric transformation.
       channel_axis: Index of axis for channels in the input tensor.
       fill_mode: Points outside the boundaries of the input
           are filled according to the given mode
           (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
       cval: Value used for points outside the boundaries
           of the input if `mode='constant'`.

   # Returns
       The transformed version of the input.
   """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

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

