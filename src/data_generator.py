import os
import random
import xml.etree.ElementTree as Et

import cv2
import numpy as np
import tensorflow as tf
import seaborn as sns


class VOCData(tf.keras.utils.Sequence):
    # class VOCData(object):
    IMAGE_DIR = "JPEGImages"
    LABEL_DIR = "Annotations"
    LABEL_LIST = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    def __init__(self, root_dir, inp_img_shape=[600, 1000, 3], batch_size=2, max_gt_instance=50, debug=False):
        # set img and xml directory
        self.root_dir = root_dir
        self.img_path = os.path.join(root_dir, self.IMAGE_DIR)
        self.xml_path = os.path.join(root_dir, self.LABEL_DIR)

        # get file name
        self.data_list = [os.path.splitext(filename)[0] for filename in os.listdir(self.xml_path)]
        self.data_list.sort()

        # DEBUG
        self.debug = debug
        if debug is True:
            self.data_list = self.data_list[0:batch_size]

        # paramters
        self.inp_img_shape = inp_img_shape
        self.batch_size = int(batch_size)
        self.max_gt_instance = int(max_gt_instance)

        # show info
        self._print_info()

    def __getitem__(self, idx):
        data_indices = self.data_list[idx * self.batch_size:(idx + 1) * self.batch_size]

        img_batch = list()
        cls_batch = list()
        bbox_batch = list()
        num_instance_batch = list()

        for i, data_idx in enumerate(data_indices):
            # get img and xml file name
            img_file = os.path.join(self.img_path, data_idx + '.jpg')
            xml_file = os.path.join(self.xml_path, data_idx + '.xml')

            # load data
            img = cv2.imread(img_file)

            # load info
            img_shape, cls_list, bbox_list = self._load_info(xml_file)

            # resize and normalization image
            img_resize = cv2.resize(img, dsize=(self.inp_img_shape[1], self.inp_img_shape[0]), interpolation=cv2.INTER_LINEAR)
            img_norm = img_resize/255.0

            # resize bbox
            bbox_list[:, [0, 2]] *= self.inp_img_shape[0]/img_shape[0]
            bbox_list[:, [1, 3]] *= self.inp_img_shape[1]/img_shape[1]

            # num_instance
            num_instance = len(cls_list)

            # zero pad
            cls_pad = np.pad(
                cls_list,
                (0, self.max_gt_instance-num_instance))
            bbox_pad = np.pad(
                bbox_list,
                ((0, self.max_gt_instance-num_instance), (0, 0)))

            img_batch.append(img_norm)
            cls_batch.append(cls_pad)
            bbox_batch.append(bbox_pad)
            num_instance_batch.append(num_instance)
        img_batch = np.array(img_batch, dtype='float32')
        cls_batch = np.array(cls_batch, dtype='int32')
        bbox_batch = np.array(bbox_batch, dtype='float32')
        num_instance_batch = np.array(num_instance_batch, dtype='int32')
        return img_batch, (cls_batch, bbox_batch, num_instance_batch)

    def __len__(self):
        return int(len(self.data_list)/float(self.batch_size))

    def _load_info(self, xml_file):
        with open(xml_file) as xml:
            # get xml root
            tree = Et.parse(xml)
            root = tree.getroot()

            # get image size
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            channels = int(size.find("depth").text)
            img_shape = [height, width, channels]

            # get object info
            cls_list = list()
            bbox_list = list()
            objects = root.findall("object")
            for _object in objects:
                # get class
                name = _object.find("name").text
                category = self.LABEL_LIST.index(name) + 1

                # get bndbox
                bndbox = _object.find("bndbox")
                xmin = int(bndbox.find("xmin").text)-1
                xmax = int(bndbox.find("xmax").text)-1
                ymin = int(bndbox.find("ymin").text)-1
                ymax = int(bndbox.find("ymax").text)-1

                # label = [category, xmin, xmax, ymin, ymax]
                cls_list.append(category)
                bbox_list.append([ymin, xmin, ymax, xmax])

        img_shape = np.array(img_shape, dtype='float32')
        cls_list = np.array(cls_list, dtype='int32')
        bbox_list = np.array(bbox_list, dtype='float32')
        return img_shape, cls_list, bbox_list

    def on_epoch_end(self):
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.data_list)

    def draw_label_img(self, img, cls_list, bbox_list, score_list=None):
        # generate color set
        color_palette = np.array(sns.color_palette("Dark2", 20))
        color_palette = (color_palette*255).astype("uint8")

        # make bboxlist `integer`W
        bbox_list = bbox_list.astype('int32')

        # un-normalize the image
        img_vis = img.copy()
        if img_vis.dtype != np.uint8:
            img_vis *= 255
            img_vis = img_vis.astype('uint8')

        # draw the bounding box and put label text on the corresponding bbox
        for i in range(len(cls_list)):
            cls_name = self.LABEL_LIST[cls_list[i] - 1]
            ymin = bbox_list[i][0]
            xmin = bbox_list[i][1]
            ymax = bbox_list[i][2]
            xmax = bbox_list[i][3]
            cv2.rectangle(
                img_vis,
                pt1=(int(xmin), int(ymin)),
                pt2=(int(xmax), int(ymax)),
                color=[int(c) for c in color_palette[cls_list[i] - 1]],
                thickness=2,
            )
            if score_list is None:
                label_text = cls_name
            else:
                label_text = cls_name + ": {:.3f}".format(score_list[i])

            cv2.putText(
                img_vis,
                label_text,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                org=(int(xmin) + 5, int(ymin) + 20),
                color=[int(c) for c in color_palette[cls_list[i] - 1]],
                thickness=2,
            )
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        return img_vis

    def _print_info(self):
        print("Load from: {}".format(self.root_dir))
        print("DEBUD MODE: {}".format(self.debug))
        print("Batch size: {}".format(self.batch_size))
        print("Images: {}".format(len(self.data_list)))
        print("Classes: {}\n".format(self.LABEL_LIST))


if __name__ == '__main__':
    BATCH_SIZE = 2

    # create voc explore dataset
    dataset_dir = 'dataset/VOC2007/trainval/VOCdevkit/VOC2007'
    vocdata = VOCData(dataset_dir, batch_size=BATCH_SIZE)

    # get max gt instance
    total_num_instance = list()
    i = 0
    for _, (_, _, num_instacne_batch) in vocdata:
        total_num_instance.append(np.max(num_instacne_batch))
        print(i, np.max(num_instacne_batch))
        i += 1
    total_num_instance = np.array(total_num_instance)
    print(np.max(total_num_instance))
