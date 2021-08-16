import os
import yaml
import datetime

import tensorflow as tf

from config.config import Config
from src.data_generator import VOCData
from src.model import build_faster_rcnn_graph, FasterRCNN


if __name__ == '__main__':
    # set model dirs
    model_dir = 'model/20210816-233122'
    train_data_dir = 'dataset/VOC2007/trainval/VOCdevkit/VOC2007'
    test_data_dir = 'dataset/VOC2007/test/VOCdevkit/VOC2007'


    # load config
    config = Config()

    test_dataset = VOCData(train_data_dir,
    # test_dataset = VOCData(test_data_dir,
                           config.INPUT_SHAPE,
                           batch_size=config.BATCH_SIZE,
                           max_gt_instance=config.MAX_GT_INSTANCE,
                           debug=True)

    # build model
    # inputs, outputs = build_faster_rcnn_graph(config)
    # model = FasterRCNN(inputs, outputs, config)
    # model.load_weights(model_dir)
    # model.compile()
    model = tf.keras.models.load_model(model_dir)
    model.summary()
    
    # test
    for img_batch, (cls_batch, bbox_batch, num_instacne_batch) in test_dataset:
        # visualize the dataset
        outputs = model(img_batch, training=False)
        rpn_cls_output = outputs[0]  # [B, A, 2]
        rpn_reg_output = outputs[1]  # [B, A, 4]
        rcnn_cls_output = outputs[2]  # [B*N, 21]
        rcnn_reg_output = outputs[3]  # [B*N, 20*4]
        roi_boxes = outputs[4]  # [B*N, 4]
        valid_num = outputs[5]  # [B]
        print(rpn_reg_output)
        exit()
        # for idx in config.BATCH_SIZE: