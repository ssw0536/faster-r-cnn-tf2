import os
import yaml
import datetime
import multiprocessing

import tensorflow as tf

from config.config import Config
from src.data_generator import VOCData
from src.model import build_faster_rcnn_graph, FasterRCNN


if __name__ == '__main__':
    # set model dirs
    cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join("model", cur_date)
    log_dir = os.path.join(model_dir, "logs")
    train_data_dir = 'dataset/VOC2007/trainval/VOCdevkit/VOC2007'
    test_data_dir = 'dataset/VOC2007/test/VOCdevkit/VOC2007'

    # create model directory
    try:
        os.mkdir(model_dir)
        print("Directory: ", model_dir, ". Created")
    except FileExistsError:
        print("Directory: ", model_dir, ". Already exists")

    # save configs
    config_vars = vars(Config)
    config_dict = {}
    for attr in list(config_vars.keys()):
        if not attr.startswith("__"):
            config_dict[attr] = config_vars[attr]
    with open(os.path.join(model_dir, 'config.yml'), 'w') as yaml_file:
        yaml.dump(config_dict, yaml_file, default_flow_style=False)

    # load config
    config = Config()

    # load dataset
    train_dataset = VOCData(train_data_dir,
                            config.INPUT_SHAPE,
                            batch_size=config.BATCH_SIZE,
                            max_gt_instance=config.MAX_GT_INSTANCE,
                            debug=True)

    test_dataset = VOCData(train_data_dir,
    # test_dataset = VOCData(test_data_dir,
                           config.INPUT_SHAPE,
                           batch_size=config.BATCH_SIZE,
                           max_gt_instance=config.MAX_GT_INSTANCE,
                           debug=True)

    # build model
    inputs, outputs = build_faster_rcnn_graph(config)
    model = FasterRCNN(inputs, outputs, config)

    # compile model
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR))
    model.compile(optimizer=tf.keras.optimizers.SGD(
        learning_rate=config.LR,
        momentum=0.9))

    # callbacks
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_dir,
        save_weights_only=False,
        monitor="val_rcnn_cls_acc",
        mode="max",
        save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=0,
                                                          write_graph=True,
                                                          write_images=False)

    # train model
    model.fit(
        train_dataset,
        epochs=config.EPOCH,
        callbacks=[tensorboard_callback],
        # callbacks=[save_callback, tensorboard_callback],
        validation_data=test_dataset,
        max_queue_size=100,
        )
