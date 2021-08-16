class Config(object):
    # Training Hyperparameters
    LR = 1e-4
    BATCH_SIZE = 1
    EPOCH = 10000

    # General
    INPUT_SHAPE = [600, 1000, 3]
    NUM_CLASS = 20
    MAX_GT_INSTANCE = 50  # trainval: 42 test: 41
    BACKBONE = None  # currently not support
    FEATURE_SHAPE = [37, 62]

    # Anchors
    ANCHOR_NUM = 9
    ANCHOR_SCALES = [128, 256, 512]
    ANCHOR_RATIOS = [0.5, 1.0, 2.0]
    FEATRUE_STRIDE = 16
    ANCHOR_STRIDE = 1

    # RPN
    RPN_POS_IOU_THOLD = 0.7
    RPN_NEG_IOU_THOLD = 0.3
    RPN_TOTAL_SAMPLE_NUM = 256
    RPN_POS_SAMPLE_RATIO = 0.5
    RPN_LAMBDA = 10.0
    RPN_NMS_NUM = 2000
    RPN_NMS_IOU = 0.7

    # RCNN
    RCNN_ROI_POOL_SIZE = 7
    RCNN_POS_IOU_THOLD = 0.5
    RCNN_NEG_IOU_THOLD = 0.1
    RCNN_TOTAL_SAMPLE_NUM = 64
    RCNN_POS_SAMPLE_RATIO = 0.25
    RCNN_LAMDA = 1.0
