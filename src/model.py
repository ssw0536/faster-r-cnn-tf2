import numpy as np
import tensorflow as tf
# from config import Config as config
# from data_generator import VOCData

# rpn -> rigion proposal
# 1) B A 4, B A 2
# 2) BxAx4, Ax4 -> BxAx4 : unparm
# 3) BxAx4, clip_shape -> BxAx4 : clip
# 4) BxAx4 -> Nx4, N, B: nms

# rcnn
# 1) Nx4 -> Nx4 : norm
# 2) BxHxWxC, Nx4, N : roi pool
# 3) Nx20x4, Nx21 : output
# 4) Nx20x4, Nx21 -> Nx4, N : collect output by score
# 5) Nx4, Nx4 -> Nx4 : unparam
# 6) Nx4 -> Nx4 : clip shape
# 5) Nx4, N, N -> BxM?x

# rpn_loss


# Validated!
@tf.function(input_signature=(
    tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 4), dtype=tf.float32)))
def box_iou(box_i, box_t):
    # split to vectors
    # len_i = box_i.shape[0]
    # len_t = box_t.shape[0]  
    # print(len_i)
    # print(len_y)
    # exit()
    y1_i, x1_i, y2_i, x2_i = tf.split(box_i, 4, axis=1)
    y1_t, x1_t, y2_t, x2_t = tf.split(box_t, 4, axis=1)

    # get min, max matrix for x, y
    min_x2 = tf.minimum(x2_i, tf.transpose(x2_t))
    min_y2 = tf.minimum(y2_i, tf.transpose(y2_t))
    max_x1 = tf.maximum(x1_i, tf.transpose(x1_t))
    max_y1 = tf.maximum(y1_i, tf.transpose(y1_t))

    # get intersection area
    intersection_area = tf.maximum((min_x2 - max_x1), 0) * tf.maximum((min_y2 - max_y1), 0)
    box_i_area = (x2_i - x1_i) * (y2_i - y1_i)
    box_t_area = (x2_t - x1_t) * (y2_t - y1_t)
    overlaps = intersection_area / (box_i_area + tf.transpose(box_t_area) - intersection_area)
    return overlaps


@tf.function
def yxyx_to_xyctrwh(yyxx):
    y_min, x_min, y_max, x_max = tf.unstack(yyxx, axis=-1)
    x_ctr = (x_min + x_max)/tf.constant(2, dtype=tf.float32)
    y_ctr = (y_min + y_max)/tf.constant(2, dtype=tf.float32)
    w = x_max - x_min
    h = y_max - y_min
    xyctrwh = tf.stack([x_ctr, y_ctr, w, h], axis=-1)
    return xyctrwh


@tf.function
def xyctrwh_to_yxyx(xyctrwh):
    x_ctr, y_ctr, w, h = tf.unstack(xyctrwh, axis=-1)
    y_min = y_ctr - h/tf.constant(2, dtype=tf.float32)
    x_min = x_ctr - w/tf.constant(2, dtype=tf.float32)
    y_max = y_ctr + h/tf.constant(2, dtype=tf.float32)
    x_max = x_ctr + w/tf.constant(2, dtype=tf.float32)
    yyxx = tf.stack([y_min, x_min, y_max, x_max], axis=-1)
    return yyxx


@tf.function
def parameterize_box(target_box, base_box):
    """
    target_box : N x 4, with [x_ctr, y_ctr, w, h]
        The most overlappped bbox with each anchor
    base_box : N x 4, with [x_ctr, y_ctr, w, h]

    Returns
    delta : N x 4 ((x_t - x_b)/w_b, (y_t - y_b)/h_b, log(w_t/w_b), log(h_t/h_b))
    """
    x_t, y_t, w_t, h_t = tf.unstack(target_box, 4, axis=-1)
    x_b, y_b, w_b, h_b = tf.unstack(base_box, 4, axis=-1)

    x_d = (x_t - x_b)/w_b
    y_d = (y_t - y_b)/h_b
    w_d = tf.math.log(w_t/w_b)
    h_d = tf.math.log(h_t/h_b)

    delta = tf.stack([x_d, y_d, w_d, h_d], axis=-1)
    return delta


@tf.function
def unparameterize_box(delta, base_box):
    """Unparameterize the RPN regression output.
    x* = tx*wa + xa, y* = ty*ha + ya, w*=exp(tw*wa), h* = exp(th*ha)

    Args:
        rpn_output_reg (`ndarray`): N x 4, RPN regression output
        anchor_list (`ndarray`): N x 4, Anchores
    """
    x_d, y_d, w_d, h_d = tf.unstack(delta, 4, axis=-1)
    x_b, y_b, w_b, h_b = tf.unstack(base_box, 4, axis=-1)

    x_o = x_d*w_b + x_b
    y_o = y_d*h_b + y_b
    w_o = tf.math.exp(w_d)*w_b
    h_o = tf.math.exp(h_d)*h_b

    output_box = tf.stack([x_o, y_o, w_o, h_o], axis=-1)
    return output_box


@tf.function
def non_maximum_surpression(boxes, scores, max_output_size, iou_threshold):
    """Prunes away boxes that have high IOU overlap with
    previously selected boxes. Bounding boxes should be
    [y_min, x_min, y_max, x_max] with absolute scale or
    normalized scale.

    Args:
        boxes (`tf.Tensor`): A 3-D float Tensor of shape [batch_size, num_boxes, 4].
        scores (`tf.Tensor`): A 2-D float Tensor of shape [batch_size, num_boxes].
        max_output_size (`tf.Tensor`): A scalar integer Tensor representing the
        maximum number of boxes to be selected by non-max suppression.
        iou_threshold (`tf.Tensor`): A scalar float Tensor.

    Returns:
        `Tuple`: (boxes, indicies)
            boxes (`tf.Tensor`): A 2-D float Tensor of shape [N, 4].
            indicies (`tf.Tensor`): A 1-D integer Tensor of shape [N]
            with int32 values in [0, batch).

            `N <= batch_size*max_output_size` denotes the
            number of boxes in a batch. 
    """
    # non maximum surpress
    selected_indices_padded, num_valid = tf.image.non_max_suppression_padded(
        boxes=boxes,
        scores=scores,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        pad_to_max_output_size=True,
        canonicalized_coordinates=True,
        )

    # make list for concat
    selected_boxes = list()
    selected_boxes_indicies = list()

    # for i bacth append to list
    bacth_size = tf.shape(num_valid)[0]
    for i in tf.range(bacth_size):
        selected_indices_i = selected_indices_padded[i, 0:num_valid[i]]
        selected_boxes_i = tf.gather(boxes[i], selected_indices_i)
        selected_boxes_indicies_i = tf.zeros_like(selected_indices_i, dtype=tf.int32) + i

        selected_boxes.append(selected_boxes_i)
        selected_boxes_indicies.append(selected_boxes_indicies_i)

    selected_boxes = tf.concat(selected_boxes, axis=0)
    selected_boxes_indicies = tf.concat(selected_boxes_indicies, axis=0)

    return selected_boxes, selected_boxes_indicies, num_valid


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.

    Return: [y1, x1, y2, x2] x N anchor boxes. Ctr order is [[x0,y0], [x1, y0],
        ..., [xN, y0], [x0, y1], [x1, y1],... [xN, yM]]

    from: https://github.com/matterport/Mask_RCNN
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride + (feature_stride // 2)
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride + (feature_stride // 2)
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes.astype('float32')


def build_faster_rcnn_graph(config):
    input_image = tf.keras.Input(shape=config.INPUT_SHAPE)

    # get the pre-trained vgg16 model
    vgg16 = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_tensor=input_image,
        pooling=None,
        )

    # get layer13 feature from the vgg16 model
    feature_map = vgg16.get_layer('block5_conv3').output

    # create region proposal network
    # 1) intermediate layer
    intermediate_layer = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        )(feature_map)
    intermediate_layer = tf.keras.layers.ReLU()(intermediate_layer)
    feature_len = intermediate_layer.shape[1]*intermediate_layer.shape[2]

    # 2) rpn output layer
    rpn_reg = tf.keras.layers.Conv2D(
        filters=4*config.ANCHOR_NUM,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        )(intermediate_layer)
    rpn_reg = tf.reshape(rpn_reg, (-1, feature_len*config.ANCHOR_NUM, 4), name='rpn_reg')

    rpn_cls = tf.keras.layers.Conv2D(
        filters=2*config.ANCHOR_NUM,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        )(intermediate_layer)
    rpn_cls = tf.reshape(rpn_cls, (-1, feature_len*config.ANCHOR_NUM, 2))
    rpn_cls = tf.keras.layers.Softmax(name='rpn_cls')(rpn_cls)

    # create rcnn model
    # 1) generate anchors
    anchors = generate_anchors(
        config.ANCHOR_SCALES,
        config.ANCHOR_RATIOS,
        config.FEATURE_SHAPE,
        config.FEATRUE_STRIDE,
        config.ANCHOR_STRIDE
        )
    anchors = tf.constant(anchors, dtype=tf.float32)

    # 2) RoI Pooling
    roi_box, roi_box_indicies, valid_num = RegionProposalLayer(
        iou_threshold=config.RPN_NMS_IOU,
        max_output_size=config.RPN_NMS_NUM,
        box_boundary=[0, 0, config.INPUT_SHAPE[0], config.INPUT_SHAPE[1]],
        anchors=anchors,
        name="region_p"
        )((rpn_cls, rpn_reg))

    norm_roi_box = tf.divide(
        roi_box,
        tf.constant([config.INPUT_SHAPE[0]-1, config.INPUT_SHAPE[1]-1,
                     config.INPUT_SHAPE[0]-1, config.INPUT_SHAPE[1]-1],
                    dtype=tf.float32))

    roi_pool = tf.image.crop_and_resize(
        image=feature_map,
        boxes=norm_roi_box,
        box_indices=roi_box_indicies,
        crop_size=(config.RCNN_ROI_POOL_SIZE, config.RCNN_ROI_POOL_SIZE),
        name='roi_pooling',
        )  # (B*rois) x crop_size x crop_size x depth
    roi_pool_flatten = tf.keras.layers.Flatten()(roi_pool)

    # 3) fully connected layer
    fc1 = tf.keras.layers.Dense(
        units=4096,
        activation='relu',
        name='roi_pool_fc1'
        )(roi_pool_flatten)  # (B*rois) x 4096
    fc2 = tf.keras.layers.Dense(
        units=4096,
        activation='relu',
        name='roi_pool_fc2'
        )(fc1)  # (B*rois) x 4096

    # 4) get cls and reg out
    rcnn_cls = tf.keras.layers.Dense(
        units=config.NUM_CLASS+1,
        activation='softmax',
        name='rcnn_cls',
        )(fc2)  # (B*rois) x (num_cls+1)
    rcnn_reg_flatten = tf.keras.layers.Dense(
        units=(4*config.NUM_CLASS),
        activation='linear',
        )(fc2)  # (B*rois) x (num_cls*4)
    rcnn_reg = tf.keras.layers.Reshape(
        target_shape=(config.NUM_CLASS, 4),
        name='rcnn_reg',
        )(rcnn_reg_flatten)

    # create inputs and outputs
    inputs = input_image
    outputs = (rpn_cls, rpn_reg, rcnn_cls, rcnn_reg, roi_box, valid_num)
    return inputs, outputs


@tf.function
def tensor_from_sparse(indices, values, length):
    # indices: (N,) int64
    # valeus : (N) ?
    # length    : (M, n) int64
    indices = tf.cast(indices, dtype=tf.int64)
    length = tf.cast(length, dtype=tf.int64)
    sparse_tensor = tf.sparse.SparseTensor(
        tf.expand_dims(tf.sort(indices), axis=1),  # (N, 1)
        values,  # (N, )
        [length],  # (1, )
        )
    return tf.sparse.to_dense(sparse_tensor)


@tf.function
def get_rpn_target(
        anchor,
        gt_box,
        image_shape=[600, 1000, 3],
        pos_threshold=0.7,
        neg_threshold=0.3,
        total_sample_num=256,
        pos_sample_ratio=0.5,
        ):
    # cls_target = tf.fill(anchor_len, tf.constant(-1, dtype=tf.int32))
    # reg_target = tf.zeros_like(anchor, dtype=tf.int32)
    # cls_weight = tf.zeros(anchor_len, dtype='float32')
    # reg_weight = tf.zeros(anchor_len, dtype='float32')

    # get length of anchor with int64 --> `sparse to dense`
    anchor_len = tf.shape(anchor, out_type=tf.int32)[0]

    # get inner anchor length(ignore cross-edge anchors)
    inner_anchor_bool_mask = tf.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] < image_shape[0]) &
        (anchor[:, 3] < image_shape[1]),
        True,
        False,
        )
    inner_anchor_float_mask = tf.where(
        inner_anchor_bool_mask,
        1.0,
        0.0,
        )

    # get IoU matrix and ignore cross-boundary anchors
    overlaps = box_iou(
        gt_box,
        anchor,
        )
    overlaps = tf.multiply(overlaps, inner_anchor_float_mask)

    # get max, argamx
    argmax_overlaps = tf.argmax(overlaps, axis=0)  # (anchor length, )
    max_overlaps = tf.reduce_max(overlaps, axis=0)  # (anchor length, )
    argmax_overlaps_bbox = tf.argmax(overlaps, axis=1)  # (bbox length, )
    max_overlaps_bbox = tf.reduce_max(overlaps, axis=1)  # (bbox length, )

    # repeated indices makes `sparse_to_dense` fail
    argmax_overlaps_bbox, _ = tf.unique(argmax_overlaps_bbox)

    # get pos/neg mask along the anchors
    # 1) get pos/neg based on IoU threshold
    neg_mask_iou = (max_overlaps < neg_threshold) & inner_anchor_bool_mask
    pos_mask_iou = (max_overlaps > pos_threshold) & inner_anchor_bool_mask
    # 2) pos anchors with the highest IoU with the ground-truth box
    pos_mask_highest = tensor_from_sparse(
        argmax_overlaps_bbox,
        tf.fill(tf.shape(argmax_overlaps_bbox), True),
        anchor_len)
    pos_mask = pos_mask_iou | pos_mask_highest
    # 3) remove pos anchors from the neg anchors
    neg_mask = neg_mask_iou & ~pos_mask

    # get pos/neg indices
    anchor_indices = tf.range(anchor_len, dtype=tf.int32)
    pos_indices = anchor_indices[pos_mask]
    neg_indices = anchor_indices[neg_mask]

    # random sample pos/neg indices
    desired_pos_num = tf.cast(
        tf.math.ceil(total_sample_num*pos_sample_ratio),
        dtype=tf.int32)
    pos_indices = tf.random.shuffle(pos_indices)[:desired_pos_num]
    pos_num = tf.shape(pos_indices, out_type=tf.int32)[0]
    neg_num = total_sample_num - pos_num
    neg_indices = tf.random.shuffle(neg_indices)[:neg_num]
    train_indices = tf.concat([pos_indices, neg_indices], axis=0)

    # cls target and weight
    cls_target = tf.pad(
        tf.fill([pos_num], 1),
        [[0, neg_num]]
        )

    # reg target and weight
    pos_target_gt_boxes = tf.gather(gt_box, tf.gather(argmax_overlaps, pos_indices))
    pos_target_anchors = tf.gather(anchor, pos_indices)
    pos_target_reg_values = parameterize_box(
        pos_target_gt_boxes,
        pos_target_anchors)
    reg_target = tf.pad(
        pos_target_reg_values,
        tf.convert_to_tensor([[0, neg_num], [0, 0]])
        )
    return cls_target, reg_target, train_indices, pos_num


@tf.function
def get_rcnn_target(
        rois,
        gt_box,
        gt_cls_id,
        num_valid,
        pos_threshold=0.5,
        neg_threshold=0.1,
        total_sample_num=64,
        pos_sample_ratio=0.25,
        ):
    # num_valid: int32

    # get rois length
    rois_len = tf.shape(rois, out_type=tf.int32)[0]

    # get IoU between RoIs and gt boxes
    overlaps = box_iou(
        gt_box,
        rois,
        )

    # get max, argamx
    argmax_overlaps = tf.argmax(overlaps, axis=0) # (rois length, 1)
    max_overlaps = tf.reduce_max(overlaps, axis=0) # (rois length, 1)

    # get valid mask
    valid_mask = tf.pad(
        tf.fill([num_valid], True),
        [[0, rois_len-num_valid]],
        constant_values=False
        )

    # get pos/neg mask
    pos_mask = (max_overlaps >= pos_threshold) & valid_mask
    neg_mask = (max_overlaps >= neg_threshold) & (max_overlaps <
                                                  pos_threshold) & valid_mask

    # get pos/neg indices
    roi_indices = tf.range(rois_len, dtype=tf.int32)
    pos_indices = roi_indices[pos_mask]
    neg_indices = roi_indices[neg_mask]

    # random sample pos/neg indices
    pos_num = tf.cast(
        tf.math.ceil(total_sample_num*pos_sample_ratio),
        dtype=tf.int32)
    pos_indices = tf.random.shuffle(pos_indices)[:pos_num]
    sampled_pos_num = tf.shape(pos_indices)[0]
    neg_num = total_sample_num - sampled_pos_num
    neg_indices = tf.random.shuffle(neg_indices)[:neg_num]
    train_indices = tf.concat([pos_indices, neg_indices], axis=0)

    # get training cls targets
    pos_target_gt_box_indices = tf.gather(argmax_overlaps, pos_indices)
    pos_target_cls_id_values = tf.gather(gt_cls_id, pos_target_gt_box_indices)

    cls_target = tf.pad(
        pos_target_cls_id_values,
        tf.convert_to_tensor([[0, neg_num]]),
        "CONSTANT")

    # get training reg targets
    pos_target_rois = tf.gather(rois, pos_indices)
    pos_target_gt_box = tf.gather(gt_box, pos_target_gt_box_indices)
    pos_target_reg_values = parameterize_box(pos_target_gt_box, pos_target_rois)
    reg_target = tf.pad(
        pos_target_reg_values,
        tf.convert_to_tensor([[0, neg_num], [0, 0]])
        )

    # the number of training indices can be less then `total_sample_num`
    # in early learning. `-1` Pad to keep size.
    valid_train_num = tf.shape(train_indices)[0]
    train_indices = tf.pad(
      train_indices,
      tf.convert_to_tensor([[0, total_sample_num-valid_train_num]]),
      constant_values=-1)

    # (T,), (T, 4), (T, ), (1, )
    return cls_target, reg_target, train_indices, valid_train_num, sampled_pos_num


@tf.function
def smooth_l1_loss(y_true, y_pred):
    error = tf.subtract(y_true, y_pred)
    abs_error = tf.abs(error)

    loss = tf.where(
        abs_error <= 1,
        0.5*tf.pow(error, 2),
        abs_error - 0.5,
        )
    loss = tf.reduce_sum(loss, axis=-1)
    return loss


@tf.function
def rpn_cls_loss_fn(target, output, train_indices):
    output = tf.gather(output, train_indices, batch_dims=1)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM
        )(target, output)
    loss = loss/tf.size(train_indices, out_type=tf.float32)
    return loss


@tf.function
def rpn_reg_loss_fn(target, output, train_indices, pos_num):
    if pos_num == 0:
        loss = tf.constant(1e-6, dtype=tf.float32)
    else:
        output = tf.gather(output, train_indices[:pos_num])
        target = target[:pos_num]
        loss = smooth_l1_loss(target, output)
        loss = tf.reduce_sum(loss)/tf.cast(pos_num, dtype=tf.float32)
    return loss


@tf.function
def rcnn_cls_loss_fn(target, output, train_indices, valid_num):
    if valid_num == 0:
        loss = tf.constant(1e-6, dtype=tf.float32)
    else:
        output = tf.gather(output, train_indices[:valid_num])
        target = target[:valid_num]
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM
            )(target, output)
        loss = loss/tf.cast(valid_num, dtype=tf.float32)
    return loss


@tf.function
def rcnn_reg_loss_fn(reg_target, reg_output, cls_target, train_indices, pos_num):
    if pos_num == 0:
        loss = tf.constant(1e-6, dtype=tf.float32)
    else:
        reg_target = tf.gather(reg_target, train_indices[:pos_num])
        reg_output = tf.gather(reg_output, train_indices[:pos_num])
        reg_output = tf.gather(reg_output, cls_target[:pos_num]-1, batch_dims=1)
        loss = smooth_l1_loss(reg_target, reg_output)
        loss = tf.reduce_sum(loss)/tf.cast(pos_num, dtype=tf.float32)
    return loss


class RegionProposalLayer(tf.keras.layers.Layer):
    def __init__(self, iou_threshold, max_output_size, box_boundary, anchors, **kwargs):
        super(RegionProposalLayer, self).__init__(**kwargs)
        self.iou_threshold = iou_threshold
        self.max_output_size = max_output_size
        self.box_boundary = box_boundary  # y_min, x_min, y_max, x_max
        self.anchors = anchors

    def call(self, inputs):
        # BxAx4, BxAx2, Ax4
        rpn_probs, rpn_deltas = inputs

        # unparameterize bounding boxes
        rpn_boxes = tf.map_fn(
            fn=lambda x: unparameterize_box(x, self.anchors),
            elems=rpn_deltas)

        # clip with image boundary
        y_min, x_min, y_max, x_max = self.box_boundary
        rpn_boxes = tf.clip_by_value(
            rpn_boxes,
            [[[y_min, x_min, y_min, x_min]]],
            [[[y_max, x_max, y_max, x_max]]])

        # scores
        rpn_scores = rpn_probs[:, :, 1]

        # non maximum surpression
        selected_indices, num_valid = tf.image.non_max_suppression_padded(
            boxes=rpn_boxes,
            scores=rpn_scores,
            max_output_size=self.max_output_size,
            iou_threshold=self.iou_threshold,
            pad_to_max_output_size=True,
            canonicalized_coordinates=True)

        selected_boxes = tf.gather(rpn_boxes, selected_indices, batch_dims=1)
        selected_boxes = tf.reshape(selected_boxes, (-1, 4))
        selected_boxes_indicies = tf.repeat(
            tf.range(tf.shape(num_valid)[0]),
            tf.fill(tf.shape(num_valid), self.max_output_size))

        return selected_boxes, selected_boxes_indicies, num_valid


class FasterRCNN(tf.keras.models.Model):
    def __init__(self, inputs, outputs, config):
        super(FasterRCNN, self).__init__(inputs=inputs, outputs=outputs)
        self.config = config

        # generate anchors
        anchors = generate_anchors(
            self.config.ANCHOR_SCALES,
            self.config.ANCHOR_RATIOS,
            self.config.FEATURE_SHAPE,
            self.config.FEATRUE_STRIDE,
            self.config.ANCHOR_STRIDE
            )
        self.anchors = tf.constant(anchors, dtype=tf.float32)
        self.anchor_len = tf.constant(len(anchors), dtype=tf.int32)

    def compile(self, optimizer='rmsprop', run_eagerly=None):
        super(FasterRCNN, self).compile(optimizer=optimizer, run_eagerly=run_eagerly)

        # rpn loss and acc tracker
        self.rpn_cls_acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='rpn_cls_acc')
        self.rpn_cls_loss_tracker = tf.keras.metrics.Mean(name='rpn_cls_loss')
        self.rpn_reg_loss_tracker = tf.keras.metrics.Mean(name='rpn_reg_loss')

        # rpn loss and acc tracker
        self.rcnn_cls_acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='rcnn_cls_acc')
        self.rcnn_cls_loss_tracker = tf.keras.metrics.Mean(name='rcnn_cls_loss')
        self.rcnn_reg_loss_tracker = tf.keras.metrics.Mean(name='rcnn_reg_loss')

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        metrics = [
            self.rpn_cls_acc_tracker,
            self.rpn_cls_loss_tracker,
            self.rpn_reg_loss_tracker,
            self.rcnn_cls_acc_tracker,
            self.rcnn_cls_loss_tracker,
            self.rcnn_reg_loss_tracker,
            ]
        return metrics

    def train_step(self, data):
        # batch data
        images, (gt_cls_ids, gt_boxes, num_gt_instance) = data

        # get rpn targets
        rpn_targets = tf.map_fn(
            fn=lambda idx: get_rpn_target(
                self.anchors,
                gt_boxes[idx][:num_gt_instance[idx]],
                # gt_box,
                image_shape=self.config.INPUT_SHAPE,
                pos_threshold=self.config.RPN_POS_IOU_THOLD,
                neg_threshold=self.config.RPN_NEG_IOU_THOLD,
                total_sample_num=self.config.RPN_TOTAL_SAMPLE_NUM,
                pos_sample_ratio=self.config.RPN_POS_SAMPLE_RATIO),
            elems=tf.range(self.config.BATCH_SIZE),
            # elems=num_gt_instance,
            fn_output_signature=(
                tf.TensorSpec(shape=[self.config.RPN_TOTAL_SAMPLE_NUM], dtype=tf.int32),
                tf.TensorSpec(shape=[self.config.RPN_TOTAL_SAMPLE_NUM, 4], dtype=tf.float32),
                tf.TensorSpec(shape=[self.config.RPN_TOTAL_SAMPLE_NUM], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.int32),),
            parallel_iterations=self.config.BATCH_SIZE,
            )

        # unpack rpn_targets
        rpn_cls_target = rpn_targets[0]
        rpn_reg_target = rpn_targets[1]
        rpn_train_indices = rpn_targets[2]
        rpn_pos_num = rpn_targets[3]

        # get gradient
        with tf.GradientTape() as tape:
            # get network output
            outputs = self(images, training=True)

            # unpack outputs
            rpn_cls_output = outputs[0]  # [B, A, 2]
            rpn_reg_output = outputs[1]  # [B, A, 4]
            rcnn_cls_output = outputs[2]  # [B*N, 21]
            rcnn_reg_output = outputs[3]  # [B*N, 20*4]
            roi_boxes = outputs[4]  # [B*N, 4]
            valid_num = outputs[5]  # [B]

            # reshape rcnn outputs
            rcnn_cls_output = tf.reshape(
                rcnn_cls_output,
                [self.config.BATCH_SIZE, self.config.RPN_NMS_NUM, self.config.NUM_CLASS+1])
            rcnn_reg_output = tf.reshape(
                rcnn_reg_output,
                [self.config.BATCH_SIZE, self.config.RPN_NMS_NUM, self.config.NUM_CLASS, 4])
            roi_boxes = tf.reshape(
                roi_boxes,
                [self.config.BATCH_SIZE, self.config.RPN_NMS_NUM, 4])

            # get rcnn targets
            rcnn_targets = tf.map_fn(
                fn=lambda idx: get_rcnn_target(
                    roi_boxes[idx][:valid_num[idx]],
                    gt_boxes[idx][:num_gt_instance[idx]],
                    gt_cls_ids[idx][:num_gt_instance[idx]],
                    valid_num[idx],
                    pos_threshold=self.config.RCNN_POS_IOU_THOLD,
                    neg_threshold=self.config.RCNN_NEG_IOU_THOLD,
                    total_sample_num=self.config.RCNN_TOTAL_SAMPLE_NUM,
                    pos_sample_ratio=self.config.RCNN_POS_SAMPLE_RATIO),
                elems=tf.range(self.config.BATCH_SIZE),
                fn_output_signature=(
                    tf.TensorSpec(shape=[self.config.RCNN_TOTAL_SAMPLE_NUM], dtype=tf.int32),
                    tf.TensorSpec(shape=[self.config.RCNN_TOTAL_SAMPLE_NUM, 4], dtype=tf.float32),
                    tf.TensorSpec(shape=[self.config.RCNN_TOTAL_SAMPLE_NUM], dtype=tf.int32),
                    tf.TensorSpec(shape=[], dtype=tf.int32),
                    tf.TensorSpec(shape=[], dtype=tf.int32),),
                parallel_iterations=self.config.BATCH_SIZE,
                )

            # unpack rcnn_targets
            rcnn_cls_target = rcnn_targets[0]
            rcnn_reg_target = rcnn_targets[1]
            rcnn_train_indices = rcnn_targets[2]
            rcnn_valid_num = rcnn_targets[3]
            rcnn_pos_num = rcnn_targets[4]

            ########
            # LOSS #
            ########
            rpn_cls_loss = rpn_cls_loss_fn(
                rpn_cls_target,
                rpn_cls_output,
                rpn_train_indices)

            rpn_reg_loss = tf.map_fn(
                fn=lambda idx: rpn_reg_loss_fn(
                    rpn_reg_target[idx],
                    rpn_reg_output[idx],
                    rpn_train_indices[idx],
                    rpn_pos_num[idx]),
                elems=tf.range(self.config.BATCH_SIZE),
                fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.float32),
                )
            rpn_reg_loss = tf.reduce_sum(rpn_reg_loss)/tf.cast(self.config.BATCH_SIZE, dtype=tf.float32)

            rcnn_cls_loss = tf.map_fn(
                fn=lambda idx: rcnn_cls_loss_fn(
                    rcnn_cls_target[idx],
                    rcnn_cls_output[idx],
                    rcnn_train_indices[idx],
                    rcnn_valid_num[idx]),
                elems=tf.range(self.config.BATCH_SIZE),
                fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.float32),
                )
            rcnn_cls_loss = tf.reduce_sum(rcnn_cls_loss)/tf.cast(self.config.BATCH_SIZE, dtype=tf.float32)

            rcnn_reg_loss = tf.map_fn(
                fn=lambda idx: rcnn_reg_loss_fn(
                    rcnn_reg_target[idx],
                    rcnn_reg_output[idx],
                    rcnn_cls_target[idx],
                    rcnn_train_indices[idx],
                    rcnn_pos_num[idx]),
                elems=tf.range(self.config.BATCH_SIZE),
                fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.float32),
                )
            rcnn_reg_loss = tf.reduce_sum(rcnn_reg_loss)/tf.cast(self.config.BATCH_SIZE, dtype=tf.float32)

            total_loss = rpn_cls_loss + rpn_reg_loss + rcnn_cls_loss + rcnn_reg_loss
            # total_loss = rpn_cls_loss + rpn_reg_loss*10

        # update gradient
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # update loss and metrics
        self.rpn_cls_acc_tracker.update_state(
            rpn_cls_target,
            tf.gather(rpn_cls_output, rpn_train_indices, batch_dims=1),
            )
        self.rpn_cls_loss_tracker.update_state(rpn_cls_loss)
        self.rpn_reg_loss_tracker.update_state(rpn_reg_loss)
        self.rcnn_cls_acc_tracker.update_state(
            rcnn_cls_target,
            tf.gather(rcnn_cls_output, rcnn_train_indices, batch_dims=1),
            )
        self.rcnn_cls_loss_tracker.update_state(rcnn_cls_loss)
        self.rcnn_reg_loss_tracker.update_state(rcnn_reg_loss)

        # pack return values
        ret = {
            "rpn_cls_acc": self.rpn_cls_acc_tracker.result(),
            "rpn_cls_loss": self.rpn_cls_loss_tracker.result(),
            "rpn_reg_loss": self.rpn_reg_loss_tracker.result(),
            "rcnn_cls_acc": self.rcnn_cls_acc_tracker.result(),
            "rcnn_cls_loss": self.rcnn_cls_loss_tracker.result(),
            "rcnn_reg_loss": self.rcnn_reg_loss_tracker.result(),
            }
        return ret

    def test_step(self, data):
        # batch data
        images, (gt_cls_ids, gt_boxes, num_gt_instance) = data

        # get outputs
        outputs = self(images, training=False)

        # unpack outputs
        rcnn_cls_output = outputs[2]  # [B*N, 21]
        rcnn_reg_output = outputs[3]  # [B*N, 20*4]
        roi_boxes = outputs[4]  # [B*N, 4]
        valid_num = outputs[5]  # [B]

        # reshape rcnn outputs
        rcnn_cls_output = tf.reshape(
            rcnn_cls_output,
            [self.config.BATCH_SIZE, self.config.RPN_NMS_NUM, self.config.NUM_CLASS+1])
        rcnn_reg_output = tf.reshape(
            rcnn_reg_output,
            [self.config.BATCH_SIZE, self.config.RPN_NMS_NUM, self.config.NUM_CLASS, 4])
        roi_boxes = tf.reshape(
            roi_boxes,
            [self.config.BATCH_SIZE, self.config.RPN_NMS_NUM, 4])

        # get rcnn targets
        rcnn_targets = tf.map_fn(
            fn=lambda idx: get_rcnn_target(
                roi_boxes[idx][:valid_num[idx]],
                gt_boxes[idx][:num_gt_instance[idx]],
                gt_cls_ids[idx][:num_gt_instance[idx]],
                valid_num[idx],
                pos_threshold=self.config.RCNN_POS_IOU_THOLD,
                neg_threshold=self.config.RCNN_NEG_IOU_THOLD,
                total_sample_num=self.config.RCNN_TOTAL_SAMPLE_NUM,
                pos_sample_ratio=self.config.RCNN_POS_SAMPLE_RATIO),
            elems=tf.range(self.config.BATCH_SIZE),
            fn_output_signature=(
                tf.TensorSpec(shape=[self.config.RCNN_TOTAL_SAMPLE_NUM], dtype=tf.int32),
                tf.TensorSpec(shape=[self.config.RCNN_TOTAL_SAMPLE_NUM, 4], dtype=tf.float32),
                tf.TensorSpec(shape=[self.config.RCNN_TOTAL_SAMPLE_NUM], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.int32),),
            parallel_iterations=self.config.BATCH_SIZE,
            )

        # unpack rcnn_targets
        rcnn_cls_target = rcnn_targets[0]
        rcnn_reg_target = rcnn_targets[1]
        rcnn_train_indices = rcnn_targets[2]
        rcnn_valid_num = rcnn_targets[3]
        rcnn_pos_num = rcnn_targets[4]

        rcnn_cls_loss = tf.map_fn(
            fn=lambda idx: rcnn_cls_loss_fn(
                rcnn_cls_target[idx],
                rcnn_cls_output[idx],
                rcnn_train_indices[idx],
                rcnn_valid_num[idx]),
            elems=tf.range(self.config.BATCH_SIZE),
            fn_output_signature=tf.float32,
            )
        rcnn_cls_loss = tf.reduce_sum(rcnn_cls_loss)/tf.cast(self.config.BATCH_SIZE, dtype=tf.float32)

        rcnn_reg_loss = tf.map_fn(
            fn=lambda idx: rcnn_reg_loss_fn(
                rcnn_reg_target[idx],
                rcnn_reg_output[idx],
                rcnn_cls_target[idx],
                rcnn_train_indices[idx],
                rcnn_pos_num[idx]),
            elems=tf.range(self.config.BATCH_SIZE),
            fn_output_signature=tf.float32,
            )
        rcnn_reg_loss = tf.reduce_sum(rcnn_reg_loss)/tf.cast(self.config.BATCH_SIZE, dtype=tf.float32)

        # update loss and metrics
        self.rcnn_cls_acc_tracker.update_state(
            rcnn_cls_target,
            tf.gather(rcnn_cls_output, rcnn_train_indices, batch_dims=1),
            )
        self.rcnn_cls_loss_tracker.update_state(rcnn_cls_loss)
        self.rcnn_reg_loss_tracker.update_state(rcnn_reg_loss)

        # pack return values
        ret = {
            # "rpn_cls_acc": self.rpn_cls_acc_tracker.result(),
            # "rpn_cls_loss": self.rpn_cls_loss_tracker.result(),
            # "rpn_reg_loss": self.rcnn_cls_acc_tracker.result(),
            "rcnn_cls_acc": self.rcnn_cls_acc_tracker.result(),
            "rcnn_cls_loss": self.rcnn_cls_loss_tracker.result(),
            "rcnn_reg_loss": self.rcnn_reg_loss_tracker.result(),
            }
        return ret


# if __name__ == '__main__':
#     # tf.config.run_functions_eagerly(True)

#     # create voc explore dataset
#     dataset_dir = 'VOC2007'
#     vocdata = VOCData(dataset_dir, batch_size=config.BATCH_SIZE)

#     inputs, outputs = build_faster_rcnn_graph()
#     model = FasterRCNN(inputs, outputs)

#     # model.summary()
#     model.compile()
#     model.fit(vocdata, epochs=100)