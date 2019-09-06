from glob import glob
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections
import re


def check_args(args):
    args.setting_file = args.checkpoint_dir + args.setting_file
    args.log_file = args.checkpoint_dir + args.log_file
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(args.setting_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print('------------ Options -------------')
        for k in args.__dict__:
            v = args.__dict__[k]
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print('%s: %s' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
        print('------------ End -------------')

    return args


def input_fn(train_dir, mpi_rank, max_seq_length=512, max_predictions_per_seq=85,
                     batch_size=32, is_training=True, num_cpu_threads=12):
    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if name == 'input_masks' or name == 'sen_masks':
                t = tf.cast(t, tf.float32)
            elif t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_masks":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "sen_masked_label":
            tf.FixedLenFeature([], tf.int64),
        "sen_masks":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "cls_label":
            tf.FixedLenFeature([], tf.int64)
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    input_files = glob(train_dir + '/*/*' + str(mpi_rank) + '.tfrecords')

    print('Parallel Rank', mpi_rank, 'training files:')
    for input_file in input_files:
        print(input_file)

    if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        d = d.repeat()
        d = d.shuffle(buffer_size=len(input_files))

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(input_files))

        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))
        d = d.shuffle(buffer_size=1024)
    else:
        d = tf.data.TFRecordDataset(input_files)
        # Since we evaluate for a fixed number of steps we don't want to encounter
        # out-of-range exceptions.
        d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_assigment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}
    new_variable_names = set()
    unused_variable_names = set()

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            if 'adam' not in name:
                unused_variable_names.add(name)
            continue
        # assignment_map[name] = name
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    for name in name_to_variable:
        if name not in initialized_variable_names:
            new_variable_names.add(name)
    return assignment_map, initialized_variable_names, new_variable_names, unused_variable_names


# loading weights
def init_from_checkpoint(init_checkpoint, tvars=None, rank=0):
    if not tvars:
        tvars = tf.trainable_variables()
    assignment_map, initialized_variable_names, new_variable_names, unused_variable_names \
        = get_assigment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    if rank == 0:
        # 显示成功加载的权重
        for t in initialized_variable_names:
            if ":0" not in t:
                print("Loading weights success: " + t)

        # 显示新的参数
        print('New parameters:', new_variable_names)

        # 显示初始化参数中没用到的参数
        print('Unused parameters', unused_variable_names)
