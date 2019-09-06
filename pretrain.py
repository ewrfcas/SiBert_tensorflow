import argparse
import numpy as np
import tensorflow as tf
import blocksparse as bs
import os
from models.SiBert import bert_model
import horovod.tensorflow as hvd
from tf_ops.optimizer import Optimizer
from tf_ops.TrainingHook import LoggingTensorHook
from tf_ops.utils import check_args, input_fn

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'


def print_rank0(*args):
    if mpi_rank == 0:
        print(*args, flush=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    tf.logging.set_verbosity(tf.logging.FATAL)

    # training parameter
    parser.add_argument('--train_steps', type=int, default=1000000)
    parser.add_argument('--n_batch', type=int, default=32)  # real batch=n_batch*n_gpu
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--loss_scale', type=float, default=2.0 ** 15)
    parser.add_argument('--warmup_iters', type=int, default=10000)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--loss_count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=556)
    parser.add_argument('--float16', type=int, default=True)  # only sm >= 7.0 (tensorcores)
    parser.add_argument('--log_interval', type=int, default=100)  # show the average loss per 100 steps
    parser.add_argument('--save_interval', type=int, default=50000)
    parser.add_argument('--show_eval', type=bool, default=True)

    # model parameter
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_timesteps', type=int, default=512)
    parser.add_argument('--n_mask', type=int, default=80)
    parser.add_argument('--max_position_embeddings', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=20879, help='20879 for char model, 70000 for spm model')
    parser.add_argument('--type_vocab_size', type=int, default=2)
    parser.add_argument('--mlp_ratio', type=int, default=4)

    # data dir
    parser.add_argument('--train_dir', type=str, default='preprocessed_data/scp_cls_data_0826/tfrecords')
    parser.add_argument('--checkpoint_dir', type=str, default='check_points/SiBertV0/')
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')

    # use some global vars for convenience
    args = check_args(parser.parse_args())

    hvd.init()
    mpi_size = hvd.size()
    mpi_rank = hvd.local_rank()

    print_rank0('######## generating data ########')
    train_dataset = input_fn(args.train_dir, mpi_rank,
                             max_seq_length=args.n_timesteps,
                             max_predictions_per_seq=args.n_mask,
                             batch_size=args.n_batch)
    train_iterator = train_dataset.make_initializable_iterator()
    train_next_element = train_iterator.get_next()

    with tf.device("/gpu:0"):
        input_ids = tf.placeholder(tf.int32, shape=[None, args.n_timesteps], name='input_ids')
        input_masks = tf.placeholder(tf.float32, shape=[None, args.n_timesteps], name='input_masks')
        segment_ids = tf.placeholder(tf.int32, shape=[None, args.n_timesteps], name='segment_ids')
        masked_lm_positions = tf.placeholder(tf.int32, shape=[None, args.n_mask], name='masked_lm_positions')
        masked_lm_ids = tf.placeholder(tf.int32, shape=[None, args.n_mask], name='masked_lm_ids')
        masked_lm_weights = tf.placeholder(tf.float32, shape=[None, args.n_mask], name='masked_lm_weights')
        sen_choice_loc = tf.placeholder(tf.int32, shape=[None, ], name='sen_choice_loc')
        sen_choice_mask = tf.placeholder(tf.float32, shape=[None, args.n_timesteps], name='sen_choice_mask')
        cls_label = tf.placeholder(tf.int32, shape=[None, ], name='cls_label')

    # needed for bs.dropout()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    bs.set_entropy()

    # build the models for training and testing/validation
    print_rank0('######## init model ########')
    model = bert_model(config=args,
                       input_ids=input_ids,
                       mask=input_masks,
                       token_type_ids=segment_ids,
                       cloze_positions=masked_lm_positions,
                       cloze_ids=masked_lm_ids,
                       cloze_weights=masked_lm_weights,
                       sen_choice_loc=sen_choice_loc,
                       sen_choice_mask=sen_choice_mask,
                       cls_label=cls_label,
                       show_eval=args.show_eval,
                       train=True)

    optimization = Optimizer(loss=model.train_loss,
                             init_lr=args.lr,
                             num_train_steps=args.train_steps,
                             num_warmup_steps=args.warmup_iters,
                             hvd=hvd,
                             use_fp16=args.float16,
                             loss_count=args.loss_count,
                             clip_norm=args.clip_norm,
                             init_loss_scale=args.loss_scale,
                             beta1=args.beta1,
                             beta2=args.beta2)

    training_hooks = [hvd.BroadcastGlobalVariablesHook(0),
                      tf.train.StopAtStepHook(last_step=args.train_steps)]

    if hvd.rank() == 0:
        show_tensors = {'step': optimization.global_step,
                        'lr': optimization.learning_rate,
                        'Loss': model.train_loss}
        if args.show_eval:
            show_tensors['MLM-Acc'] = model.MLM_acc
            show_tensors['SCP-Acc'] = model.SCP_acc
        training_hooks.append(LoggingTensorHook(tensors=show_tensors,
                                                every_n_iter=args.log_interval,
                                                save_file=args.log_file))

    # Free up some python memory now that models are built
    bs.clear_bst_constants()

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(mpi_rank)
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    checkpoint_dir = args.checkpoint_dir if hvd.rank() == 0 else None

    # skip some batches to retrain the model
    jump_i = 0
    start_training_flag = False
    old_gsp = 0
    scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=10))
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           scaffold=scaffold,
                                           hooks=training_hooks,
                                           save_checkpoint_steps=args.save_interval,
                                           log_step_count_steps=50000,
                                           config=config) as sess:
        sess.run(train_iterator.initializer)
        while not sess.should_stop():
            batch_data, gsp = sess.run([train_next_element, optimization.global_step])
            if jump_i < gsp:
                jump_i += 1
                if jump_i % 10000 == 0:
                    print_rank0('jump', jump_i)
                continue
            else:
                start_training_flag = True

            if start_training_flag and old_gsp == gsp:
                print_rank0('loss NAN or INF in', old_gsp)

            old_gsp = gsp
            feed_data = {input_ids: batch_data['input_ids'],
                         input_masks: batch_data['input_masks'],
                         segment_ids: batch_data['segment_ids'],
                         masked_lm_positions: batch_data['masked_lm_positions'],
                         masked_lm_ids: batch_data['masked_lm_ids'],
                         masked_lm_weights: batch_data['masked_lm_weights'],
                         sen_choice_loc: batch_data['sen_masked_label'],
                         sen_choice_mask: batch_data['sen_masks'],
                         cls_label: batch_data['cls_label']}
            sess.run(optimization.train_op, feed_dict=feed_data)
            jump_i += 1

        print('Rank', mpi_rank, 'Over!!!')
