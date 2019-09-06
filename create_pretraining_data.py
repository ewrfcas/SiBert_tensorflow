"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tensorflow as tf
from tqdm import tqdm
import os
import numpy as np
import copy
import horovod.tensorflow as hvd
import json

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file",
                    default=None,
                    help="Input raw text file (or comma-separated list of files).")

flags.DEFINE_integer("stage",
                     default=1,
                     help="stage1:loading documents,stage2:writing to tfrecords")

flags.DEFINE_integer("split_num",
                     default=4,
                     help="The split num of one txt")

flags.DEFINE_string("tokenizer_type",
                    default='char',
                    help="'char','spm'")

flags.DEFINE_string(
    "documents_dir",
    default='preprocessed_data/scp_cls_data_0826/documents',
    help="Output documents examples file.")

flags.DEFINE_string(
    "tfrecords_dir",
    default='preprocessed_data/scp_cls_data_0826/tfrecords',
    help="Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file",
                    default='vocab_file/vocab_20879_sen_mask.txt',
                    help="The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 80,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 5566, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 4,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_integer("parallel_num", 8, help='equals to ngpu number')

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")

flags.DEFINE_float(
    "max_sen_mask", 10,
    "Maximum number of sentence masked per sequence")

if FLAGS.tokenizer_type == 'char':
    import tokenization_char as tokenization
elif FLAGS.tokenizer_type == 'spm':
    import tokenization_spm as tokenization
else:
    raise NotImplementedError


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, sen_mask,
                 sen_masked_label, cls_label):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.sen_mask = sen_mask
        self.sen_masked_label = sen_masked_label
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.cls_label = cls_label

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "sen_mask: %s\n" % (" ".join([str(x) for x in self.sen_mask]))
        s += "sen_masked_label: %s\n" % (str(self.sen_masked_label))
        s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "cls_label: %s" % (str(self.cls_label))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def print_rank0(*args):
    if mpi_rank == 0:
        print(*args, flush=True)


def write_instance_to_example_files(writers, instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq):
    """Create TF example files from `TrainingInstance`s."""

    writer_index = 0
    total_written = 0
    for (inst_index, instance) in enumerate(tqdm(instances)):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        sen_mask = list(instance.sen_mask)
        sen_masked_label = instance.sen_masked_label
        cls_label = instance.cls_label

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            sen_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_masks"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)

        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        features["sen_masks"] = create_int_feature(sen_mask)
        features["sen_masked_label"] = create_int_feature([sen_masked_label])
        features["cls_label"] = create_int_feature([cls_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20 and mpi_rank == 0:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info("%s: %s" % (feature_name, " ".join([str(x) for x in values])))
            sd = dict()
            for j in range(len(sen_mask)):
                sd[j] = sen_mask[j]
            tf.logging.info("sen_span: %s" % str(sd))

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(input_files, tokenizer, rng):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]
    printed_dict = set()
    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)
                if len(all_documents) % 10000 == 0 and len(all_documents) not in printed_dict:
                    printed_dict.add(len(all_documents))
                    print('documents added:', len(all_documents))

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)
    return all_documents


def create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob, masked_lm_prob,
                                   max_predictions_per_seq, vocab_words, rng, max_sen_mask=10):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:  # 为了同一doc存在更多的无法容纳的句子，因此需要更高的short概率
        target_seq_length = rng.randint(16, max_num_tokens)

    instances = []
    current_chunk = []
    segment_idxs = set()  # build a set to save the index of samples of one doc
    current_length = 0
    i = 0
    start_i = 0
    while i < len(document):
        segment = document[i]  # get the i sentence of the doc
        if len(current_chunk) == 0:
            start_i = i
        current_chunk.append(segment)
        segment_idxs.add(i)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk and sum([len(cc) for cc in current_chunk]) > 0:
                # SI strategy: 55% select one sentence from current_chunk, 45% select one sentence from random passage as the query
                # in the random passage selecting, 16.75% select the sentence in the diff doc, and others in the same doc but diff passage
                # in the current chunk selecting, [MASK_SEN] will randomly insert among the passage sentences, one sentence will be replaced with REAL[MASK]
                random_idx = None
                sen_mask = []
                sen_masked_label = None
                other_sentences_idxs = []
                for s_idx in range(len(document)):
                    if s_idx not in segment_idxs:
                        other_sentences_idxs.append(s_idx)
                random_sentence_idx = None

                the_first_choice = rng.random()
                if len(current_chunk) == 1:
                    break
                if the_first_choice < 0.45:  # 45%选择外部句子
                    is_random_sen = True
                    sen_masked_label = 0  # scp位置为0

                    if the_first_choice < 0.1675 or len(other_sentences_idxs) == 0:  # 由于other_sentence!=0的比较少，所以实际占比会变高
                        cls_label = 2
                        # 并且保证不是同一篇doc的
                        random_document_index = 0
                        for _ in range(10):
                            random_document_index = rng.randint(0, len(all_documents) - 1)
                            if random_document_index != document_index:
                                break
                        random_document = all_documents[random_document_index]
                        random_sentence = random_document[rng.randint(0, len(random_document) - 1)]

                    else:  # 35%在同一doc但是不在内部的句子，实际比例和cls_label=2应该1:1
                        cls_label = 1
                        random_sentence_idx = rng.sample(other_sentences_idxs, 1)[0]
                        random_sentence = document[random_sentence_idx]

                else:  # 55%选择内部句子
                    cls_label = 0
                    is_random_sen = False
                    random_idx = rng.randint(0, len(current_chunk) - 1)
                    random_sentence = current_chunk[random_idx]
                    current_chunk[random_idx] = ['REAL_[MASK_SEN]']

                # 随机添加[MASK_SEN], 最多有len(current_chunk)+1个地方可以加入 |_|_|
                # 决定额外的MASK_SEN数, random_idx前后已经没有必要再插入[MASK_SEN]了。 len(current_chunk)+1->len(current_chunk)-1
                if is_random_sen:
                    mask_sen_num = rng.randint(1, len(current_chunk) + 1)
                elif len(current_chunk) > 1:
                    mask_sen_num = rng.randint(1, len(current_chunk) - 1)
                else:
                    mask_sen_num = 0

                mask_sen_num = int(min(max_sen_mask, mask_sen_num))
                mask_pos = list(np.arange(len(current_chunk) + 1))
                if random_idx is not None:
                    mask_pos = [p for p in mask_pos if p != random_idx and p != random_idx + 1]

                # 这里要注意，如果选择同一doc其他的句子，有可能出现在开头以及末尾，因此如果遇到这种情况，
                # 开头or末尾是不能加[mask_sen]的
                if random_sentence_idx == i + 1:
                    mask_pos.remove(len(current_chunk))
                    mask_sen_num -= 1
                elif random_sentence_idx == start_i - 1:
                    mask_pos.remove(0)
                    mask_sen_num -= 1

                # 长度限制，max_num_tokens已经减去了3，mask_sen_num是外加的[MASK_SEN]，
                # len(random_sentence)为随机抽取的句子长度
                instance_length_limit = max_num_tokens - mask_sen_num - len(random_sentence)

                if instance_length_limit <= 0:
                    # 极少情况可能发生，mask_sen数量比token数还多
                    current_chunk = []
                    current_length = 0
                    continue

                current_chunk = truncate_seq_pair(current_chunk, instance_length_limit, rng)

                if len(mask_pos) > 0:
                    mask_pos = set(np.random.choice(mask_pos, mask_sen_num, replace=False))
                else:
                    mask_pos = set()
                sen_tokens = []
                for i_c, sen_ in enumerate(current_chunk):
                    if i_c in mask_pos and len(sen_) > 0:
                        if len(sen_tokens) > 0 and sen_tokens[-1] != 'REAL_[MASK_SEN]' \
                                and sen_tokens[-1] != '[MASK_SEN]' and sen_ != 'REAL_[MASK_SEN]':
                            sen_tokens.append('[MASK_SEN]')
                    if len(sen_) > 0:
                        sen_tokens.extend(sen_)
                if len(current_chunk) in mask_pos and \
                        sen_tokens[-1] != 'REAL_[MASK_SEN]' and sen_tokens[-1] != 'REAL_[MASK_SEN]':
                    sen_tokens.append('[MASK_SEN]')

                # 保险处理，确保在特殊情况下首位无mask_sen
                if random_sentence_idx == i + 1 and sen_tokens[-1] == '[MASK_SEN]':
                    sen_tokens = sen_tokens[:-1]
                if random_sentence_idx == start_i - 1 and sen_tokens[0] == '[MASK_SEN]':
                    sen_tokens = sen_tokens[1:]

                assert max_num_tokens >= len(sen_tokens) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                # [CLS]是sentence mask的label
                sen_mask.append(1)
                segment_ids.append(0)
                for token in random_sentence:
                    tokens.append(token)
                    segment_ids.append(0)
                    sen_mask.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)
                sen_mask.append(0)

                for token in sen_tokens:
                    if '[MASK_SEN]' in token:  # [MASK_SEN] or REAL_[MASK_SEN]
                        sen_mask.append(1)
                    else:
                        sen_mask.append(0)
                    if token == 'REAL_[MASK_SEN]':  # 说明该处为真实的sentence的位置
                        sen_masked_label = len(tokens)
                        token = '[MASK_SEN]'
                    tokens.append(token)
                    segment_ids.append(1)

                tokens.append("[SEP]")
                segment_ids.append(1)
                sen_mask.append(0)

                assert len(sen_mask) == len(tokens)
                assert len(tokens) <= max_seq_length

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(tokens,
                                                                                               masked_lm_prob,
                                                                                               max_predictions_per_seq,
                                                                                               vocab_words, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    sen_mask=sen_mask,
                    sen_masked_label=sen_masked_label,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                    cls_label=cls_label)
                instances.append(instance)

            current_chunk = []
            segment_idxs = set()
            current_length = 0
        i += 1

    return instances


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]" or token == "[MASK_SEN]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    masked_lm = collections.namedtuple("masked_lm", ["index", "label"])  # pylint: disable=invalid-name

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(masked_lm(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(sentences, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""

    sentences2 = copy.deepcopy(sentences)

    while True:
        total_length = sum([len(sen) for sen in sentences2])
        if total_length <= max_num_tokens:
            break

        assert len(sentences2) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            i = 0
            while len(sentences2[i]) == 0 or sentences2[i][0] == 'REAL_[MASK_SEN]':
                i += 1
            del sentences2[i][0]
        else:
            i = len(sentences2) - 1
            while len(sentences2[i]) == 0 or sentences2[i][0] == 'REAL_[MASK_SEN]':
                i -= 1
            sentences2[i].pop()

    return sentences2


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    split_num = FLAGS.split_num
    os.makedirs('/'.join(FLAGS.documents_dir.split('/')[0:-1]), exist_ok=True)
    os.makedirs(FLAGS.tfrecords_dir, exist_ok=True)

    rng = random.Random(FLAGS.random_seed)

    # first stage，load documents and shuffle
    tf.logging.info("*** Reading from input files ***")
    tf.logging.info("  %s", FLAGS.input_file)

    all_documents = create_training_instances([FLAGS.input_file], tokenizer, rng)
    once_num = len(all_documents) // split_num
    for i in range(split_num):
        print('Writing split', i + 1, '...')
        with open(FLAGS.documents_dir + '/documents_' + FLAGS.input_file.split('/')[-1].split('.txt')[0]
                  + '_split' + str(i) + '.json', 'w') as w:
            for doc in tqdm(all_documents[i * once_num:(i + 1) * once_num]):
                w.write(json.dumps(doc, ensure_ascii=False) + '\n')

    # second stage，get multi epoch data
    # save each epoch data in ONE FOLDER WITH several files for parallel
    hvd.init()
    mpi_size = hvd.size()
    mpi_rank = hvd.local_rank()

    cls_dict = {0: 0,
                1: 0,
                2: 0}

    input_file = FLAGS.documents_dir + '/documents_' + FLAGS.input_file.split('/')[-1].split('.txt')[0] \
                 + '_split' + str(mpi_rank) + '.json'

    all_documents = []
    with open(input_file, 'r') as f:
        for line in tqdm(f):
            all_documents.append(json.loads(line.strip()))

    output_file = os.path.join(FLAGS.tfrecords_dir, input_file.split('/')[-1].split('.json')[0])

    if mpi_rank == 0:
        tf.logging.info("*** Writing to output files ***")
        tf.logging.info("  %s", output_file)
    os.makedirs(output_file, exist_ok=True)
    vocab_words = list(tokenizer.vocab.keys())
    tfr_dirs = [os.path.join(output_file, 'parallel' + str(ip) + '.tfrecords') for ip in range(FLAGS.parallel_num)]
    writers = []
    for tfr_dir in tfr_dirs:
        writers.append(tf.python_io.TFRecordWriter(tfr_dir))

    for ir in range(FLAGS.dupe_factor):
        instances = []
        print_rank0('Repeat', ir + 1)
        for document_index in tqdm(range(len(all_documents))):
            instances.extend(
                create_instances_from_document(all_documents, document_index, FLAGS.max_seq_length,
                                               FLAGS.short_seq_prob, FLAGS.masked_lm_prob,
                                               FLAGS.max_predictions_per_seq, vocab_words, rng))

        for ins in instances:
            cls_dict[ins.cls_label] += 1
        print(cls_dict)
        rng.shuffle(instances)
        write_instance_to_example_files(writers, instances, tokenizer, FLAGS.max_seq_length,
                                        FLAGS.max_predictions_per_seq)

    for writer in writers:
        writer.close()
