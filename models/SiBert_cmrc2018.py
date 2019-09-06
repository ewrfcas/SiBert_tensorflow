from .bert_layers import *
import tensorflow as tf
import blocksparse as bs


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
       float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


def get_custom_getter(compute_type):
    return float32_variable_storage_getter if compute_type == tf.float16 else None


class bert_model(object):
    def __init__(self,
                 config,
                 input_ids,
                 mask,
                 token_type_ids,
                 start_position=None,
                 end_position=None,
                 train=False):
        self.config = config

        with tf.variable_scope("bert_model", reuse=tf.AUTO_REUSE,
                               custom_getter=get_custom_getter(tf.float16 if config.float16 else tf.float32)):

            with tf.device("/gpu:0"):

                # embed discrete inputs to continous space and add learned position embeddings
                with tf.variable_scope('embedding'):
                    word_embedding = tf.get_variable("word_embedding",
                                                     [self.config.vocab_size, self.config.hidden_size],
                                                     dtype=tf.float16 if config.float16 else tf.float32, trainable=True,
                                                     initializer=create_initializer())
                    position_embedding = tf.get_variable('position_embedding',
                                                         [self.config.max_position_embeddings, self.config.hidden_size],
                                                         dtype=tf.float16 if config.float16 else tf.float32,
                                                         trainable=True,
                                                         initializer=create_initializer())
                    token_type_embedding = tf.get_variable('token_type_embedding',
                                                           [self.config.type_vocab_size, self.config.hidden_size],
                                                           dtype=tf.float16 if config.float16 else tf.float32,
                                                           trainable=True,
                                                           initializer=create_initializer())

                    # bs.embedding_lookup can be much faster than tf version for low entropy indexes or small vocabs
                    (batch_size, seq_length) = shape_list(input_ids)
                    x = bs.embedding_lookup(word_embedding, input_ids)
                    pos_ids = tf.expand_dims(tf.range(seq_length), 0)
                    pos_ids = tf.tile(pos_ids, (batch_size, 1))
                    pos = bs.embedding_lookup(position_embedding, pos_ids)
                    tok = bs.embedding_lookup(token_type_embedding, token_type_ids)

                    x = x + pos + tok
                    x = self.layer_norm(x, name='LayerNorm_emb')

                    if train and self.config.dropout > 0.0:
                        x = tf.nn.dropout(x, rate=self.config.dropout)

                # transformer
                with tf.variable_scope('transformers'):
                    masks = tf.reshape(mask, (-1, 1, 1, mask.shape[1]))  # [bs, len]->[bs, 1, 1, len]
                    for l in range(self.config.n_layer):
                        layer_name = 'layer_%d' % l
                        x = self.transformer_block(x, masks, layer_name, train=train)
                    self.sequence_output = x

                # useless in the cmrc2018 MRC task
                # with tf.variable_scope('pooler'):
                #     self.pooled_output = self.conv1d(self.sequence_output,
                #                                      'pooler_dense',
                #                                      self.config.hidden_size)
                #     self.pooled_output = bs.tanh(self.pooled_output)
                #
                #     # CLS pooler
                #     self.cls_pooled_output = self.conv1d(self.sequence_output[:, 0, :],  # [bs, dim]
                #                                          'cls_pooler_dense',
                #                                          self.config.hidden_size)
                #     self.cls_pooled_output = bs.tanh(self.cls_pooled_output)

                # finetune mrc
                with tf.variable_scope('finetune_mrc'):
                    # [bs, len]
                    self.start_logits = tf.squeeze(self.conv1d(self.sequence_output, 'start_dense', 1), -1)
                    self.end_logits = tf.squeeze(self.conv1d(self.sequence_output, 'end_dense', 1), -1)
                    self.start_logits += tf.cast(-10000. * (1 - mask), self.start_logits.dtype)
                    self.end_logits += tf.cast(-10000. * (1 - mask), self.end_logits.dtype)

                    if train and start_position is not None and end_position is not None:
                        start_loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=tf.cast(self.start_logits, tf.float32),
                            labels=start_position)
                        end_loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=tf.cast(self.end_logits, tf.float32),
                            labels=end_position)
                        start_loss = tf.reduce_mean(start_loss_)
                        end_loss = tf.reduce_mean(end_loss_)
                        self.train_loss = (start_loss + end_loss) / 2.0

    def transformer_block(self, x, masks, scope, train=False):
        """
        core component of transformer
        performs attention + residual mlp + layer normalization
        """
        n_state = x.shape[-1].value

        with tf.variable_scope(scope):

            with tf.variable_scope('self_attention'):
                q = self.conv1d(x, 'proj_q', n_state)
                k = self.conv1d(x, 'proj_k', n_state)
                v = self.conv1d(x, 'proj_v', n_state)

                q = split_heads(q, self.config.n_head)
                k = split_heads(k, self.config.n_head)
                v = split_heads(v, self.config.n_head)

                qk = tf.matmul(q, k, transpose_b=True)  # [bs, head, len, len]
                qk += tf.cast(-10000. * (1 - masks), qk.dtype)
                qk = bs.softmax(qk, scale=1.0 / np.sqrt(n_state / self.config.n_head))
                qkv = tf.matmul(qk, v)  # [bs, head, len, dim]
                att = merge_heads(qkv)  # [bs, len, dim*head]

                # This is actually dropping out entire tokens to attend to, which might
                # seem a bit unusual, but is taken from the original Transformer paper.
                if train and self.config.dropout > 0.0:
                    att = tf.nn.dropout(att, rate=self.config.dropout)

            # Run a linear projection of `hidden_size` then add a residual
            # with `layer_input`.
            with tf.variable_scope('attention_output'):
                att = self.conv1d(att, 'proj_a', n_state)
                if train and self.config.dropout > 0.0:
                    att = tf.nn.dropout(att, rate=self.config.dropout)
                x1 = bs.add(att, x)
                x1 = self.layer_norm(x1, name='LayerNorm_att')

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope('intermediate'):
                x2 = self.conv1d(x1, 'proj_m1', n_state * self.config.mlp_ratio, fast_gelu=True)

            with tf.variable_scope('output'):
                x2 = self.conv1d(x2, 'proj_m2', n_state)
                if train and self.config.dropout > 0.0:
                    x2 = tf.nn.dropout(x2, rate=self.config.dropout)
                x = bs.add(x2, x1)
                x = self.layer_norm(x, name='LayerNorm_output')

            return x

    def layer_norm(self, x, name, epsilon=1e-5, relu=False):
        """
        normalize state vector to be zero mean / unit variance + learned scale/shift
        """
        n_state = x.shape[-1].value
        with tf.variable_scope(name):
            gain = tf.get_variable('gamma', [n_state], initializer=tf.constant_initializer(1.0))
            bias = tf.get_variable('beta', [n_state], initializer=tf.constant_initializer(0.0))

            return bs.layer_norm(x, gain, bias, axis=-1, epsilon=epsilon, relu=relu)

    def conv1d(self, x, scope, channel, std=0.02, relu=False, fast_gelu=False, bias=True):
        with tf.variable_scope(scope):
            nx = x.shape[-1].value
            ndims = x.shape.ndims

            # Note: param initializers are not particularly well tuned in this code
            w = tf.get_variable("w", [nx, channel], initializer=create_initializer(initializer_range=std),
                                dtype=tf.float16 if self.config.float16 else tf.float32)
            if bias:
                b = tf.get_variable("bias", [channel], initializer=tf.constant_initializer(0.0))
            else:
                b = 0

            # merge context and batch dims for more efficient matmul
            if ndims > 2:
                y_shape = tf.concat([tf.shape(x)[: ndims - 1], [channel]], axis=0)
                x = tf.reshape(x, [-1, nx])

            y = tf.matmul(x, w)

            # avoid atomics in bias grad, but be careful as tf handles temp memory badly in the presense of async ops like all-reduce
            y = bs.bias_relu(y, b, relu=relu, fast_gelu=fast_gelu, atomics=False)

            if ndims > 2:
                y = tf.reshape(y, y_shape)

            return y
