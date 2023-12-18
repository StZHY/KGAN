import tensorflow as tf
import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import roc_auc_score


class KGAN(object):
    def __init__(self, args, n_entity, n_relation):
        self._parse_args(args, n_entity, n_relation)
        self._build_inputs()
        self._build_embeddings()
        self._build_model(args)
        self._build_loss()
        self._build_train()

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.n_relations = args.n_relations
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.using_all_hops = args.using_all_hops

    
    def _build_inputs(self):
        self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name="labels")
        self.memories_h = []
        self.memories_r = []
        self.memories_t = []

        for hop in range(self.n_hop):
            self.memories_h.append(
                tf.placeholder(dtype=tf.int32, shape=[None, 7, self.n_memory], name="memories_h_" +str(hop))
            )
            self.memories_r.append(
                tf.placeholder(dtype=tf.int32, shape=[None, 7, self.n_memory], name="memories_r_" +str(hop))
            )
            self.memories_t.append(
                tf.placeholder(dtype=tf.int32, shape=[None, 7, self.n_memory], name="memories_t_" +str(hop))
            )

    
    def _build_embeddings(self):
        self.entity_emb_matrix = tf.get_variable(name="entity_emb_matrix", dtype=tf.float32,
                                                 shape=[self.n_entity, self.dim],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        last_entity_init = tf.get_variable(name="last_entity", dtype=tf.float32,
                                                 shape=[1, self.dim],
                                                 initializer=tf.zeros_initializer())
        self.entity_emb_matrix = tf.concat([self.entity_emb_matrix, last_entity_init], 0)
        self.relation_emb_matrix = tf.get_variable(name="relation_emb_matrix", dtype=tf.float32,
                                                   shape=[self.n_relation, self.dim, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())

    def _build_model(self, args):
        self.transform_matrix = tf.get_variable(name="transform_matrix", shape=[self.dim, self.dim], dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())

        # [batch size, dim]
        self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        
        for i in range(self.n_hop):

            memories_h = tf.reshape(self.memories_h[i], shape=[-1, args.n_memory])
            memories_r = tf.reshape(self.memories_r[i], shape=[-1, args.n_memory])
            memories_t = tf.reshape(self.memories_t[i], shape=[-1, args.n_memory])
            
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, memories_h))

            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, memories_r))

            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, memories_t))

        o_list = self.intra_inter_group_attention(args)

        self.scores = tf.squeeze(self.predict(self.item_embeddings, o_list))
        self.scores_normalized = tf.sigmoid(self.scores)


    def intra_inter_group_attention(self, args):
        o_list = []
        for hop in range(self.n_hop):

            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)

            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)

            Rh = tf.reshape(Rh, shape=[-1, args.n_relations, args.n_memory, args.dim])

            v = tf.expand_dims(self.item_embeddings, axis=1)

            v = tf.expand_dims(v, axis=-1)

            probs = tf.squeeze(tf.matmul(Rh, v), axis=3)

            probs=tf.reshape(probs, shape=[-1, args.n_memory])

            probs_normalized = tf.nn.softmax(probs)

            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)

            o = tf.reshape(o, shape=[-1, args.n_relations, args.dim])

            attention=Dense(
                units=args.dim, activation=None, name='aggregate_relation',
                kernel_regularizer = tf.contrib.layers.l2_regularizer(args.l2_weight)
            )(o)

            attention=Dropout(0.5)(attention)

            attention=Dense(
                units=1, activation='relu', name='activate_dense', use_bias = False,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(args.l2_weight)
            )(attention)

            attention=tf.squeeze(attention, axis=2)

            attention_weight_norm = tf.nn.softmax(attention, dim=-1)

            attention_weight_expand = tf.expand_dims(attention_weight_norm, axis=-1)

            o=tf.reduce_sum(o * attention_weight_expand, axis=1)

            self.item_embeddings = self.update_item_embedding(self.item_embeddings, o)
            o_list.append(o)
        return o_list

    def update_item_embedding(self, item_embeddings, o):

        item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix)

        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # [batch_size]
        scores = tf.reduce_sum(item_embeddings * y, axis=1)
        return scores

    def _build_loss(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))

        self.kge_loss = 0
        for hop in range(self.n_hop):
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
            t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, self.r_emb_list[hop]), t_expanded))
            self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
        self.kge_loss = -self.kge_weight * self.kge_loss

        self.l2_loss = 0
        for hop in range(self.n_hop):
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))
            self.l2_loss += tf.nn.l2_loss(self.transform_matrix)
        self.l2_loss = self.l2_weight * self.l2_loss

        self.loss = self.base_loss + self.kge_loss + self.l2_loss

    def _build_train(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc

    def top_k(self, sess, feed_dict):
        scores = sess.run(self.scores_normalized, feed_dict)
        return scores