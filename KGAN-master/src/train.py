import tensorflow as tf
import numpy as np
from model_transD import KGAN
import logging


logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def train(args, data_info, show_loss):
    train_data = data_info[0]
    test_data = data_info[1]
    n_entity = data_info[2]
    n_relation = data_info[3]
    aggregate_set = data_info[4]
    relation_set = data_info[5]

    model = KGAN(args, n_entity, n_relation)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epoch):
            # training
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]:
                _, loss = model.train(
                    sess, get_feed_dict(args, model, train_data, aggregate_set, relation_set, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss))

            # evaluation
            train_auc, train_acc = evaluation(sess, args, model, train_data, aggregate_set, relation_set, args.batch_size)
            test_auc, test_acc = evaluation(sess, args, model, test_data, aggregate_set, relation_set, args.batch_size)

            print('epoch %d    train auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                  % (step, train_auc, train_acc, test_auc, test_acc))

            if args.show_topk:
                topk_eval(args, sess, model, train_data, test_data, aggregate_set, relation_set)


def get_feed_dict(args, model, data, aggregate_set, relation_set, start, end):
    feed_dict = dict()
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    
    for i in range(args.n_hop):
        m_h=[]
        m_r=[]
        m_t=[]
        for user in data[start:end, 0]:
            h=[]
            r=[]
            t=[]
            for relation in list(relation_set):
                h.append(aggregate_set[user][i][relation][0])
                r.append(aggregate_set[user][i][relation][1])
                t.append(aggregate_set[user][i][relation][2])
            m_h.append(h)
            m_r.append(r)
            m_t.append(t)
        
        feed_dict[model.memories_h[i]] = m_h
        feed_dict[model.memories_r[i]] = m_r
        feed_dict[model.memories_t[i]] = m_t

    return feed_dict


def evaluation(sess, args, model, data, aggregate_set, relation_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    while start < data.shape[0]:
        auc, acc = model.eval(sess, get_feed_dict(args, model, data, aggregate_set, relation_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list))

def topk_eval(args, sess, model, train_data, test_data, aggregate_set, relation_set):
    k_list=[1, 5, 10, 20, 50]
    recall_list = {k: [] for k in k_list}
    precision_list = {k: [] for k in k_list}
    f1_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    item_set = set(train_data[:, 1].tolist() + test_data[:, 1].tolist())
    train_record = _get_user_record(args, train_data, True)
    test_record = _get_user_record(args, test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))

    for user in user_list:
        test_item_list = list(item_set - set(train_record[user]))
        item_score_map = dict()
        
        input_data = _get_took_feed_data(user, test_item_list)
        scores = model.top_k(sess, get_feed_dict(args, model, input_data, aggregate_set, relation_set, 0, len(test_item_list)))
        for item, score in zip(test_item_list, scores):
            item_score_map[item] = score
        
        item_score_pair_sorted = sorted(item_score_map.items(), key = lambda x: x[1], reverse = True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            #recall@k
            recall = hit_num / len(set(test_record[user]))
            recall_list[k].append(recall)
            #precision@k
            precision = hit_num / len(set(item_sorted[:k]))
            precision_list[k].append(precision)
            #f1@k
            if (precision+recall) == 0:
                f1 =0
            else:
                f1 = 2*(recall*precision) / (precision + recall)
            f1_list[k].append(f1)
            #ndcg@k
            pos_items = list(test_record[user])
            rank_list = item_sorted[:k]
            ndcg = getNDCG(rank_list, pos_items)
            ndcg_list[k].append(ndcg)
    recall_dict = [np.mean(recall_list[k]) for k in k_list]
    precision_dict = [np.mean(precision_list[k]) for k in k_list]
    f1_dict = [np.mean(f1_list[k]) for k in k_list]
    ndcg_dict = [np.mean(ndcg_list[k]) for k in k_list]
    
    _show_recall_info(zip(k_list, recall_dict))
    _show_recall_info(zip(k_list, precision_dict))
    _show_recall_info(zip(k_list, f1_dict))
    _show_recall_info(zip(k_list, ndcg_dict))



def _get_user_record(args, data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def _get_took_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user, item, 0])
    return np.array(res)

def _show_recall_info(recall_zip):
    res = ""
    for i, j in recall_zip:
        res += "K@%d:%.4f    "%(i, j)
    logging.info(res)

def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    if len(pos_items) == 1:
        it2rel = {pos_items[0] : 1}
    else :
        it2rel = {it:r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype = np.float32)

    idcg = getDCG(relevance)
    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0
    else:
        ndcg = dcg / idcg
        return ndcg


def getDCG(scores):
    dcg = np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype = np.float32) + 2)), dtype = np.float32)
    return dcg