import collections
import os
import numpy as np


def load_data(args):
    train_data, test_data, user_history_dict = load_rating(args)
    n_entity, n_relation, kg, relation_set = load_kg(args)
    aggregate_set = get_aggregate_set(args, kg, user_history_dict, relation_set, n_entity)
    return train_data, test_data, n_entity, n_relation, aggregate_set, relation_set


def load_rating(args):

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    return dataset_split(rating_np)


def dataset_split(rating_np):
    print('splitting dataset ...')

    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    test_indices = np.random.choice(n_ratings, size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(set(range(n_ratings)) - set(test_indices))

    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]

    train_data = rating_np[train_indices]
    test_data = rating_np[test_indices]

    return train_data, test_data, user_history_dict


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    relation_set = set(kg_np[:, 1])
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg, relation_set


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def get_aggregate_set(args, kg, user_history_dict, relation_set, n_entity):

    aggregate_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(args.n_hop):

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_aggregation_R = aggregate_set[user][-1]
                tails_of_last_hop=[]
                for relation in list(relation_set):
                    t=tails_of_aggregation_R[relation][2]
                    tails_of_last_hop = t + tails_of_last_hop

            _aggregation_R=[]
            for relation in list(relation_set):
                h=[]
                r=[]
                t=[]
                for entity in tails_of_last_hop:
                    if entity == n_entity:
                        continue
                    for tail_and_relation in kg[entity]:
                        if tail_and_relation[1] == relation:
                            h.append(entity)
                            r.append(tail_and_relation[1])
                            t.append(tail_and_relation[0])
                if len(r) == 0:
                    _aggregation_R.append([[n_entity],[relation],[n_entity]])
                else:
                    _aggregation_R.append([h,r,t])

            for relation in list(relation_set):
                replace = len(_aggregation_R[relation][0]) < args.n_memory
                indices = np.random.choice(len(_aggregation_R[relation][0]), size=args.n_memory, replace=replace)
                _aggregation_R[relation][0] = [_aggregation_R[relation][0][i] for i in indices]
                _aggregation_R[relation][1] = [_aggregation_R[relation][1][i] for i in indices]
                _aggregation_R[relation][2] = [_aggregation_R[relation][2][i] for i in indices]
            aggregate_set[user].append(_aggregation_R)

    return aggregate_set