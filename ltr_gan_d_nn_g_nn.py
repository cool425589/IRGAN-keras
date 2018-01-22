import os
import time
import _pickle as cPickle
import random
import tensorflow as tf
import numpy as np
from eval.precision import precision_at_k
from eval.ndcg import ndcg_at_k
from eval.map import MAP
from eval.mrr import MRR
import utils as ut
from dis_model_pointwise_nn import DIS
from gen_model_nn import GEN


FEATURE_SIZE = 46
HIDDEN_SIZE = 46
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.001
G_LEARNING_RATE = 0.001
TEMPERATURE = 0.2
LAMBDA = 0.5

workdir = 'MQ2008-semi'
DIS_TRAIN_FILE = workdir + '/run-train-gan.txt'
GAN_MODEL_BEST_FILE = workdir + '/gan_best_nn.model'
"""
print ('now - load')
query_url_feature, query_url_index, query_index_url =\
    ut.load_all_query_url_feature(workdir + '/Large_norm.txt', FEATURE_SIZE)
query_pos_train = ut.get_query_pos(workdir + '/train.txt')
query_pos_test = ut.get_query_pos(workdir + '/test.txt')
print ('end - end')
"""
#query id -> url -> feature
query_url_feature = np.load(workdir + '/query_url_feature.npy').item()
#query id -> url -> index
query_url_index = np.load(workdir + '/query_url_index.npy').item()
#query id -> url
query_index_url = np.load(workdir + '/query_index_url.npy').item()
#train_query id -> url(postive)
query_pos_train = np.load(workdir + '/query_pos_train.npy').item()
#test_query id -> url(postive)
query_pos_test = np.load(workdir + '/query_pos_test.npy').item()

def generate_for_d(generator, filename):
    data = []
    print('negative sampling for d using g ...')
    for query in query_pos_train:
        #get query all url (postive)
        pos_list = query_pos_train[query]
        #get query all url
        all_list = query_index_url[query]
        candidate_list = all_list
        #get all url feature
        candidate_list_feature = [query_url_feature[query][url] for url in candidate_list]
        candidate_list_feature = np.asarray(candidate_list_feature)
        """
        score = generator.get_score(candidate_list_feature[np.newaxis, :])
        score = score[0].reshape([-1])
        # softmax for all
        exp_rating = np.exp(score - np.max(score))
        prob = exp_rating / np.sum(exp_rating) 
        """
        prob = generator.get_prob(candidate_list_feature[np.newaxis, :])
        prob = prob[0]
        prob = prob.reshape([-1])
        # G generate some url (postive doc num)
        neg_list = np.random.choice(candidate_list, size = [len(pos_list)], p = prob)
        # list -> ( query id , pos url , neg url )
        for i in range(len(pos_list)):
            data.append((query, pos_list[i], neg_list[i]))
    #shuffle
    random.shuffle(data)
    with open(filename, 'w') as fout:
        #pos feature [tab] neg feature
        for (q, pos, neg) in data:
            fout.write(','.join([str(f) for f in query_url_feature[q][pos]])
                       + '\t'
                       + ','.join([str(f) for f in query_url_feature[q][neg]]) + '\n')
            fout.flush()


def main():
    #call discriminator, generator
    discriminator = DIS(FEATURE_SIZE, HIDDEN_SIZE, WEIGHT_DECAY, D_LEARNING_RATE)
    generator = GEN(FEATURE_SIZE, HIDDEN_SIZE, WEIGHT_DECAY, G_LEARNING_RATE, temperature=TEMPERATURE)
    print('start adversarial training')
    p_best_val = 0.0
    ndcg_best_val = 0.0
    for epoch in range(30):
        if epoch >= 0:
            # G generate negative for D, then train D
            print('Training D ...')
            for d_epoch in range(100):
                if d_epoch % 30 == 0:
                    generate_for_d(generator, DIS_TRAIN_FILE)
                    train_size = ut.file_len(DIS_TRAIN_FILE)
                index = 1
                while True:
                    if index > train_size:
                        break
                    if index + BATCH_SIZE <= train_size + 1:
                        input_pos, input_neg = ut.get_batch_data(DIS_TRAIN_FILE, index, BATCH_SIZE)
                    else:
                        input_pos, input_neg = ut.get_batch_data(DIS_TRAIN_FILE, index, train_size - index + 1)
                    index += BATCH_SIZE
                    pred_data = []
                    #prepare pos and neg data
                    pred_data.extend(input_pos)
                    pred_data.extend(input_neg)
                    pred_data = np.asarray(pred_data)
                    #prepara pos and neg label
                    pred_data_label = [1.0] * len(input_pos)
                    pred_data_label.extend([0.0] * len(input_neg))
                    pred_data_label = np.asarray(pred_data_label)
                    #train
                    discriminator.train(pred_data, pred_data_label)
        # Train G
        print('Training G ...')
        for g_epoch in range(10):
            start_time = time.time()
            print ('now_ G_epoch : ', str(g_epoch))
            for query in query_pos_train.keys():
                pos_list = query_pos_train[query]
                pos_set = set(pos_list)
                #all url
                all_list = query_index_url[query]
                #all feature
                all_list_feature = [query_url_feature[query][url] for url in all_list]
                all_list_feature = np.asarray(all_list_feature)
                # G generate all url prob
                prob = generator.get_prob(all_list_feature[np.newaxis, :])
                prob = prob[0]
                prob = prob.reshape([-1])
                #important sampling, change doc prob
                prob_IS = prob * (1.0 - LAMBDA)
            
                for i in range(len(all_list)):
                    if all_list[i] in pos_set:
                        prob_IS[i] += (LAMBDA / (1.0 * len(pos_list)))
                # G generate some url (5 * postive doc num)
                choose_index = np.random.choice(np.arange(len(all_list)), [5 * len(pos_list)], p=prob_IS)
                #choose url
                choose_list = np.array(all_list)[choose_index]
                #choose feature
                choose_feature = [query_url_feature[query][url] for url in choose_list]
                #prob / importan sampling prob (loss => prob * reward * prob / importan sampling prob) 
                choose_IS = np.array(prob)[choose_index] / np.array(prob_IS)[choose_index]
                choose_index = np.asarray(choose_index)
                choose_feature = np.asarray(choose_feature)
                choose_IS = np.asarray(choose_IS)
                #get reward((prob  - 0.5) * 2 )                
                choose_reward = discriminator.get_preresult(choose_feature)
                #train
                generator.train(choose_feature[np.newaxis, :], choose_reward.reshape([-1])[np.newaxis, :], choose_IS[np.newaxis, :])       
            print("train end--- %s seconds ---" % (time.time() - start_time))
            p_5 = precision_at_k(generator, query_pos_test, query_pos_train, query_url_feature, k=5)
            ndcg_5 = ndcg_at_k(generator, query_pos_test, query_pos_train, query_url_feature, k=5)            
            if p_5 > p_best_val:
                p_best_val = p_5
                ndcg_best_val = ndcg_5
                generator.save_model(GAN_MODEL_BEST_FILE)
                print("Best:", "gen p@5 ", p_5, "gen ndcg@5 ", ndcg_5)
            elif p_5 == p_best_val:
                if ndcg_5 > ndcg_best_val:
                    ndcg_best_val = ndcg_5
                    generator.save_model(GAN_MODEL_BEST_FILE)
                    print("Best:", "gen p@5 ", p_5, "gen ndcg@5 ", ndcg_5)           
if __name__ == '__main__':
    main()