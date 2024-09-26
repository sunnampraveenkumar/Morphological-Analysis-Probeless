import numpy as np
import pickle
INF = np.inf
EPOCHS=10
BATCH_SIZE=32
ABLATION_BATCH_SIZE=8
LEARNING_RATE=0.01
BERT_OUTPUT_DIM = 768
SEED=1
SUBSET_SIZE = 150
# VOCAB_SIZE = 28996
VOCAB_SIZE = 119547
ABLATION_NUM_SENTENCES = 6500
ABLATION_NUM_TOKENS = 100000
UM_FEATS = ["id", "form", "lemma", "upos", "xpos", "um_feats", "head", "deprel", "deps", "misc"]

language_map = {
    'eng': 'en',
    'chi': 'zh',
    'hin': 'hi',
    'heb': 'he',
    'jap': 'ja'
}

train_paths = {'eng': 'en_ewt-um-train.conllu',
               'chi': 'zh_gsd-um-train.conllu',
               'hin': 'hi_hdtb-um-train.conllu',
               'heb': 'he_htb-um-train.conllu',
               'jap': 'ja_gsd-um-train.conllu'}

dev_paths = {'eng': 'data/UM/en/en_ewt-um-dev.conllu',
             'chi': 'data/UM/zh/zh_gsd-um-dev.conllu',
             'hin': 'data/UM/hi/hi_hdtb-um-dev.conllu',
             'heb': 'data/UM/he/he_htb-um-dev.conllu',
             'jap': 'data/UM/ja/ja_gsd-um-dev.conllu',
             'bul': 'data/UM/bg/bg_btb-um-dev.conllu',
             'tur': 'data/UM/tr/tr_imst-um-dev.conllu',
             'spa': 'data/UM/es/es_gsd-um-dev.conllu',
             'fra': 'data/UM/fr/fr_gsd-um-dev.conllu'}

test_paths = {'eng': 'data/UM/en/en_ewt-um-test.conllu',
              'chi': 'data/UM/zh/zh_gsd-um-test.conllu',
              'hin': 'data/UM/hi/hi_hdtb-um-test.conllu',
              'heb': 'data/UM/he/he_htb-um-test.conllu',
              'jap': 'data/UM/ja/ja_gsd-um-test.conllu',
              'bul': 'data/UM/bg/bg_btb-um-test.conllu',
              'tur': 'data/UM/tr/tr_imst-um-test.conllu',
              'spa': 'data/UM/es/es_gsd-um-test.conllu',
              'fra': 'data/UM/fr/fr_gsd-um-test.conllu'}

model_names = {'bert':'bert-base-multilingual-cased',
               'xlm': 'xlm-roberta-base',
               'eng-bert':'bert-base-cased',
               'spa-bert':'dccuchile/bert-base-spanish-wwm-cased',
               'fra-bert':'camembert-base'}