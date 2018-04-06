import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

sentences = ["나 고양이 좋다",
             "나 개 싫다",
             "너 고양이 싫다",
             "너 개 좋다",
             "너 너무 좋다",
             "너 정말 싫다",
             "나 너무 싫다",
             "나 정말 좋다",
             "너 좋다", "나 좋다", "너 싫다", "나 싫다"]

class word2vec:
    def __init__(self, corpus=None, embedding_size=2, window=1, iteration=10000, sg=1):
        self.corpus = corpus
        self.embedding_size = embedding_size
        self.window = window
        self.iteration = iteration
        self.sg = sg
        self.batch_size = 10
        self.ns_sample = self.batch_size

        self.word_list = []
        self.word_dic = defaultdict(lambda: [0, 0])  # [word_idx, word_frequency]
        self.word_count()
        self.num_classes = len(self.word_list)
        self.batch_x = []
        self.batch_y = []
        self.make_batch()
        self.sess = tf.Session()

        self.x = tf.placeholder(dtype=tf.int32, shape=[None], name="input")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="output")

        self._embeddings = tf.get_variable(name='embeddings_input', shape=[self.num_classes, self.embedding_size],
                                           dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        self.embed = tf.nn.embedding_lookup(params=self._embeddings, ids=self.x)

        self.nce_weights = tf.Variable(tf.random_uniform([self.embedding_size, self.num_classes], -1.0, 1.0))
        self.nce_biases = tf.Variable(tf.zeros([self.num_classes]))

        self.logit = tf.add(tf.matmul(self.embed, self.nce_weights), self.nce_biases)
        # self.total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     labels=tf.one_hot(self.y, depth=self.num_classes),logits=self.logit))

        # self.pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=tf.ones_like(tf.squeeze(tf.one_hot(self.y, depth=self.num_classes), 1)),
        #     logits=self.logit)
        # self.neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=tf.zeros_like(tf.one_hot(self.negative_sampling(self.x), depth=self.num_classes)),
        #     logits=self.logit)
        # self.total_loss = tf.reduce_mean(tf.add(tf.reduce_sum(self.pos_loss, 1), tf.reduce_sum(self.neg_loss, 1)))

        self.pos_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.ones_like(tf.squeeze(tf.one_hot(self.y, depth=self.num_classes), 1)),
            logits=self.logit)
        self.neg_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.zeros_like(tf.one_hot(self.negative_sampling(self.x), depth=self.num_classes)),
            logits=self.logit)
        self.total_loss = tf.add(tf.reduce_mean(self.pos_loss), tf.reduce_mean(self.neg_loss))

        self.tr_loss_hist = []
        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.total_loss)
        self.train()

    def word_count(self):
        for sentence in self.corpus:
            for word in sentence.split():
                self.word_list.append(word)
                self.word_dic[word][1] += 1
                self.word_list = list(set(self.word_list))
        for idx, word in enumerate(self.word_list):
            self.word_dic[word][0] = idx

    def make_batch(self):
        for sentence in self.corpus:
            splited_words = sentence.split()
            for idx, word in enumerate(splited_words):
                center_word = splited_words[idx]
                for i in range(-self.window, self.window + 1):
                    if i == 0:
                        continue
                    if (idx + i < 0) | (idx + i >= len(splited_words)):
                        continue
                    context_word = splited_words[idx + i]
                    if self.sg == 1:  # skip-gram
                        self.batch_x.append(self.word_dic[center_word][0])
                        self.batch_y.append([self.word_dic[context_word][0]])
                    else:  # cbow
                        self.batch_x.append(self.word_dic[context_word][0])
                        self.batch_y.append([self.word_dic[center_word][0]])

    def train(self):
        train_batch_x = np.array(self.batch_x)
        train_batch_y = np.array(self.batch_y)
        self.sess.run(tf.global_variables_initializer())
        for iter in range(self.iteration * round(train_batch_x.shape[0] / self.batch_size)):
            batch_idx = np.random.choice(train_batch_x.shape[0], self.batch_size, False)
            _, loss = self.sess.run([self.train_step, self.total_loss],
                                    feed_dict={self.x: train_batch_x[batch_idx], self.y: train_batch_y[batch_idx]})
            if iter % 300 == 0:
                print('iter : {}, loss : {}'.format(iter, loss))
                self.tr_loss_hist.append(loss)

    def negative_sampling(self, center):
        ## examples = center_word
        ## labels = 주변단어
        ## sample_idx = 주변아닌 다른 단어들
        labels = tf.cast(tf.expand_dims(center, 1), tf.int64)
        # labels = tf.reshape(tf.cast(batch_y[:3],tf.int64),[len(batch_y[:3]),1])
        negative_sample_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=self.ns_sample,
            range_max=7,
            unique=True,
            unigrams=[self.word_dic[word][1] for word in self.word_list]))  ##word 별 리스트 필요

        return negative_sample_ids

if __name__ == "__main__":
    a = word2vec(sentences)

    plt.plot(a.tr_loss_hist)
    for word in a.word_list:
        tmp = a.sess.run(a._embeddings[a.word_dic[word][0]])
        x, y = tmp[0], tmp[1]
        plt.scatter(x, y)
        plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

