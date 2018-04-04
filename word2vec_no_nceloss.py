import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

sentences = ["나 고양이 노의미 좋다",
             "나 개 노의미 싫다",
             "너 고양이 노의미 싫다",
             "너 개 노의미 좋다"]

class word2vec:
    def __init__(self, corpus=None, embedding_size=2, window=3, iteration=2000, sg=1):
        self.corpus = corpus
        self.embedding_size = embedding_size
        self.window = window
        self.iteration = iteration
        self.sg = sg

        self.word_list = []
        self.num_classes = self.word_count()
        self.word_dic = {w: i for i, w in enumerate(self.word_list)}
        self.batch_x = []
        self.batch_y = []
        self.make_batch()
        self.sess = tf.Session()

        self.x = tf.placeholder(dtype=tf.int32, shape=[None], name="input")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="output")

        self._embeddings = tf.get_variable(name='embeddings_input', shape=[self.num_classes, self.embedding_size],
                                           dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        self._selected_embed = tf.nn.embedding_lookup(params=self._embeddings, ids=self.x)

        self.nce_weights = tf.Variable(tf.random_uniform([self.embedding_size, self.num_classes], -1.0, 1.0))
        self.nce_biases = tf.Variable(tf.zeros([self.num_classes]))

        self.pred = tf.add(tf.matmul(self._selected_embed, self.nce_weights), self.nce_biases)
        self.nce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y, depth=self.num_classes),logits=self.pred))

        self.batch_size = 10
        self.tr_loss_hist = []
        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.nce_loss)
        self.train()

    def word_count(self):
        for sentence in self.corpus:
            for word in sentence.split():
                self.word_list.append(word)
                self.word_list = list(set(self.word_list))
        return len(self.word_list)

    def make_batch(self):
        for sentence in self.corpus:
            tmp = sentence.split()
            for idx, word in enumerate(tmp):
                center_word = tmp[idx]
                for i in range(-self.window, self.window + 1):
                    if i == 0:
                        continue
                    if (idx + i < 0) | (idx + i >= len(tmp)):
                        continue
                    context_word = tmp[idx + i]
                    if self.sg == 1:  # skip-gram
                        self.batch_x.append(self.word_dic[center_word])
                        self.batch_y.append([self.word_dic[context_word]])
                    else:  # cbow
                        self.batch_x.append(self.word_dic[context_word])
                        self.batch_y.append([self.word_dic[center_word]])

    def train(self):
        train_batch_x = np.array(self.batch_x)
        train_batch_y = np.array(self.batch_y)
        self.sess.run(tf.global_variables_initializer())
        for iter in range(self.iteration * round(train_batch_x.shape[0] / self.batch_size)):
            batch_idx = np.random.choice(train_batch_x.shape[0], self.batch_size, False)
            _, loss = self.sess.run([self.train_step, self.nce_loss],
                                    feed_dict={self.x: train_batch_x[batch_idx], self.y: train_batch_y[batch_idx]})
            if iter % 300 == 0:
                print('iter : {}, loss : {}'.format(iter, loss))
                self.tr_loss_hist.append(loss)
    
    def negative_sampling(self):
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=self.batch_y,
            num_true=1,
            num_sampled=2,
            range_max=7,
            unique=True,
            unigrams=[2, 2, 4, 2, 2, 2, 2]))##word 별 리스트 필요
        
if __name__ == "__main__":
    a = word2vec(sentences)

    plt.plot(a.tr_loss_hist)
    for word in a.word_list:
        tmp = a.sess.run(a._embeddings[a.word_dic[word]])
        x, y = tmp[0], tmp[1]
        plt.scatter(x, y)
        plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
