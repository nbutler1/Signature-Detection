import numpy as np
import tensorflow as tf

from model import Model
from q2_initialization import xavier_weight_init


class ComparisonNet():
    def init(self, P, n, h1, h2, lr= 0.05, epochs=20):
        self.P = P
        self.n = n
        self.h1 = h1
        self.h2 = h2
        self.lr = lr
        self.sess = tf.Session()

        self.saver = tf.train.Saver()
        self.n_epochs = epochs

    def add_placeholders(self):
        """
        Adds the place holders.  P is the number of features extracted
        and n is the length of each feature vector
        """
        self.input_placeholders = tf.placeholder(tf.float32, shape = (None, 2, self.P * self.n))
        self.labels = tf.placeholder(tf.bool, shape =(None,self.P))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, input_batch, labels_batch=None, dropout=1):
        """
        Creates the feed dict.
        """
        feed_dict = {
                self.input_placeholders: input_batch,
                self.dropout_placeholder: dropout
                }
        if laebls_batch is not None:
            feed_dict[self.labels] = labels_batch
        return feed_dict

    def predict(self):
        xavier_initializer = xavier_weight_init()
        
        #TODO: create architecture
        # First layer
        W1 = tf.Variable(xavier_initializer((self.P*self.n, self.h1)))
        b1 = tf.Variable(tf.zeros([self.h1]))

        W2 = tf.Variable(xavier_initializer((self.h1, self.h2)))
        b2 = tf.Variable(tf.zeros([self.h2]))

        W3 = tf.Variable(xavier_initializer((self.h2,)))

        layer1 = tf.nn.relu(tf.matmul(self.input_placeholders, W1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
        self.preds = tf.matmul(layer2, W3)
        return self.preds
    
    def add_loss(self):
        loss_vec = tf.nn.softmax_cross_entropy_with_logits(self.preds, self.labels_placeholder)
        self.loss = tf.reduce_mean(loss_vec)
        return self.loss

    def add_optimizers(self, pred):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)
        return self.train_op

    def train_batch(self, inputs, labels, dropout = 0.1):
        feed = self.create_feed_dict(inputs, labels, dropout)
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict = feed])
        return loss

    def run_epoch(self, train_examples, dev_set):
        for i in range(len(train_examples)):
            labels, train_ex = train_examples[i]
            loss = self.train_batch(train_ex, labels)
        print "Evaluating on Dev Set"
        dev_right = 0
        for i in range(len(dev_set)):
            label, train_ex = dev_set[i]
            feed = self.create_feed_dict(train_ex)
            preds = self.sess.run([self.preds], feed_dict = feed)
            if label[0] > label[1]:
                dev_right += 1 if preds[0] > preds[1]
            else:
                dev_right += 1 if preds[1] > preds[0]
        return dev_right

    def fit(self, train_examples, dev_set, filename = 'BEST_MODEL'):
        best_accuracy = 0
        for i in range(self.n_epochs):
            print "Running Epoch: " + str(i)
            dev_accuracy = self.run_epoch()
            if dev_accuracy > best_accuracy:
                print "New best accuracy of " + str(dev_accuracy) + "!!"
                best_accuracy = dev_accuracy
                self.saver.save(self.sess, filename)
            
def main(input_batch, labels=None, P=30, n=1024, h1=250, h2=125):
    comp_model = ComparisonNet(P, n, h1, h2)
    # Need to parse data now... 
    # So havent teted this architecture yet, seems
    # alright but definitely going to have some bugs lol

if __name__ == '__main__':
    main()
