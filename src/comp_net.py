import numpy as np
import tensorflow as tf

#from model import Model
from q2_initialization import xavier_weight_init


class ComparisonNet():

    def add_placeholders(self):
        """
        Adds the place holders.  P is the number of features extracted
        and n is the length of each feature vector
        """
        self.input_placeholders = tf.placeholder(tf.float32, shape = (None, self.P * self.n))
        self.labels = tf.placeholder(tf.float32, shape =(None, 2))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, input_batch, labels_batch=None, dropout=1):
        """
        Creates the feed dict.
        """
        feed_dict = {
                self.input_placeholders: input_batch,
                self.dropout_placeholder: dropout
                }
        if labels_batch is not None:
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

        W3 = tf.Variable(xavier_initializer((self.h2,2)))

        layer1 = tf.nn.relu(tf.matmul(self.input_placeholders, W1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
        self.preds = tf.matmul(layer2, W3)
        return self.preds
    
    def add_loss(self):
        loss_vec = tf.nn.softmax_cross_entropy_with_logits(self.preds, self.labels)
        self.loss = tf.reduce_mean(loss_vec)
        return self.loss

    def add_optimizers(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)
        return self.train_op

    def train_batch(self, inputs, labels, dropout = 0.1):
        feed = self.create_feed_dict(inputs, labels, dropout)
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, train_examples, dev_set):
        train_batch = train_examples[0]
        labels = train_examples[1]
        avg_loss = 0
        for i in range(len(train_batch)):
            avg_loss += self.train_batch(train_batch[i], labels[i])
        print "Train loss = " + str(float(avg_loss) / len(train_batch))
        dev_right = 0
        dev_batch = dev_set[0]
        labels = dev_set[1]
        for i in range(len(dev_batch)):
            label, train_ex = labels[i], dev_batch[i]
            feed = self.create_feed_dict(train_ex)
            preds = self.sess.run([self.preds], feed_dict=feed)
            preds = preds[0][0]
            #print preds
            if label[0, 0] > label[0,1]:
                dev_right += 1 if preds[0] > preds[1] else 0
            else:
                dev_right += 1 if preds[1] > preds[0] else 0
        return float(dev_right) / len(dev_batch)

    def fit(self, train_examples, dev_set, filename = 'BEST_MODEL'):
        best_accuracy = 0
        for i in range(self.n_epochs):
            print "Running Epoch: " + str(i)
            dev_accuracy = self.run_epoch(train_examples, dev_set)
            print "DEV ACCURACY: " + str(dev_accuracy)
            if dev_accuracy > best_accuracy:
                print "New best accuracy of " + str(dev_accuracy) + "!!"
                best_accuracy = dev_accuracy
                self.saver.save(self.sess, filename)
            
    def __init__(self, P, n, h1, h2, lr= 0.01, epochs=20):
        # Set initial values
        self.P = P
        self.n = n
        self.h1 = h1
        self.h2 = h2
        self.lr = lr
        self.sess = tf.Session()
        self.n_epochs = epochs
        
        # Build Framework
        self.add_placeholders()
        self.pred = self.predict()
        self.loss = self.add_loss()
        self.train_op = self.add_optimizers()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
        

def main(input_batch, labels=None, P=30, n=1024, h1=250, h2=125):
    comp_model = ComparisonNet(P, n, h1, h2)
    # Need to parse data now... 
    # So havent teted this architecture yet, seems
    # alright but definitely going to have some bugs lol

if __name__ == '__main__':
    main()
