import cPickle
import gzip
import os
    
import numpy as np
import theano as th
from theano import tensor as T
from numpy import random as rng
import scipy
# np.set_printoptions(threshold=np.inf)

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

class AutoEncoder(object):
    def __init__(self, X, hidden_size, activation_function,output_function):
        #X is the data, an m x n numpy matrix
        #where rows correspond to datapoints
        #and columns correspond to features.
        assert type(X) is np.ndarray
        assert len(X.shape)==2
        self.X=X
        self.X=th.shared(name='X', value=np.asarray(self.X, 
                         dtype=th.config.floatX),borrow=True)
        #The config.floatX and borrow=True stuff is to get this to run
        #fast on the gpu. I recommend just doing this without thinking about
        #it until you understand the code as a whole, then learning more
        #about gpus and theano.
        self.n = X.shape[1]
        self.m = X.shape[0]
        #Hidden_size is the number of neurons in the hidden layer, an int.
        assert type(hidden_size) is int
        assert hidden_size > 0
        self.hidden_size=hidden_size
        initial_W = np.asarray(rng.uniform(
                 low=-4 * np.sqrt(6. / (self.hidden_size + self.n)),
                 high=4 * np.sqrt(6. / (self.hidden_size + self.n)),
                 size=(self.n, self.hidden_size)), dtype=th.config.floatX)
        self.W = th.shared(value=initial_W, name='W', borrow=True)
        self.b1 = th.shared(name='b1', value=np.zeros(shape=(self.hidden_size,),
                            dtype=th.config.floatX),borrow=True)
        self.b2 = th.shared(name='b2', value=np.zeros(shape=(self.n,),
                            dtype=th.config.floatX),borrow=True)
        self.activation_function=activation_function
        self.output_function=output_function
                    
    def trainAutoencoder(self, n_epochs=100, mini_batch_size=1, learning_rate=0.1):
        index = T.lscalar()
        x=T.matrix('x')
        params = [self.W, self.b1, self.b2]
        hidden = self.activation_function(T.dot(x, self.W)+self.b1)
        output = T.dot(hidden,T.transpose(self.W))+self.b2
        output = self.output_function(output)
        
        #Use cross-entropy loss.
        L = -T.sum(x*T.log(output) + (1-x)*T.log(1-output), axis=1)
        cost=L.mean()       
        updates=[]
        
        #Return gradient with respect to W, b1, b2.
        gparams = T.grad(cost,params)
        
        #Create a list of 2 tuples for updates.
        for param, gparam in zip(params, gparams):
            updates.append((param, param-learning_rate*gparam))
        
        #Train given a mini-batch of the data.
        train = th.function(inputs=[index], outputs=[cost], updates=updates,
                            givens={x:self.X[index:index+mini_batch_size,:]})
                            

        import time
        start_time = time.clock()
        for epoch in xrange(n_epochs):
            # print "Epoch:",epoch
            count = 0
            total = 0
            for row in xrange(0,self.m, mini_batch_size):
                count = count + 1
                val = train(row)    
                total = total +val[0]
            print total / count
        end_time = time.clock()
        print "Average time per epoch=", (end_time-start_time)/n_epochs
                   
    def get_hidden(self,data):
        x=T.dmatrix('x')
        # print type(x)
        hidden = self.activation_function(T.dot(x,self.W)+self.b1)
        transformed_data = th.function(inputs=[x], outputs=[hidden])
        return transformed_data(data)

    def get_output(self, data):
        print "in getoutput function"
        x =  T.dmatrix('x')
        h = self.get_hidden(data)
        # x = th.shared(h)
        print type(h)
        print h
        print type(x)
        print x
        output = T.dot(h,T.transpose(self.W))+self.b2
        output_data= th.function(inputs = [x], outputs = [output], on_unused_inputs = 'warn')
        return output_data(h)
    
    def get_weights(self):
        return [self.W.get_value(), self.b1.get_value(), self.b2.get_value()]
        

def load_data(dataset):
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
   
   '''
    train_set, valid_set, test_set format: tuple(input, target)
    input is an numpy.ndarray of 2 dimensions (a matrix)
    witch row's correspond to an example. target is a
    numpy.ndarray of 1 dimensions (vector)) that have the same length as
    the number of rows in the input. It should give the target
    target to the example with the same index in the input.
    '''

    return (train_set, valid_set, test_set)
        

path='/home/sensors/Desktop/neural/mnist.pkl.gz'

data=load_data(path)
       
def plot_first_k_numbers(X,k):
    from matplotlib import mpl,pyplot
    m=X.shape[0]
    k=min(m,k)
    j = int(round(k / 10.0))
    
    fig, ax = pyplot.subplots(j,10)
   
    for i in range(k):

        w=X[i,:]

        
        w=w.reshape(28,28)
        ax[i/10, i%10].imshow(w,cmap=pyplot.cm.gist_yarg,
                      interpolation='nearest', aspect='equal')
        ax[i/10, i%10].axis('off')

    
    pyplot.tick_params(\
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    pyplot.tick_params(\
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off', 
        right='off',    # ticks along the top edge are off
        labelleft='off')
    
    fig.show()
    pyplot.savefig('plot.png')
    
def toimg(X, name):
    Y = np.reshape(X, (-1,28))
    scipy.misc.imsave(name,Y)
        
def m_test(data):
    print "--------X---------50000*784--- num_inputs*img_size--- Images inputs"
    X=data[0][0]
    print type(X)
    print X.size
    print X.shape
    print len(X.shape)
    toimg(X[11], 'input_img.png')   #Cnvert an array to Image, data[0][0][0]

    activation_function = T.nnet.sigmoid
    output_function=activation_function
    hidden_size = [10, 30, 60,100,120]
    for size in hidden_size:
        print "Training for hidden_size ", str(size)
        A = AutoEncoder(X, size, activation_function, output_function)
        A.trainAutoencoder(30,100)
        W=np.transpose(A.get_weights()[0])
        b1=np.transpose(A.get_weights()[1])
        b2=np.transpose(A.get_weights()[2])
        np.savez(str(size)+'_weights.npz', W=W, b1=b1, b2=b2)

m_test(data)