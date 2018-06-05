import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

 
class model(object):

    def __init__(self, nh, nc, ne, de, cs):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end


        #each speialization represent a matrices of possible inputs to the different layers

        #W_x represent the W term in each expression of the formula is a (de*cs)*nh
        #W_h represent the term that is propagated through iterations and is a nh*nh matrix
        #W_c represent the convolutional operator

        self.W_i = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (de*cs, nh)).astype(theano.config.floatX)) #the input vector for the current input, that contains the dimension of word embedding and window size
        self.U_i = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX)) #output of LSMT unit
        
        self.W_f = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (de*cs, nh)).astype(theano.config.floatX))
        self.U_f = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))
        
        self.W_c = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (de*cs, nh)).astype(theano.config.floatX))
        self.U_c = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))
        
        self.W_o = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (de*cs, nh)).astype(theano.config.floatX))
        self.U_o = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))
        
        self.V = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nc)).astype(theano.config.floatX))    #output layer with nh input and only number of classes as output
        
        #create bias vectors
        self.b_i = theano.shared(numpy.cast[theano.config.floatX](numpy.random.uniform(-0.5,.5,size = nh)))
        self.b_f = theano.shared(numpy.cast[theano.config.floatX](numpy.random.uniform(0, 1.,size = nh)))
        self.b_c = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.b_o = theano.shared(numpy.cast[theano.config.floatX](numpy.random.uniform(-0.5,.5,size = nh)))
        self.b = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))

        #after that 3 layers that are identical except for the last one that needs to map the class output are created and can be used for
        #whatever purpose, from this to GRU, by simply apply formulas

        self.c0 = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.h0 = T.tanh(self.c0)
        
        # bundle weights
        self.params = [self.emb, self.W_i, self.U_i, self.b_i, self.W_f, self.U_f, \
                       self.b_f, self.W_c, self.U_c, self.b_c, self.W_o, self.U_o, \
                       self.b_o, self.V, self.b, self.c0]
        self.names  = ['embeddings', 'W_i', 'U_i', 'b_i', 'W_f', 'U_f', 'b_f', \
                       'W_c', 'U_c', 'b_c', 'W_o', 'U_o', 'b_o', 'V', 'b', 'c0']
        
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y = T.iscalar('y') # label

        def recurrence(x_t, h_tm1, c_tm1):
            #implementation for  a convolutional LSTM
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_i) + T.dot(h_tm1, self.U_i) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_f) + T.dot(h_tm1, self.U_f) + self.b_f)
            c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_c) + T.dot(h_tm1, self.U_c) + self.b_c)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_o)+ T.dot(h_tm1, self.U_o) + self.b_o)
            h_t = o_t * T.tanh(c_t)
                
            s_t = T.nnet.softmax(T.dot(h_t, self.V) + self.b)
            
            return [h_t, c_t, s_t]

        [h, _, s], _ = theano.scan(fn=recurrence, sequences=x, outputs_info=[self.h0, self.c0, None], n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1,0,:]
        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.mean(T.log(p_y_given_x_lastword)[y])
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.train = theano.function(inputs=[idxs, y, lr], outputs=nll, updates=updates)

        self.normalize = theano.function(inputs=[], updates={self.emb: self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    #save model
    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

    #load model (edited)
    def load(self, folder):
        print('loading the params in folder...')
        updates = OrderedDict((param, theano.shared(numpy.load(os.path.join(folder, name + '.npy')).astype(theano.config.floatX))) for param, name in zip( self.params , self.names))
        loadParam = theano.function(inputs = [],updates = updates)
        loadParam()