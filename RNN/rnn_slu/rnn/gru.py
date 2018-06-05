import theano
import numpy as numpy
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
        self.U_z = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh,nh)).astype(theano.config.floatX))
        self.W_z = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de*cs,nh)).astype(theano.config.floatX))
        self.U_r = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh,nh)).astype(theano.config.floatX))
        self.W_r = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de*cs,nh)).astype(theano.config.floatX))
        self.U_h = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh,nh)).astype(theano.config.floatX))
        self.W_h = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de*cs,nh)).astype(theano.config.floatX))
        self.V = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.b = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.b_z = theano.shared(numpy.cast[theano.config.floatX](numpy.random.uniform(0,1.,size = nh)))
        self.b_r = theano.shared(numpy.cast[theano.config.floatX](numpy.random.uniform(0,1.,size = nh)))
        self.b_h = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

        # bundle
        self.params = [self.emb, self.U_z, self.W_z, self.U_r, self.W_r, self.U_h, self.W_h, self.V, self.b, self.b_z, self.b_r, self.b_h, self.h0]
        self.names  = ["emb", "uz", "wz", "ur", "wr", "uh", "wh", "V", "b", "bz", "br", "bh", "h0"]
        
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs)) #reshape into a 1*D dimensions for each element
        y = T.iscalar('y') # label

        def recurrence(x_t, h_tm1):
            #create the GRU implementation
            z_t = T.nnet.hard_sigmoid(T.dot(x_t, self.W_z) + T.dot(h_tm1, self.U_z) + self.b_z)
            r_t = T.nnet.hard_sigmoid(T.dot(x_t, self.W_r) + T.dot(h_tm1, self.U_r) + self.b_r)

            #define the output vector
            c_t = T.tanh(T.dot(x_t, self.W_h) + T.dot(h_tm1*r_t, self.U_h) + self.b_h)
            h_t = (T.ones_like(z_t)-z_t)*c_t + z_t*h_tm1

            #to next function pass h_t and o_t
            o_t = T.nnet.softmax(T.dot(h_t, self.V) + self.b)
            return [h_t, o_t]

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
            n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1,0,:]
        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.log(p_y_given_x_lastword)[y]
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.pcost = theano.function( inputs  = [idxs, y], outputs = nll)

        self.train = theano.function( inputs  = [idxs, y, lr],
                                      outputs = nll,
                                      updates = updates )

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

    def load(self, folder):   
        for param, name in zip(self.params, self.names):
            param.set_value(numpy.load(os.path.join(folder, name + '.npy')))