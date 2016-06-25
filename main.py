# coding=utf-8
import numpy as np
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

# Deep Market with Deep Instrumental Variable
from model.DeepMarket import DeepConvolutionMarket
from model.DeepMarket import Convolution1dMaxPooingLayer

# Utility
import cPickle as pkl


def main(batch_size=300, n_epoch=10000):
    # We add noise in our data to prevent some trobbles......
    weather = pkl.load(open('./data/weather.pkl', 'rb'))
    price = pkl.load(open('./data/price.pkl', 'rb'))
    volumn = pkl.load(open('./data/volumn.pkl', 'rb'))
    weather = weather - weather.mean(0)
    price = (price - price.mean(0)) / price.std(0)
    volumn = (volumn - volumn.mean(0)) / volumn.std(0)

    weather_data = theano.shared(weather, borrow=True)
    weather_shape = (batch_size,) + \
        weather_data.get_value(borrow=True).shape[1:]
    n_batch = weather_shape[0] / batch_size

    price_data = theano.shared(
        price.astype(theano.config.floatX).reshape((price.shape[0], 1)),
        borrow=True)

    volumn_data = theano.shared(
        volumn.astype(theano.config.floatX).reshape((volumn.shape[0], 1)),
        borrow=True)

    price = T.matrix('price')
    volumn = T.matrix('volumn')
    weather = T.tensor4('weather')
    model = DeepConvolutionMarket(price, volumn, weather, weather_shape,
                                  layers=[Convolution1dMaxPooingLayer]*4,
                                  filter_shapes=[[3, 6, 1, 15], [2, 3, 1, 15],
                                                 [2, 2, 1, 5], [1, 2, 1, 4]])
    model.building_feedforward()
    lr = T.fscalar("lr")
    index = T.iscalar('idx')
    cost, updates = model.training_2SLS(lr)
    print "building the computional graph"
    train = theano.function([index, lr], cost, updates=updates,
                            givens={
                                price: price_data[index*batch_size:(index+1)*batch_size],
                                volumn: volumn_data[index*batch_size:(index+1)*batch_size],
                                weather: weather_data[index*batch_size:(index+1)*batch_size]}
                            ,mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    print "building done"
    record_cost = []
    lr0 = 1e-5
    for epoch in range(n_epoch):
        hist_cost = []
        for idx in range(n_batch):
            hist_cost.extend([train(idx, lr0/(epoch+1))])
        record_cost += [np.mean(hist_cost)]
        print "epoch {0}: training cost {1}".format(epoch, np.mean(hist_cost))
        if epoch % 100 == 0:
            pkl.dump({'model': model, 'cost': record_cost},
                     open('./model/trained_model/deepmarket_4layers.not.model',
                          'wb'))



if __name__ == "__main__":
    main()
