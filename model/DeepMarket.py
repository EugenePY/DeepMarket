import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
import numpy as np


class BN(object):
    """
    Batch normalization
    """
    pass


class ResidualLayer(object):
    def __init__(self, input, inital_x, input_shape, filter_shape,
                 poolingsize=(2, 2), W=None, activate=T.nnet.relu,
                 identity=False, bias=None):
        """
        parm: W (n_feature_map, n_input_feature, n_filter_width,
                 n_filter_length)
        """

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolingsize))

        if input is None:
            self.input = T.tensor4("input")
        else:
            self.input = input

        if W is None:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                np.random.uniform(low=-W_bound,
                                  high=W_bound,
                                  size=filter_shape).astype(
                                      theano.config.floatX), borrow=True)

        if bias is None:
            self.bias = theano.shared(
                np.zeros(filter_shape[0]).astype(theano.config.floatX),
                borrow=True)

        if identity:
            self.project = T.eye(input_shape[1], dtype=theano.config.floatX)
        else:
            self.project_shape = (filter_shape[0], input_shape[1]) + \
                (1, 1)
            self.project = theano.shared(
                np.random.normal(
                    size=self.project_shape).astype(theano.config.floatX),
                borrow=True)

        self.params = [self.W, self.bias]
        self.filter_shape = filter_shape
        self.image_shape = input_shape

        self.activate = activate
        pre_activate = conv2d(input=self.input,
                              filters=self.W,
                              filter_shape=self.filter_shape,
                              input_shape=self.image_shape, border_mode='half')
        project = conv2d(input=inital_x,
                         filters=self.project,
                         filter_shape=self.project_shape,
                         border_mode='valid')

        self.output = self.activate(pre_activate +
                                    self.bias.dimshuffle("x", 0, "x", "x")) +\
            project
        self.backward = None


class Convolution2dMaxPooingLayer(object):
    def __init__(self, input, image_shape, filter_shape, poolingsize=(2, 2),
                 W=None, bias=None, activate=T.nnet.relu):
        # the input is a minibatch of gray image
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolingsize = poolingsize
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolingsize))

        if input is None:
            self.input = T.tensor4("input")
        else:
            self.input = input

        if W is None:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                np.random.uniform(low=-W_bound,
                                  high=W_bound,
                                  size=filter_shape).astype(
                                      theano.config.floatX), borrow=True)

        if bias is None:
            self.bias = theano.shared(
                np.zeros(filter_shape[0]).astype(theano.config.floatX),
                borrow=True)

        self.activate = activate

        conv_2d = conv2d(input=self.input,
                         filters=self.W,
                         filter_shape=self.filter_shape,
                         input_shape=self.image_shape, border_mode='half')
        # using the default stride

        pooled_out = pool_2d(conv_2d, poolingsize, ignore_border=True)
        # using the default stride
        self.output = self.activate(pooled_out +
                                    self.bias.dimshuffle("x", 0, "x", "x"))
        self.params = [self.W, self.bias]


class Convolution1dMaxPooingLayer(object):
    def __init__(self, input, input_dim, filter_shape, poolingsize=(1, 4),
                 W=None, bias=None, activate=T.nnet.relu):
        # the input is a minibatch of gray image
        """
        input_dim: (n_feature, 1, n_dimension)
        filter_shape: (n_feture_out, n_feature_in, 1, n_dim_filter)
        """
        assert input_dim[1] == filter_shape[1]
        assert filter_shape[2] == 1
        assert poolingsize[0] == 1

        self.filter_shape = filter_shape
        self.image_shape = input_dim
        self.poolingsize = poolingsize

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolingsize))

        if input is None:
            self.input = T.tensor4("input")
        else:
            self.input = input

        if W is None:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                np.random.uniform(low=-W_bound,
                                  high=W_bound,
                                  size=filter_shape).astype(
                                      theano.config.floatX), borrow=True,
            name='W')

        if bias is None:
            self.bias = theano.shared(
                np.zeros(filter_shape[0]).astype(theano.config.floatX),
                borrow=True, name='bias')

        self.activate = activate

        conv_2d = conv2d(input=self.input,
                         filters=self.W,
                         filter_shape=self.filter_shape,
                         input_shape=self.image_shape, border_mode='half')
        # using the default stride

        pooled_out = pool_2d(conv_2d, poolingsize, ignore_border=True)
        # using the default stride
        self.output = self.activate(pooled_out +
                                    self.bias.dimshuffle("x", 0, "x", "x"))
        self.params = [self.W, self.bias]


class DeepConvolutionMarket(object):
    def __init__(self, price, volumn, weather,
                 weather_input_shape,
                 layers=[Convolution1dMaxPooingLayer,
                         Convolution1dMaxPooingLayer],
                 filter_shapes=[[3, 7, 1, 15], [1, 3, 1, 10]],
                 demand_cof=None, demand_bias=None,
                 supply_cof=None, supply_bias=None,
                 supply_shock_coef=None):
        """
        The class is trying to solve a system of linear eq which describing
        Demand and Supply

        param: image_IV: Image as an Instrumental Variable.
        market_input (maket_volumn, market_price)
        weather_input_shape: (n_feature, 1, n_dim)
        """
        assert len(layers) == len(filter_shapes)
        assert weather_input_shape[1] == filter_shapes[0][1]

        for i in range(len(filter_shapes)-1):
            assert filter_shapes[i][0] == filter_shapes[i+1][1]

        self.price = price
        self.volumn = volumn
        self.weather_input = weather
        self.weather_shape = weather_input_shape
        self.filter_shapes = filter_shapes

        if demand_cof is None:
            demand_cof = theano.shared(
                -np.random.normal(size=1).astype(theano.config.floatX)**2)

        if demand_bias is None:
            demand_bias = theano.shared(
                np.random.normal(size=1).astype(theano.config.floatX))

        if supply_cof is None:
            supply_cof = theano.shared(
                np.random.normal(size=1).astype(theano.config.floatX)**2)

        if supply_bias is None:
            supply_bias = theano.shared(
                np.random.normal(size=1).astype(theano.config.floatX))

        # HELPER function
        def last_layer_ouputshape(input_shape, n_layers, poolingsize,
                                  last_feature_map):
            """
            input_shape:tuple (n_feature, 1, n_dim)
            n_layer: int
            poolingsize: (1, n_pooling_size) # only for 1 d convolution
            """
            size = np.ceil(
                float(last_feature_map*input_shape[-1]) /
                (poolingsize[-1]**n_layers)).astype(int)
            if size == 0:
                return 1
            else:
                return size

        if supply_shock_coef is None:
            supply_shock_coef = theano.shared(
                np.random.normal(scale=1e-6,
                    size=(last_layer_ouputshape(self.weather_shape, len(layers),
                                                (1, 4),
                                                self.filter_shapes[-1][0]), 1)
                ).astype(theano.config.floatX))

        self.demand_cof = demand_cof
        self.demand_bias = demand_bias
        self.supply_cof = supply_cof
        self.supply_bias = supply_bias
        self.supply_shock_coef = supply_shock_coef
        self.layers = layers
        self.layers_build = []
        self.params = [self.demand_cof, self.demand_bias, self.supply_cof,
                       self.supply_bias, self.supply_shock_coef]

    def demand(self, price):
        # just for clean expression
        return self.demand_cof * price + self.demand_bias

    def supply(self, volumn, supply_shock):
        # just for clean expression
        return self.supply_cof * volumn + self.supply_bias + \
            T.dot(supply_shock, self.supply_shock_coef)

    def building_feedforward(self):
        def caloutput_shape(layer_m0):
            max_pool_2d_shape = (1, 1) + layer_m0.poolingsize
            output_kernel = layer_m0.filter_shape[0]
            out = [int(np.ceil(i/j)) for i, j in zip(layer_m0.image_shape,
                                                     max_pool_2d_shape)]
            out[1] = output_kernel
            for i in range(len(out)):
                if out[i] == 0:
                    out[i] = 1
            return out

        print "Stacking the Network"
        for i, layer in enumerate(self.layers):
            if i == 0:
                input = self.weather_input
                input_shape = self.weather_shape
            else:
                input = self.layers_build[-1].output
                input_shape = caloutput_shape(self.layers_build[-1])

            if layer == Convolution1dMaxPooingLayer:
                print "Stack ConvolutionMaxPoolingLayer"
                self.layers_build += [layer(input, input_shape,
                                            self.filter_shapes[i])]
            elif layer == ResidualLayer:
                print "Stack Residual Layer"
                self.layers_build += [layer(input, input, input_shape,
                                            self.filter_shapes[i])]

            self.params.extend(self.layers_build[-1].params)
        print "Done Stacking"
        return self

    def training(self, lr):
        # Using Market syetem to estimate
        supply_shock = self.layers_build[-1].output.flatten(2)

        updates = theano.OrderedUpdates()

        cost = T.mean(T.sqr(self.volumn - self.demand(self.price))) + \
            T.mean(T.sqr(self.price - self.supply(self.volumn, supply_shock)))

        self.gparams = T.grad(cost, self.params)

        for param, gparam in zip(self.params, self.gparams):
            updates[param] = param - lr * gparam

        return cost, updates

    def training_2SLS(self, lr):
        # using two stage least square
        # First stage in order to get the consistant estimator
        # about the parameters (Get ride of the endoenity)
        # The percedure is similar with the systematic equation other than
        # we feed the \hat{p}_{s} into the demand funciton
        # In practice the 2SLS has a better converge rate?
        supply_shock = self.layers_build[-1].output.flatten(2)
        updates = theano.OrderedUpdates()

        p_hat = self.supply(self.volumn, supply_shock)
        cost = T.mean(T.sqr(self.volumn - self.demand(p_hat))) + \
            T.mean(T.sqr(self.price - self.supply(self.volumn, supply_shock)))

        self.gparams = T.grad(cost, self.params, consider_constant=[p_hat])

        for param, gparam in zip(self.params, self.gparams):
            updates[param] = param - lr * gparam

        return cost, updates

    def compiling(self):
        # Do not need to pre-train
        pass


class DeepImageIVMarket(object):
    def __init__(self, price, volumn, image_IV, image_shape,
                 layers=[Convolution2dMaxPooingLayer, ResidualLayer],
                 filter_shapes=[[3, 7, 1, 15], [1, 3, 15, 15]],
                 demand_cof=None, demand_bias=None,
                 supply_cof=None, supply_bias=None,
                 supply_shock_coef=None):

        """
        The class is trying to solve a system of linear eq which describing
        Demand and Supply
        param: image_IV: Image as an Instrumental Variable.
        """
        assert len(layers) == len(filter_shapes)
        assert image_shape[1] == filter_shapes[0][1]

        for i in range(len(filter_shapes)-1):
            assert filter_shapes[i][0] == filter_shapes[i+1][1]

        self.price = price
        self.volumn = volumn
        self.image_IV = image_IV
        self.filter_shapes = filter_shapes

        if demand_cof is None:
            demand_cof = theano.shared(
                np.random.normal(size=1).astype(theano.config.floatX))

        if demand_bias is None:
            demand_bias = theano.shared(
                np.random.normal(size=1).astype(theano.config.floatX))

        if supply_cof is None:
            supply_cof = theano.shared(
                np.random.normal(size=1).astype(theano.config.floatX))

        if supply_bias is None:
            supply_bias = theano.shared(
                np.random.normal(size=1).astype(theano.config.floatX))

        if supply_shock_coef is None:
            supply_shock_coef = theano.shared(
                np.random.normal(size=(2500, 1)).astype(
                    theano.config.floatX))

        self.demand_cof = demand_cof
        self.demand_bias = demand_bias
        self.supply_cof = supply_cof
        self.supply_bias = supply_bias
        self.image_shape = image_shape
        self.supply_shock_coef = supply_shock_coef
        self.layers = layers
        self.layers_build = []
        self.params = [self.demand_cof, self.demand_bias, self.supply_cof,
                       self.supply_bias, self.supply_shock_coef]

    def demand(self, volumn, supply_shock):
        # just for clean expression
        return self.demand_cof * volumn + self.demand_bias + \
            T.dot(supply_shock, self.supply_shock_coef)

    def supply(self, price):
        # just for clean expression
        return self.supply_cof * price + self.supply_bias

    def building_feedforward(self):
        def caloutput_shape(layer_m0):
            max_pool_2d_shape = (1, 1) + layer_m0.poolingsize
            output_kernel = layer_m0.filter_shape[0]

            out = [int(np.ceil(i/j)) for i, j in zip(layer_m0.image_shape,
                                                     max_pool_2d_shape)]
            out[1] = output_kernel
            return out

        print "Stacking the Network"
        for i, layer in enumerate(self.layers):
            if i == 0:
                input = self.image_IV
                input_shape = self.image_shape
            else:
                input = self.layers_build[-1].output
                input_shape = caloutput_shape(self.layers_build[-1])

            if layer == Convolution2dMaxPooingLayer:
                print "Stack ConvolutionMaxPoolingLayer"
                self.layers_build += [layer(input, input_shape,
                                            self.filter_shapes[i])]
            elif layer == ResidualLayer:
                print "Stack Residual Layer"
                self.layers_build += [layer(input, input, input_shape,
                                            self.filter_shapes[i])]
        print "Done Stacking"
        return self

    def training(self, lr):
        supply_shock = self.layers_build[-1].output.flatten(2)

        updates = theano.OrderedUpdates()
        cost = T.mean(T.sqr(self.volumn -
                            self.demand(self.price, supply_shock))) +\
            T.mean(T.sqr(self.price - self.supply(self.volumn)))

        self.gparams = T.grad(cost, self.params)

        for param, gparam in zip(self.params, self.gparams):
            updates[param] = param - lr * gparam

        return cost, updates

    def compiling(self):
        # Do not need to pre-train
        pass


def test_model():
    price = theano.shared(
        np.random.normal(size=(100, 1)).astype(theano.config.floatX),
        borrow=True)
    volumn = theano.shared(
        np.random.normal(size=(100, 1)).astype(theano.config.floatX),
        borrow=True)
    images_IV = theano.shared(
        np.random.normal(size=(100, 40, 100, 100)).astype(theano.config.floatX),
        borrow=True)

    images_shape = images_IV.get_value(borrow=True).shape
    model = DeepImageIVMarket(price=price, volumn=volumn, image_IV=images_IV,
                              image_shape=images_shape)
    model.building_feedforward()
    lr = T.fscalar("lr")
    cost, updates = model.training(lr)
    print "Building the computaional graph"
    train = theano.function([lr], cost, updates=updates)
    print "Building Done"
    train(0.5)


def test_model1():
    price = theano.shared(
        np.random.normal(size=(100, 1)).astype(theano.config.floatX),
        borrow=True)
    volumn = theano.shared(
        np.random.normal(size=(100, 1)).astype(theano.config.floatX),
        borrow=True)
    weather = theano.shared(
        np.random.normal(size=(100, 7, 1, 100)).astype(theano.config.floatX),
        borrow=True)

    weather_shape = weather.get_value(borrow=True).shape
    model = DeepConvolutionMarket(price, volumn, weather, weather_shape)
    model.building_feedforward()
    lr = T.fscalar("lr")
    cost, updates = model.training_2SLS(lr)
    print "building the computional graph"
    train = theano.function([lr], cost, updates=updates)
    print "building done"
    for i in range(1000):
        print train(0.01)


if __name__ == "__main__":
    test_model1()
