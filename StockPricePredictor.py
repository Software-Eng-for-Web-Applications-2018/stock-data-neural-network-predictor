#
# CREATED BY JOHN GRUN
#   APRIL 21 2018 
#
# TESTED BY JOHN GRUN
#
#MODIFIED BY JOHN GRUN 
#

#Based upon examples from the tensorflow cookbook

#http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
import os
import sys
import argparse;
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from DatabaseORM import session, StockPriceMinute
from DataArrayTools import ShitftAmount,TrimArray
from SupportPredictorFunctions import GetStockDataList, SaveModelAndQuit


# def RestoreModel(ModelName):
#     new_saver = tf.train.import_meta_graph(ModelName)
#     new_saver.restore(sess, tf.train.latest_checkpoint('./')) 
#     graph = tf.get_default_graph()
#     w1 = graph.get_tensor_by_name("w1:0")
#     w2 = graph.get_tensor_by_name("w2:0")
#     feed_dict ={w1:13.0,w2:17.0}

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '', 'Working directory.')
tf.app.flags.DEFINE_string('sym', '', 'Stock Symbol')  
tf.app.flags.DEFINE_integer('shiftamount', 1, 'Amount of time we wish to attept to predict into the future')
tf.app.flags.DEFINE_integer('DEBUG', 0, 'Enable the debugging output')  
FLAGS = tf.app.flags.FLAGS

def TrainNeuralNetwork(session,DatabaseTables,stocksym,RelativeTimeShift,TrainThreshold,DEBUG):


    #Xdata = GetStockDataList(session,StockPriceMinute,'AMD');
    Xdata = GetStockDataList(session,DatabaseTables,stocksym);

    #print(Xdata)

    # Shitf the training dat by X timeuits into the "future"
    Ydata = ShitftAmount(Xdata,RelativeTimeShift)

    #Make the data arrays the same length 
    Xdata = TrimArray(Xdata,(-1*RelativeTimeShift))

    LengthOfDataSet = len(Xdata)

    # Training and test data
    train_start = 0
    train_end = int(np.floor(0.8*LengthOfDataSet))
    test_start = train_end + 1
    test_end = LengthOfDataSet

    Xdata_train = Xdata[np.arange(train_start, train_end), :]
    Ydata_train = Ydata[np.arange(train_start, train_end), :]

    Xdata_test = Xdata[np.arange(test_start, test_end), :]
    Ydata_test = Ydata[np.arange(test_start, test_end), :]

    # Scale data
    Xscaler = MinMaxScaler(feature_range=(-1, 1))
    Xscaler.fit(Xdata_train)
    Xdata_train = Xscaler.transform(Xdata_train)
    Xdata_test = Xscaler.transform(Xdata_test)

    Yscaler = MinMaxScaler(feature_range=(-1, 1))
    Yscaler.fit(Ydata_train)
    Ydata_train = Yscaler.transform(Ydata_train)
    Ydata_test = Yscaler.transform(Ydata_test)


    # # Build X and y
    X_train = Xdata_train;
    y_train = Ydata_train[:, 2]

    if(DEBUG == 1):
        print("X data\n")
        print(X_train)
        print("\n")
        print("y_train\n")
        print(y_train[0])

    # # print(y_train)
    X_test = Xdata_test
    y_test = Ydata_test[:, 2]

    # Number of stocks in training data
    NumElementsPerRow = X_train.shape[1]
    NumElementsOut = y_train.shape[0]

    if(DEBUG == 1):
        print("Length of y_train "+ str( len( y_train) ))

        print("NumElementsPerRow " + str(NumElementsPerRow) + " NumElementsOut " + str(NumElementsOut))
    # # Neuron config
    NumNeurons1 = 1024
    NumNeurons2 = 512
    NumNeurons3 = 256
    NumNeurons4 = 128

    #failed attempt to reduce the number of Neurons 
    # NumNeurons1 = 16
    # NumNeurons2 = 8
    # NumNeurons3 = 4
    # NumNeurons4 = 2
    n_target = 1

    # # Session variable -- Need to look up the differences 
    net = tf.InteractiveSession()

    # # Placeholder
    #X = tf.placeholder(dtype=tf.float32, shape=[None, NumElementsPerRow])
    #Y = tf.placeholder(dtype=tf.float32, shape=[None])
    X = tf.placeholder('float', shape=[None, NumElementsPerRow])
    Y = tf.placeholder('float', shape=[None])

    # # Initializers
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Hidden weights
    WeightHidden1 = tf.Variable(weight_initializer([NumElementsPerRow, NumNeurons1]),name='WeightHidden1')
    BiasHidden1 = tf.Variable(bias_initializer([NumNeurons1]),name='BiasHidden1')
    WeightsHidden2 = tf.Variable(weight_initializer([NumNeurons1, NumNeurons2]),name='WeightsHidden2')
    BiasHidden2 = tf.Variable(bias_initializer([NumNeurons2]),name='BiasHidden2')
    WeightsHidden3 = tf.Variable(weight_initializer([NumNeurons2, NumNeurons3]),name='WeightsHidden3')
    BiasHidden3 = tf.Variable(bias_initializer([NumNeurons3]),name='BiasHidden3')
    WeightsHidden4 = tf.Variable(weight_initializer([NumNeurons3, NumNeurons4]),name='WeightsHidden4')
    BiasHidden4 = tf.Variable(bias_initializer([NumNeurons4]),name='BiasHidden4')

    # Output weights
    WeightsOut = tf.Variable(weight_initializer([NumNeurons4, n_target]),name='WeightsOut')
    BiasOut = tf.Variable(bias_initializer([n_target]),name='BiasOut')

    # Hidden layer
    Hidden1 = tf.nn.relu(tf.add(tf.matmul(X, WeightHidden1), BiasHidden1),name='Hidden1')
    Hidden2 = tf.nn.relu(tf.add(tf.matmul(Hidden1, WeightsHidden2), BiasHidden2),name='Hidden2')
    Hidden3 = tf.nn.relu(tf.add(tf.matmul(Hidden2, WeightsHidden3), BiasHidden3),name='Hidden3')
    Hidden4 = tf.nn.relu(tf.add(tf.matmul(Hidden3, WeightsHidden4), BiasHidden4),name='Hidden4')

    # Output layer (transpose!)
    Out = tf.transpose(tf.add(tf.matmul(Hidden4, WeightsOut), BiasOut))

    # Cost function
    Cost = tf.reduce_mean(tf.squared_difference(Out, Y))

    # Optimizer
    Optimizer = tf.train.AdamOptimizer().minimize(Cost)

    # Init
    net.run(tf.global_variables_initializer())

    # # Setup plot
    if(DEBUG == 1):
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        line1, = ax1.plot(y_test)
        line2, = ax1.plot(y_test * 0.5)
        plt.show()

    # Fit neural net
    batch_size = 1
    Cost_train = []
    Cost_test = []

    # Run
    epochs = 5
    for e in range(epochs):

        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run Optimizerimizer with batch
            net.run(Optimizer, feed_dict={X: batch_x, Y: batch_y})

            # Show progress
            if np.mod(i, 50) == 0:
                # Cost train and test
                Cost_train.append(net.run(Cost, feed_dict={X: X_train, Y: y_train}))
                Cost_test.append(net.run(Cost, feed_dict={X: X_test, Y: y_test}))

                # Prediction
                pred = net.run(Out, feed_dict={X: X_test})
                Error = np.average(np.abs(y_test - pred))
                if(Error < TrainThreshold):
                    #ModelName = 'NN' + stocksym
                    #SaveModelAndQuit(net,ModelName)
                        # Export model
                    export_path_base = FLAGS.work_dir + 'NN_' + stocksym
                    export_path = os.path.join(tf.compat.as_bytes(export_path_base),tf.compat.as_bytes(str(FLAGS.model_version)))
                    #export_path = ModelName + '/' + export_path 
                    print('Exporting trained model to', export_path)
                    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

                    tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
                    tensor_info_y = tf.saved_model.utils.build_tensor_info(Out) #THIS IS IMPORTANT!!! NOT THE PLACEHOLDER!!!!!!!!

                    prediction_signature = (
                        tf.saved_model.signature_def_utils.build_signature_def(
                          inputs={'input': tensor_info_x},
                          outputs={'output': tensor_info_y},
                          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

                    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
                    builder.add_meta_graph_and_variables(
                        net, [tf.saved_model.tag_constants.SERVING],
                        signature_def_map={
                          'prediction':
                              prediction_signature,
                      },
                      legacy_init_op=legacy_init_op)

                    builder.save()

                    print('Done exporting!')
                    sys.exit(0)


                if(DEBUG == 1):
                    line2.set_ydata(pred)
                    plt.title('Epoch ' + str(e) + ', Batch ' + str(i))


                    print('Cost Train: ', Cost_train[-1])
                    print('Cost Test: ', Cost_test[-1])
                    print("Pred shape 0: " + str(pred.shape[0]) + ", 1: " + str(pred.shape[1]))
                    print(pred)
                    print("\n")
                    
                    print("Error\n")
                    print(Error)
                    print("\n")

            if(DEBUG == 1):
                plt.pause(0.01)

# def RunNeralNetwork(session,DatabaseTables,stocksym):
#     #Restore model 
#     #load data from table

#     #Run data into model 

#     #  

#TrainThreshold = 5.0

def main():

    #user input
    # learning rate 
    parser = argparse.ArgumentParser();
    parser.add_argument('--sym', dest= 'sym', default='');
    parser.add_argument('--DEBUG',type=int, dest= 'debug', default=0);
    parser.add_argument('--shiftamount',type=int, dest= 'shiftamount', default=1);
    args = parser.parse_args();

    print("Input Arguments: ") 
    print(args)

    #TrainSVMLinearRegression(session,StockPriceMinute,args.sym,args.shiftamount,args.debug);
    TrainNeuralNetwork(session,StockPriceMinute,args.sym,args.shiftamount,0.03,args.debug)

main();
