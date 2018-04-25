#! /bin/sh 

#
# CREATED BY JOHN GRUN
#   APRIL 21 2018 
#
# TESTED BY JOHN GRUN
#
#MODIFIED BY JOHN GRUN 
#


tensorflow_model_server --port=9000 --model_name=NNModel --model_base_path=$(pwd)/NN_RT_AABA &
tensorflow_model_server --port=9001 --model_name=NNModel --model_base_path=$(pwd)/NN_RT_AAPL &
tensorflow_model_server --port=9002 --model_name=NNModel --model_base_path=$(pwd)/NN_RT_AMD &
tensorflow_model_server --port=9003 --model_name=NNModel --model_base_path=$(pwd)/NN_RT_AMZN &
tensorflow_model_server --port=9004 --model_name=NNModel --model_base_path=$(pwd)/NN_RT_C &
tensorflow_model_server --port=9005 --model_name=NNModel --model_base_path=$(pwd)/NN_RT_GOOG &
tensorflow_model_server --port=9006 --model_name=NNModel --model_base_path=$(pwd)/NN_RT_GOOGL &
tensorflow_model_server --port=9007 --model_name=NNModel --model_base_path=$(pwd)/NN_RT_INTC &
tensorflow_model_server --port=9008 --model_name=NNModel --model_base_path=$(pwd)/NN_RT_MSFT &
tensorflow_model_server --port=9009 --model_name=NNModel --model_base_path=$(pwd)/NN_RT_VZ &
#Historical
tensorflow_model_server --port=9010 --model_name=NNModel --model_base_path=$(pwd)/NN_PAST_AABA &
tensorflow_model_server --port=9011 --model_name=NNModel --model_base_path=$(pwd)/NN_PAST_AAPL &
tensorflow_model_server --port=9012 --model_name=NNModel --model_base_path=$(pwd)/NN_PAST_AMD &
tensorflow_model_server --port=9013 --model_name=NNModel --model_base_path=$(pwd)/NN_PAST_AMZN &
tensorflow_model_server --port=9014 --model_name=NNModel --model_base_path=$(pwd)/NN_PAST_C &
tensorflow_model_server --port=9015 --model_name=NNModel --model_base_path=$(pwd)/NN_PAST_GOOG &
tensorflow_model_server --port=9016 --model_name=NNModel --model_base_path=$(pwd)/NN_PAST_GOOGL &
tensorflow_model_server --port=9017 --model_name=NNModel --model_base_path=$(pwd)/NN_PAST_INTC &
tensorflow_model_server --port=9018 --model_name=NNModel --model_base_path=$(pwd)/NN_PAST_MSFT &
tensorflow_model_server --port=9019 --model_name=NNModel --model_base_path=$(pwd)/NN_PAST_VZ &