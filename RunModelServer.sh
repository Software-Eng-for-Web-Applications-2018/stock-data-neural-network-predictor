#! /bin/sh 
tensorflow_model_server --port=9000 --model_name=NNModel --model_base_path=$(pwd)/NN_AABA &
tensorflow_model_server --port=9001 --model_name=NNModel --model_base_path=$(pwd)/NN_AAPL &
tensorflow_model_server --port=9002 --model_name=NNModel --model_base_path=$(pwd)/NN_AMD &
tensorflow_model_server --port=9003 --model_name=NNModel --model_base_path=$(pwd)/NN_AMZN &
tensorflow_model_server --port=9004 --model_name=NNModel --model_base_path=$(pwd)/NN_C &
tensorflow_model_server --port=9005 --model_name=NNModel --model_base_path=$(pwd)/NN_GOOG &
tensorflow_model_server --port=9006 --model_name=NNModel --model_base_path=$(pwd)/NN_GOOGL &
tensorflow_model_server --port=9007 --model_name=NNModel --model_base_path=$(pwd)/NN_INTC &
tensorflow_model_server --port=9008 --model_name=NNModel --model_base_path=$(pwd)/NN_MSFT &
tensorflow_model_server --port=9009 --model_name=NNModel --model_base_path=$(pwd)/NN_VZ &
