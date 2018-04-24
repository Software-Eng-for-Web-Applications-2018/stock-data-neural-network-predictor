#! /bin/sh 
tensorflow_model_server --port=9000 --model_name=NNModel --model_base_path=$(pwd)
