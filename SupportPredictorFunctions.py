import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

from DatabaseORM import session, StockPriceMinute, StockPriceDay

def GetStockDataList(session,DatabaseTables,stocksym):
    #Grabs the given data elements from the passed in database table object as defined in the ORM for a given stock symbol
    QueryList = session.query(DatabaseTables.high,DatabaseTables.low,DatabaseTables.volume).filter(DatabaseTables.sym == stocksym).all()
    #QueryList = session.query(DatabaseTables.high).filter(DatabaseTables.sym == stocksym).all()
    #, StockPriceMinute.volume
    Formatedlist = []; 
    for DataElement in  QueryList:
        Formatedlist.append(list(DataElement))
    # Returns a np.array of [[row],[row],[row],[row]] where row = [element1, element2,etc] rows are defined by the query above
    return np.array(Formatedlist);
    #print(Formatedlist)

# Xdata = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]]);

def SaveModelAndQuit(sessionname,ModelName):
    #saver = tf.train.Saver()  
    ModelName = './' + ModelName
    saver.save(sessionname, ModelName)
    
  #   export_path_base = ModelName
  #   print('Exporting trained model to' + export_path_base)

  #   builder = saved_model_builder.SavedModelBuilder(export_path_base)

  #   classification_inputs = utils.build_tensor_info(inputs)
  #   classification_outputs_classes = utils.build_tensor_info(prediction_classes)
  #   classification_outputs_scores = utils.build_tensor_info(values)

   
  #   classification_signature = signature_def_utils.build_signature_def(
  #       inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs},
  #       outputs={signature_constants.CLASSIFY_OUTPUT_CLASSES:classification_outputs_classes,signature_constants.CLASSIFY_OUTPUT_SCORES:classification_outputs_scores},
  #       method_name=signature_constants.CLASSIFY_METHOD_NAME)

  #   tensor_info_x = utils.build_tensor_info(x)
  #   tensor_info_y = utils.build_tensor_info(y)

  #   prediction_signature = signature_def_utils.build_signature_def(
  #     inputs={'images': tensor_info_x},
  #     outputs={'scores': tensor_info_y},
  #     method_name=signature_constants.PREDICT_METHOD_NAME)

  #   legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
  
  #   #add the sigs to the servable
  #   builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING],signature_def_map={'predict_images':prediction_signature,signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:classification_signature,},legacy_init_op=legacy_init_op)

  # #save it!
  #   builder.save()

    print('Done exporting!\n')
    print("Exiting Normally\n")


    sys.exit(0)

# def RestoreModel(ModelName):
#     new_saver = tf.train.import_meta_graph(ModelName)
#     new_saver.restore(sess, tf.train.latest_checkpoint('./')) 
#     graph = tf.get_default_graph()
#     w1 = graph.get_tensor_by_name("w1:0")
#     w2 = graph.get_tensor_by_name("w2:0")
#     feed_dict ={w1:13.0,w2:17.0}