'''
Created on 2018年10月17日

@author: 95890
'''

"""Send text to tensorflow serving and gets result
"""


# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf
import data_helpers
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib import learn
import numpy as np


tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string('server', '192.168.99.100:8500',
                           'PredictionService host:port')
FLAGS = tf.flags.FLAGS

x_text=[]
y=[]
max_document_length=40


def main(_):

  # Send request
    # See prediction_service.proto for gRPC request/response details.
  testStr =["wisegirls is its low-key quality and genuine"]

  
  x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
  max_document_length = max([len(x.split(" ")) for x in x_text])

  vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
  vocab_processor.fit(x_text)
  x = np.array(list(vocab_processor.fit_transform(testStr)))
  
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = "text_classified_model"
  request.model_spec.signature_name = 'textclassified'
  dropout_keep_prob = np.float(1.0)
  
  request.inputs['inputX'].CopyFrom(
  tf.contrib.util.make_tensor_proto(x, shape=[1,40],dtype=np.int32))
  
  request.inputs['input_dropout_keep_prob'].CopyFrom(
  tf.contrib.util.make_tensor_proto(dropout_keep_prob, shape=[1],dtype=np.float))
  
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  print(result)


if __name__ == '__main__':
  tf.app.run()
