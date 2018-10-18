from tensorflow.python.platform import gfile
import tensorflow as tf
 
 
sess = tf.Session()
with gfile.FastGFile('./data/model/1/saved_model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    print(f.read())
    