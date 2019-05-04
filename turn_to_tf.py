from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf

model_content = open("models/model.json").read()

K.set_learning_phase(0)
model = model_from_json(config)
model.load_weights("models/weights.h5")

saver = tf.train.Saver()
sess = K.get_session()
saver.save(sess, "./TF_Model/tf_model")

fw = tf.summary.FileWriter('logs', sess.graph)
fw.close()

