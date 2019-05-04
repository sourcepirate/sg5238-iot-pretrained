import os
import keras2graph
import ncs_process

model_file = open("models/model.json")
weight_file = open("models/weights.h5")


keras2graph.keras_to_graph('models/model.json', 'input_1', 'activation_7/Softmax', 'Keras_Model/weights.h5', './graph', False)

ncs_model, device = ncs_process.ready_ai_ncs('./graph', device_index=0)
output = ncs_process.ncs_predict(ncs_model, example_image)
print(output[0][0])



