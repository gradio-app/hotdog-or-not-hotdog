import tensorflow as tf
import sys
import os
import gradio
from PIL import Image
import random

def predict(inp):
    inp = Image.fromarray(inp.astype('uint8'), 'RGB')

    filename = str(random.getrandbits(10)) + ".jpeg"
    inp.save(filename, "JPEG")
    image_file = tf.gfile.FastGFile(filename, 'rb')
    data = image_file.read()
    
    with tf.gfile.FastGFile("hot_dog_graph.pb", 'rb') as inception_graph:
        definition = tf.GraphDef()
        definition.ParseFromString(inception_graph.read())
        _ = tf.import_graph_def(definition, name='')
    
    with tf.Session() as session:
        tensor = session.graph.get_tensor_by_name('final_result:0')
        result = session.run(tensor, {'DecodeJpeg/contents:0': data})[0]
    
    os.remove(filename)
    return {'hot dog': result[0], 'not hot dog': result[1]}

examples = [
    ["Big-Italian-Salad.jpg"],
    ["hotdog2.jpeg"],
    ["soup.jpg"],
    ["hotdog.jpeg"]
]

gradio.Interface(predict, "image", "label", title="Hotdog or Not?",
                 description="Based off of Jian Yang's infamous model from the TV show Silicon Valley, this model has one job, just like its name suggests.",
                 thumbnail="https://raw.githubusercontent.com/gradio-app/hotdog-or-not-hotdog/master/gradio-screenshot2.png",
                 examples=examples
                 ).launch(
    inbrowser=True)
