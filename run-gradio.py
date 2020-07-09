import tensorflow as tf
import sys
import os
import gradio
from PIL import Image
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

classes = [line.rstrip() for line in tf.gfile.GFile("hot_dog_labels.txt")]

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
        # ^ Feeding data as input and find the first prediction
        result = session.run(tensor, {'DecodeJpeg/contents:0': data})

        top_results = result[0].argsort()[-len(result[0]):][::-1]
        label = {}
        for type in top_results:
            hot_dog_or_not = classes[type]
            score = result[0][type]
            label[hot_dog_or_not] = str(score)
    os.remove(filename)
    return label

examples = [
    ["Big-Italian-Salad.jpg"],
    ["hotdog2.jpeg"],
    ["soup.jpg"],
    ["hotdog.jpeg"]
]

gradio.Interface(predict, "image", "label", title="Hotdog / Not Hotdog",
                 description="Jian Yang's infamous model. Credit to "
                             "https://github.com/VPanjeta/hotdog-or-not-hotdog",
                 examples=examples
                 ).launch(
    inbrowser=True)