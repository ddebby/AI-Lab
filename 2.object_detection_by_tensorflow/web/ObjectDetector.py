import tensorflow as tf
import numpy as np
import imutils
import cv2
import numpy
from PIL import Image
# initialize a set of colors for our class labels
#COLORS = np.random.uniform(0, 255, size=(5, 3))
COLORS = [
	 [255,0,0], 	
         [0,255,255],
         [255,255,255],
         [0,0,255],
         [0,255,0]
	]

categoryIdx = {1: {'id': 1, 'name': 'blue'},
 2: {'id': 2, 'name': 'yellow'},
 3: {'id': 3, 'name': 'white'},
 4: {'id': 4, 'name': 'red'},
 5: {'id': 5, 'name': 'none'}}

model_file = "/home/inesa/helmet-webserver/model/frozen_inference_graph.pb"

class Detector:
    def __init__(self):
        # initialize the model
        self.model = tf.Graph()

        # create a context manager that makes this model the default one for
        # execution
        with self.model.as_default():
            # initialize the graph definition
            graphDef = tf.GraphDef()

            # load the graph from disk
            with tf.gfile.GFile(model_file, "rb") as f:
                serializedGraph = f.read()
                graphDef.ParseFromString(serializedGraph)
                tf.import_graph_def(graphDef, name="")

    def detectObject(self, imName):
        image = cv2.cvtColor(numpy.array(imName), cv2.COLOR_BGR2RGB)
        # create a session to perform inference
        with self.model.as_default():
            with tf.Session(graph=self.model) as sess:
                # grab a reference to the input image tensor and the boxes
                # tensor
                imageTensor = self.model.get_tensor_by_name("image_tensor:0")
                boxesTensor = self.model.get_tensor_by_name("detection_boxes:0")

                # for each bounding box we would like to know the score
                # (i.e., probability) and class label
                scoresTensor = self.model.get_tensor_by_name("detection_scores:0")
                classesTensor = self.model.get_tensor_by_name("detection_classes:0")
                numDetections = self.model.get_tensor_by_name("num_detections:0")

                (H, W) = image.shape[:2]

                # check to see if we should resize along the width
                if W > H and W > 1000:
                    image = imutils.resize(image, width=1000)

                # otherwise, check to see if we should resize along the
                # height
                elif H > W and H > 1000:
                    image = imutils.resize(image, height=1000)

                # prepare the image for detection
                (H, W) = image.shape[:2]
                output = image.copy()
                image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image, axis=0)

                # perform inference and compute the bounding boxes,
                # probabilities, and class labels
                (boxes, scores, labels, N) = sess.run(
                    [boxesTensor, scoresTensor, classesTensor, numDetections],
                    feed_dict={imageTensor: image})

                # squeeze the lists into a single dimension
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                labels = np.squeeze(labels)

                # loop over the bounding box predictions
                for (box, score, label) in zip(boxes, scores, labels):
                    # if the predicted probability is less than the minimum
                    # confidence, ignore it
                    if score < 0.5:
                        continue

                    # scale the bounding box from the range [0, 1] to [W, H]
                    (startY, startX, endY, endX) = box
                    startX = int(startX * W)
                    startY = int(startY * H)
                    endX = int(endX * W)
                    endY = int(endY * H)

                    # draw the prediction on the output image
                    label = categoryIdx[label]
                    idx = int(label["id"]) - 1
                    label = "{}: {:.2f}".format(label["name"], score)
                    cv2.rectangle(output, (startX, startY), (endX, endY),
                        COLORS[idx], 2)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(output, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)    
        img = cv2.imencode('.jpg', output)[1].tobytes()
        return img
