# USAGE
# $ python predict_video.py \
# 	--model dlib_front_and_rear_vehicles_v1/experiments/exported_model/frozen_inference_graph.pb \
# 	--labels dlib_front_and_rear_vehicles_v1/records/classes.pbtxt \
# 	--input example_input.avi --output output.avi --num-classes 2

# import the necessary packages

import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2

from PIL import Image, ImageDraw, ImageFont
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="base path for frozen checkpoint detection graph")

ap.add_argument("-i", "--input", required=False,
	help="path to input video")
ap.add_argument("-o", "--output", required=False,
	help="path to output video")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability used to filter weak detections")
args = vars(ap.parse_args())

# initialize the colors list and the model

font_file = r'/font/msyhl.ttc'
label2short = {1:"白色",2:"红色",3:"黄色",4:"蓝色",5:"未佩戴"}
font = ImageFont.truetype(font_file,10)
label2color = {1:(255,255,255),2:(0,0,255),3:(0,255,255),4:(255,0,0),5:(0,255,0)}

caution = Image.open("./caution.png")
caution = caution.resize((50,50))
caution = cv2.cvtColor(np.array(caution), cv2.COLOR_BGR2RGB)
caution = Image.fromarray(np.array(caution))

model = tf.Graph()

# create a context manager that makes this model the default one for
# execution
with model.as_default():
	# initialize the graph definition
	graphDef = tf.GraphDef()

	# load the graph from disk
	with tf.gfile.GFile(args["model"], "rb") as f:
		serializedGraph = f.read()
		graphDef.ParseFromString(serializedGraph)
		tf.import_graph_def(graphDef, name="")

# load the class labels from disk
categoryIdx = {1: {'id': 1, 'name': 'white'},
				2: {'id': 2, 'name': 'red'},
				3: {'id': 3, 'name': 'yellow'},
				4: {'id': 4, 'name': 'blue'},
				5: {'id': 5, 'name': 'none'}}

# create a session to perform inference
with model.as_default():
	with tf.Session(graph=model) as sess:
		# initialize the points to the video files
		if not args["input"]:
			 stream = cv2.VideoCapture(1)  # Change only if you have more than one webcams

		else:
			stream = cv2.VideoCapture(args["input"])
		writer = None

		# loop over frames from the video file stream
		while True:
			# grab the next frame
			(grabbed, image) = stream.read()
			# if not args["input"]:
			# 	image = imutils.resize(image, width=1000)
				#image = cv2.cvtColor(numpy.array(imName), cv2.COLOR_BGR2RGB)
			# if the frame was not grabbed, then we have reached the
			# end of the stream
			if not grabbed:
				break

			# grab a reference to the input image tensor and the
			# boxes
			imageTensor = model.get_tensor_by_name("image_tensor:0")
			boxesTensor = model.get_tensor_by_name("detection_boxes:0")

			# for each bounding box we would like to know the score
			# (i.e., probability) and class label
			scoresTensor = model.get_tensor_by_name("detection_scores:0")
			classesTensor = model.get_tensor_by_name("detection_classes:0")
			numDetections = model.get_tensor_by_name("num_detections:0")

			# grab the image dimensions
			(H, W) = image.shape[:2]

			# # check to see if we should resize along the width
			if W > H and W > 1000:
				image = imutils.resize(image, width=1000)

			# otherwise, check to see if we should resize along the
			# height
			elif H > W and H > 1000:
				image = imutils.resize(image, height=1000)

			#output = image.copy()
			output = Image.fromarray(np.array(image))
			image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
			image = np.expand_dims(image, axis=0)

			# if the video writer is None, initialize if
			if (writer is None) and args["output"]:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(args["output"], fourcc, 20,
					(W, H), True)

			# perform inference and compute the bounding boxes,
			# probabilities, and class labels
			(boxes, scores, labels, N) = sess.run(
				[boxesTensor, scoresTensor, classesTensor, numDetections],
				feed_dict={imageTensor: image})

			# squeeze the lists into a single dimension
			boxes = np.squeeze(boxes)
			scores = np.squeeze(scores)
			labels = np.squeeze(labels)
			draw = ImageDraw.Draw(output)
			# loop over the bounding box predictions
			for (box, score, label) in zip(boxes, scores, labels):
				# if the predicted probability is less than the minimum
				# confidence, ignore it
				if score < args["min_confidence"]:
					continue

				# scale the bounding box from the range [0, 1] to [W, H]
				(startY, startX, endY, endX) = box
				xmin = int(startX * W)
				ymin = int(startY * H)
				xmax = int(endX * W)
				ymax = int(endY * H)


				showLabel = label2short[label] + ":" + ('%.2f' % score)
				draw.rectangle([xmin, ymin, xmax, ymax], outline=label2color[label])
				labelSize, baseLine = cv2.getTextSize(showLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
				ymin = max(ymin, labelSize[1])

				draw.rectangle([xmin, ymin-labelSize[1],xmin+labelSize[0]/2,ymin],
							outline=label2color[label],fill=label2color[label])

				draw.text([xmin, ymin-labelSize[1]], showLabel, (0,0,0),font=font)

				# more process for none users
				if label == 5:
					output.paste(caution,(xmin-50, ymin))

				# draw the prediction on the output image

			if args["output"]:
				# write the frame to the output file
				writer.write(np.array(output))
			else:
				# Display output
				#cv2.imshow('object detection', cv2.resize(output, (400, 300)))

				cv2.imshow('object detection', np.array(output))
				if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
		# close the video file pointers
		if args["output"]:
			writer.release()
		stream.release()
