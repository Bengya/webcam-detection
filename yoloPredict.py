import cv2
import time
import numpy as np

CONFIDENCE = 0.5
THRESHOLD = 0.4
configPath = 'config/yolov3.cfg'
weightsPath = 'config/yolov3.weights'
classesPath = 'config/objects.txt'


class Predictor:
    def __init__(self, w, h):

        self.img_width = w
        self.img_height = h
        try:
            # Initializing the network, configuration and weights needs to be added.
            self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        except cv2.error:
            # Weights are not uploaded to version control so it should be downloaded separately
            raise FileNotFoundError("Darknet weights might not be added to /config\nYou can download it from https://pjreddie.com/media/files/yolov3.weights")

        # Prediction classes shall be defined in a .txt then populated in a list
        self.classes = None
        with open(classesPath, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Initialize random colors for visualization
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")

    def predict(self, image):
        # Construct output layer name list, which is needed for feeding data through the network
        layer_names = self.net.getLayerNames()
        layer_names = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Construct a blob based on the image we want to feed the network.
        # The blob involves some image processing and transorfming to a form which is understandable by the network
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.img_width, self.img_height), swapRB=True, crop=False)

        # Set the blob as the input of the network
        self.net.setInput(blob)

        start = time.time()

        # Passing the blobs through the network up until the last layers.
        # It will return the bounding boxes of each category, predefined in classes with the probability.
        layer_outputs = self.net.forward(layer_names)
        end = time.time()

        print("Prediction took: {:.6f} seconds".format(end-start))
        return self.visualize(image, layer_outputs)

    def visualize(self, image, layer_outputs):
        boxes = []
        confidences = []
        class_ids = []
        for output in layer_outputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                print(detection)
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > CONFIDENCE:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([self.img_width, self.img_height, self.img_width, self.img_height])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.colors[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.classes[class_ids[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image