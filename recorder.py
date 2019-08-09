import cv2
import yoloPredict

EXIT_KEY = 'e'

class Detector:
    def __init__(self):
        # Create VideoCapture object to use webcam
        self.camera = cv2.VideoCapture(0)

        # get height and width of webcam, it is important for the darknet
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Create predictor object
        self.predictor = yoloPredict.Predictor(width, height)

    def run(self):
        # Infinite loop to stream camera
        while True:

            # Getting camera image, check is for successful read, frame is the read video frame
            check, frame = self.camera.read()

            # if the check is successful, the predictted frame is shown
            if check:
                predicted_frame = self.predictor.predict(frame)
                cv2.imshow('Darknet Object Detector', predicted_frame)

            # wait 1ms for key press
            key = cv2.waitKey(1)

            # if defined exit key was pressed video cycle shall be stopped
            if key == ord(EXIT_KEY):
                break

        # dismantle camera object and openCV framework
        self.camera.release()
        cv2.destroyAllWindows()