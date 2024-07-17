import cv2
import time


class FaceDetector:
    def __init__(self, cascade_path, min_detection_scale=1.1, min_neighbors=5):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.min_detection_scale = min_detection_scale
        self.min_neighbors = min_neighbors

    def find_faces(self, img, draw=True):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=self.min_detection_scale,
                                                   minNeighbors=self.min_neighbors)
        bboxs = []
        for (x, y, w, h) in faces:
            bbox = (x, y, x + w, y + h)  # Convert to (x1, y1, x2, y2) format
            bboxs.append(bbox)
            if draw:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        return img, bboxs


def main():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(cascade_path)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.resize(img, (800, 480), interpolation=cv2.INTER_AREA)
        img, bboxs = detector.find_faces(img)

        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
