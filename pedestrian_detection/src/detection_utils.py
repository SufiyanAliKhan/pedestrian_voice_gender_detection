import cv2
import numpy as np

class PedestrianDetector:
    def __init__(self, weights_path, config_path, confidence_threshold=0.5):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.confidence_threshold = confidence_threshold

    def detect_pedestrians(self, input_data):
        if isinstance(input_data, str):
            # Input is a file path (image)
            image = cv2.imread(input_data)
        else:
            # Input is a video frame
            image = input_data

        height, width = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        outs = self.net.forward(self.get_output_layer_names())

        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold and class_id == 0:  # Class ID for pedestrians is typically 0
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                    x, y = center_x - w // 2, center_y - h // 2

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)

        # Check if there are any detections before accessing indices
        if indices is not None and len(indices) > 0:
            indices = indices.flatten()
            filtered_boxes = [boxes[i] for i in indices]
            filtered_confidences = [confidences[i] for i in indices]
            filtered_class_ids = [class_ids[i] for i in indices]
        else:
            filtered_boxes = []
            filtered_confidences = []
            filtered_class_ids = []

        return filtered_boxes, filtered_confidences, filtered_class_ids

    def get_output_layer_names(self):
        layer_names = self.net.getUnconnectedOutLayersNames()
        print(f"layer_names: {layer_names}")

        if isinstance(layer_names, tuple) and all(isinstance(name, str) for name in layer_names):
            # For newer versions of OpenCV, layer_names is a tuple of strings
            return list(layer_names)
        else:
            # Handle other cases or raise an exception
            raise ValueError("Unsupported layer_names format")


def draw_boxes(image, boxes, confidences, class_ids, class_labels):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        class_label = class_labels[class_id]

        color = (0, 255, 0)  # Green color for bounding boxes
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Display class label and confidence score
        label = f"{class_label} {confidence:.2f}"
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


