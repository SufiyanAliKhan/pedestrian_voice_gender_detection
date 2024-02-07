import tkinter as tk
from tkinter import filedialog
from detection_utils import PedestrianDetector, draw_boxes
import cv2
from PIL import Image, ImageTk

class PedestrianDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Pedestrian Detection")

        # Create and configure the title label
        self.title_label = tk.Label(master, text="Pedestrian Detection", font=("Helvetica", 16))
        self.title_label.pack(pady=10)

        # Create buttons for image and video selection
        self.image_button = tk.Button(master, text="Select Image", command=self.select_image)
        self.image_button.pack(pady=5)

        self.video_button = tk.Button(master, text="Select Video", command=self.process_and_display_video)
        self.video_button.pack(pady=5)

        # Initialize the pedestrian detector
        self.detector = PedestrianDetector("../models/yolov3.weights", "../models/yolov3.cfg")

        self.detectorforvideo = PedestrianDetector("../models/yolov3-tiny.weights", "../models/yolov3-tiny.cfg")

        # Initialize video stream and window
        self.video_stream = None
        self.video_window = None


    def select_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            print("Selected Image:", file_path)

            # Detect pedestrians in the image
            boxes, confidences, class_ids = self.detector.detect_pedestrians(file_path)

            # Load the image and draw bounding boxes with class labels
            image = cv2.imread(file_path)
            class_labels = ["Pedestrian"]  # Add more labels if needed
            image_with_boxes = draw_boxes(image.copy(), boxes, confidences, class_ids, class_labels)

            # Display the image with bounding boxes in a new window
            self.display_image_in_window(image_with_boxes)

    def process_and_display_video(self):
        file_path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video files", "*.mp4;*.avi")])
        if file_path:
            print("Selected Video:", file_path)

            # Release previous video stream and window if they exist
            if self.video_stream is not None:
                self.video_stream.release()
                cv2.destroyAllWindows()

            # Initialize a new video stream
            self.video_stream = cv2.VideoCapture(file_path)

            # Process and save the video with detected pedestrians
            self.process_video(file_path)

    def process_video(self, video_path):
        # Get video properties
        width = int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video_stream.get(cv2.CAP_PROP_FPS)

        # Create video writer
        output_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("Video files", "*.avi")])
        if output_path:
            video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

            while True:
                ret, frame = self.video_stream.read()

                if not ret:
                    break

                # Detect pedestrians in the frame
                boxes, confidences, class_ids = self.detectorforvideo.detect_pedestrians(frame)

                # Draw bounding boxes on the frame
                frame_with_boxes = draw_boxes(frame.copy(), boxes, confidences, class_ids, class_labels=["Pedestrian"])

                # Write the frame to the output video
                video_writer.write(frame_with_boxes)

                # Break the loop if the 'Esc' key is pressed
                if cv2.waitKey(1) == 27:
                    break

            # Release video writer
            video_writer.release()
            print("Video Saved")

    def display_image_in_window(self, image):
        # Convert image from BGR to RGB (OpenCV uses BGR, while Tkinter uses RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert image to ImageTk format
        image_tk = ImageTk.PhotoImage(Image.fromarray(image_rgb))

        # Create a new window
        window = tk.Toplevel(self.master)
        window.title("Pedestrian Detection Result")

        # Create a label to display the image in the new window
        image_label = tk.Label(window, image=image_tk)
        image_label.image = image_tk
        image_label.pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = PedestrianDetectionApp(root)
    root.mainloop()
