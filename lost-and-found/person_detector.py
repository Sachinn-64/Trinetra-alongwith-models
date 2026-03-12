import cv2
import face_recognition
import numpy as np
import os
import argparse
from datetime import datetime


class PersonDetector:
    def __init__(self, person_image_path, video_path):
        """
        Initialize the person detector.

        Args:
            person_image_path: path to the target person's image
            video_path: video file OR RTSP stream URL
        """
        self.person_image_path = person_image_path
        self.video_path = video_path
        self.person_encoding = None
        self.output_dir = "detected_frames"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_person_encoding(self):
        """Load and encode the target person's face."""
        try:
            person_image = face_recognition.load_image_file(self.person_image_path)

            face_encodings = face_recognition.face_encodings(person_image)

            if len(face_encodings) == 0:
                print("No face detected in the person's image.")
                return False

            self.person_encoding = face_encodings[0]
            print("Person encoding loaded successfully.")
            return True

        except Exception as e:
            print("Error loading person image:", e)
            return False

    def detect_person(self, tolerance=0.6, frame_skip=5):
        """
        Detect the person in a video file or RTSP stream.

        Args:
            tolerance: face matching tolerance
            frame_skip: process every nth frame
        """

        if self.person_encoding is None:
            print("Person encoding not loaded.")
            return

        print("Opening stream/video:", self.video_path)

        video = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)

        if not video.isOpened():
            print("Error: Could not open video/stream.")
            return

        frame_count = 0
        detected_frames = 0

        print("Starting detection...")

        while True:

            ret, frame = video.read()

            if not ret:
                print("Stream ended or frame not received.")
                break

            if frame_count % frame_skip == 0:

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_frame, face_locations
                )

                for i, face_encoding in enumerate(face_encodings):

                    matches = face_recognition.compare_faces(
                        [self.person_encoding],
                        face_encoding,
                        tolerance=tolerance
                    )

                    if matches[0]:

                        top, right, bottom, left = face_locations[i]

                        cv2.rectangle(
                            frame,
                            (left, top),
                            (right, bottom),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            "PERSON FOUND",
                            (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2
                        )

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        filename = f"detection_{timestamp}_{frame_count}.jpg"

                        frame_path = os.path.join(self.output_dir, filename)

                        cv2.imwrite(frame_path, frame)

                        detected_frames += 1

                        print("Person detected → saved:", filename)

            if frame_count % 100 == 0:
                print("Processed frames:", frame_count)

            frame_count += 1

        video.release()

        print("\nDetection complete")
        print("Total frames:", frame_count)
        print("Detections:", detected_frames)
        print("Saved in:", self.output_dir)

    def run_detection(self, tolerance=0.6, frame_skip=5):

        print("===== Lost Person Detection =====")
        print("Person image:", self.person_image_path)
        print("Video/Stream:", self.video_path)
        print()

        if not self.load_person_encoding():
            return

        self.detect_person(tolerance, frame_skip)


def main():

    parser = argparse.ArgumentParser(
        description="Detect a person in video or RTSP stream"
    )

    parser.add_argument(
        "--person",
        required=True,
        help="Path to person's image"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Video file OR RTSP stream URL"
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=10
    )

    args = parser.parse_args()

    if not os.path.exists(args.person):
        print("Person image not found.")
        return

    detector = PersonDetector(args.person, args.video)

    detector.run_detection(args.tolerance, args.frame_skip)


if __name__ == "__main__":
    main()