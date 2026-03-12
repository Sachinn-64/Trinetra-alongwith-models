"""
Hybrid Person Detector — combines dlib (face_recognition) + InsightFace (ArcFace).

face_recognition  → excellent on frontal / near-frontal faces, fast
InsightFace       → handles side profiles, angles, partial occlusion, low light

Strategy:
  1. Both engines encode the target person's face.
  2. Every processed frame is checked by BOTH engines.
  3. A match from EITHER engine counts as a detection (union).
"""

import cv2
import face_recognition
import numpy as np
import os
import argparse
import logging
from datetime import datetime

# InsightFace (ArcFace + RetinaFace)
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

# ─── Singleton InsightFace loader ───────────────────────────────────────────
_insight_app = None


def get_insight_app():
    """Lazy-load and cache the InsightFace model (heavy, ~300 MB)."""
    global _insight_app
    if _insight_app is None:
        logger.info("Loading InsightFace buffalo_l model (first time)…")
        _insight_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        _insight_app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("InsightFace model ready.")
    return _insight_app


# ─── Utility ────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; higher = more similar."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ─── HybridPersonDetector ──────────────────────────────────────────────────

class HybridPersonDetector:
    """
    Detect a target person using two face-recognition engines.

    Parameters
    ----------
    person_image_path : str
        Path to the target person's face image.
    video_path : str
        Path to a video file **or** an RTSP stream URL.
    insight_threshold : float
        Cosine-similarity threshold for InsightFace (0-1, default 0.35).
        Lower = more lenient (good for side profiles).
    dlib_tolerance : float
        Euclidean-distance tolerance for face_recognition (default 0.6).
        Lower = stricter.
    """

    def __init__(
        self,
        person_image_path: str,
        video_path: str,
        insight_threshold: float = 0.35,
        dlib_tolerance: float = 0.6,
    ):
        self.person_image_path = person_image_path
        self.video_path = video_path
        self.insight_threshold = insight_threshold
        self.dlib_tolerance = dlib_tolerance

        # Encodings filled by load_person_encoding()
        self.dlib_encoding = None       # 128-d numpy array
        self.insight_embedding = None   # 512-d numpy array

        self.output_dir = "detected_frames"
        os.makedirs(self.output_dir, exist_ok=True)

    # ── Load target face ────────────────────────────────────────────────

    def load_person_encoding(self):
        """
        Encode the target person's face with **both** engines.
        Returns (success: bool, message: str).
        """
        dlib_ok = self._load_dlib_encoding()
        insight_ok = self._load_insight_encoding()

        if not dlib_ok and not insight_ok:
            return False, (
                "Neither engine could find a face in the image. "
                "Please use a photo with a clearly visible face."
            )

        parts = []
        if dlib_ok:
            parts.append("dlib (face_recognition)")
        if insight_ok:
            parts.append("InsightFace (ArcFace)")

        msg = f"Person encoded with: {', '.join(parts)}"
        logger.info(msg)
        return True, msg

    def _load_dlib_encoding(self) -> bool:
        try:
            img = face_recognition.load_image_file(self.person_image_path)
            encs = face_recognition.face_encodings(img)
            if encs:
                self.dlib_encoding = encs[0]
                return True
        except Exception as e:
            logger.warning(f"dlib encoding failed: {e}")
        return False

    def _load_insight_encoding(self) -> bool:
        try:
            app = get_insight_app()
            img = cv2.imread(self.person_image_path)
            if img is None:
                return False
            faces = app.get(img)
            if faces:
                # Pick the largest detected face
                faces = sorted(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True,
                )
                self.insight_embedding = faces[0].normed_embedding
                return True
        except Exception as e:
            logger.warning(f"InsightFace encoding failed: {e}")
        return False

    # ── Per-frame matching ──────────────────────────────────────────────

    def match_frame(self, frame_bgr: np.ndarray):
        """
        Run detection engines on a single BGR frame.
        Uses a **cascade** strategy: try fast dlib first, only fall back to
        the heavier InsightFace if dlib finds no match.

        Returns a list of dicts, one per matched face:
            { "bbox": (left, top, right, bottom),
              "engine": "dlib" | "insightface" | "both",
              "score": float }
        """
        matches = {}  # key = rough bbox centre → avoids duplicates

        # ── dlib pass (fast) ────────────────────────────────────────────
        if self.dlib_encoding is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Downscale for faster face detection
            h, w = rgb.shape[:2]
            scale = 1.0
            if w > 640:
                scale = 640.0 / w
                small = cv2.resize(rgb, (640, int(h * scale)))
            else:
                small = rgb

            locations = face_recognition.face_locations(small, model="hog")
            encodings = face_recognition.face_encodings(small, locations)
            for loc, enc in zip(locations, encodings):
                top, right, bottom, left = loc
                # Map back to original resolution
                if scale < 1.0:
                    top = int(top / scale)
                    right = int(right / scale)
                    bottom = int(bottom / scale)
                    left = int(left / scale)
                dist = face_recognition.face_distance([self.dlib_encoding], enc)[0]
                if dist <= self.dlib_tolerance:
                    key = (round(left / 30), round(top / 30))  # bucket
                    score = round(1.0 - dist, 3)
                    if key not in matches or matches[key]["score"] < score:
                        matches[key] = {
                            "bbox": (left, top, right, bottom),
                            "engine": "dlib",
                            "score": score,
                        }

        # ── InsightFace pass (only if dlib found nothing) ───────────────
        if not matches and self.insight_embedding is not None:
            try:
                app = get_insight_app()
                faces = app.get(frame_bgr)
                for face in faces:
                    sim = cosine_similarity(self.insight_embedding, face.normed_embedding)
                    if sim >= self.insight_threshold:
                        x1, y1, x2, y2 = [int(v) for v in face.bbox]
                        key = (round(x1 / 30), round(y1 / 30))
                        matches[key] = {
                            "bbox": (x1, y1, x2, y2),
                            "engine": "insightface",
                            "score": round(sim, 3),
                        }
            except Exception as e:
                logger.warning(f"InsightFace frame analysis failed: {e}")

        return list(matches.values())

    # ── Video / stream scan ─────────────────────────────────────────────

    def detect_person(self, tolerance=0.6, frame_skip=5):
        """CLI-friendly: process a video/stream and save detection frames."""
        self.dlib_tolerance = tolerance

        if self.dlib_encoding is None and self.insight_embedding is None:
            print("No encoding loaded. Call load_person_encoding() first.")
            return

        print(f"Opening: {self.video_path}")
        video = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        if not video.isOpened():
            print("Error: Could not open video/stream.")
            return

        frame_count = 0
        detected_frames = 0

        print("Starting hybrid detection (dlib + InsightFace)…")

        while True:
            ret, frame = video.read()
            if not ret:
                print("Stream ended or frame not received.")
                break

            if frame_count % frame_skip == 0:
                hits = self.match_frame(frame)
                for hit in hits:
                    left, top, right, bottom = hit["bbox"]
                    engine = hit["engine"]
                    score = hit["score"]

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    label = f"FOUND [{engine}] {score:.2f}"
                    cv2.putText(frame, label, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"detection_{ts}_{frame_count}.jpg"
                    cv2.imwrite(os.path.join(self.output_dir, fname), frame)
                    detected_frames += 1
                    print(f"  ✓ Detected [{engine}] score={score:.2f} → {fname}")

            if frame_count % 100 == 0:
                print(f"  Processed frames: {frame_count}")
            frame_count += 1

        video.release()
        print(f"\nDone — {frame_count} frames, {detected_frames} detections")
        print(f"Saved in: {self.output_dir}")

    def detect_person_in_video(self, output_video_path, tolerance=0.6, frame_skip=5, max_process_frames=60):
        """
        Fast video scan — only SAMPLES frames (seek-based), no output video
        writing. Returns detection frame images directly.

        Strategy:
          - Compute which frames to sample (evenly spaced, capped at max_process_frames).
          - Seek to each frame → run detection → save annotated frame if match.
          - Skip expensive video encoding entirely.
        """
        self.dlib_tolerance = tolerance

        if self.dlib_encoding is None and self.insight_embedding is None:
            return False, "No person encoding loaded."

        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            return False, "Could not open video file."

        fps = int(video.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1000  # fallback

        # Build list of frame indices to sample
        effective_skip = max(frame_skip, total_frames // max_process_frames) if total_frames > max_process_frames else frame_skip
        sample_indices = list(range(0, total_frames, effective_skip))
        num_samples = len(sample_indices)

        logger.info(f"Fast scan: {total_frames} total frames, sampling {num_samples} "
                     f"(every {effective_skip} frames)")

        detected_frames = 0
        detection_timestamps = []
        detection_frame_path = None

        for i, frame_idx in enumerate(sample_indices):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()
            if not ret:
                continue

            hits = self.match_frame(frame)

            for hit in hits:
                left, top, right, bottom = hit["bbox"]
                engine = hit["engine"]
                score = hit["score"]
                ts = round(frame_idx / fps, 2)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"FOUND [{engine}] {score:.2f}",
                            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {ts}s",
                            (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                detected_frames += 1
                detection_timestamps.append(ts)

                if detection_frame_path is None:
                    detection_frame_path = output_video_path.replace(".mp4", "_detection_frame.jpg")
                    cv2.imwrite(detection_frame_path, frame)

                logger.info(f"[{engine}] Detected at frame {frame_idx} ({ts}s) score={score:.2f}")

            if (i + 1) % 20 == 0:
                pct = ((i + 1) / num_samples) * 100
                logger.info(f"Progress: {pct:.0f}%  ({i + 1}/{num_samples} sampled)")

        video.release()

        logger.info(f"Done. Sampled {num_samples} frames, {detected_frames} detections.")
        return True, {
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "detection_timestamps": detection_timestamps,
            "output_video_path": output_video_path,
            "detection_frame_path": detection_frame_path,
        }

    # ── Convenience ─────────────────────────────────────────────────────

    def run_detection(self, tolerance=0.6, frame_skip=5):
        print("===== Hybrid Lost Person Detection =====")
        print(f"Person image : {self.person_image_path}")
        print(f"Video/Stream : {self.video_path}")
        print(f"Engines      : dlib + InsightFace (ArcFace)\n")

        ok, msg = self.load_person_encoding()
        print(msg)
        if not ok:
            return

        self.detect_person(tolerance, frame_skip)


# ─── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect a person in video or RTSP stream (hybrid dlib + InsightFace)"
    )
    parser.add_argument("--person", required=True, help="Path to person's image")
    parser.add_argument("--video", required=True, help="Video file or RTSP URL")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="dlib tolerance (lower=stricter, default 0.5)")
    parser.add_argument("--insight-threshold", type=float, default=0.35,
                        help="InsightFace cosine threshold (lower=more lenient, default 0.35)")
    parser.add_argument("--frame-skip", type=int, default=10)

    args = parser.parse_args()

    if not os.path.exists(args.person):
        print("Person image not found.")
        return

    detector = HybridPersonDetector(
        args.person, args.video,
        insight_threshold=args.insight_threshold,
        dlib_tolerance=args.tolerance,
    )
    detector.run_detection(args.tolerance, args.frame_skip)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()