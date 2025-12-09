import cv2
import numpy as np
from pathlib import Path

from ml_manager import MLManager

def process_video(video_path: str, ml_manager: MLManager, output_path: str = None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing: {video_path}")
    print(f"Resolution: {width}x{height} @ {fps}fps, Total frames: {total_frames}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        action_detections, ball_detection, player_keypoints = ml_manager.detect_all(frame)

        vis_frame = frame.copy()

        if action_detections:
            vis_frame = ml_manager.visualizer.draw_detections(vis_frame, action_detections)

        if ball_detection:
            vis_frame = ml_manager.visualizer.draw_detections(vis_frame, [ball_detection])

        if writer:
            writer.write(vis_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    if writer:
        writer.release()
        print(f"Output saved to: {output_path}")

    print(f"Completed processing {frame_count} frames")


def main():
    video_files = [
        Path("test_clips", "clip_004.mp4")
    ]

    ml_manager = MLManager()

    for video_path in video_files:
        if not Path(video_path).exists():
            print(f"Video not found: {video_path}")
            continue

        output_path = f"output_{Path(video_path).stem}.mp4"
        process_video(video_path, ml_manager, output_path)


if __name__ == "__main__":
    main()
