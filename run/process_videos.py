import cv2
import numpy as np
from pathlib import Path
from collections import deque

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

    # Frame buffer for game state classification (need 16 frames)
    frame_buffer = deque(maxlen=16)
    game_state = "unknown"
    game_state_confidence = 0.0
    classify_interval = 8  # Classify every 8 frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        action_detections, ball_detection, player_keypoints = ml_manager.detect_all(frame)

        vis_frame = frame.copy()

        # Draw action detections
        if action_detections:
            vis_frame = ml_manager.visualizer.draw_detections(vis_frame, action_detections)

        # Draw ball detection
        if ball_detection:
            vis_frame = ml_manager.visualizer.draw_detections(vis_frame, [ball_detection])

        # Draw player bounding boxes and keypoints
        if player_keypoints:
            for player in player_keypoints:
                # Draw bounding box around player
                if player.bbox:
                    x1, y1, x2, y2 = int(player.bbox.x1), int(player.bbox.y1), int(player.bbox.x2), int(player.bbox.y2)
                    # Draw player bounding box in blue
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Draw confidence
                    label = f"Player: {player.confidence:.2f}"
                    cv2.putText(vis_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Draw keypoints and skeleton
                all_keypoints = player.get_all_keypoints()

                # Define skeleton connections (COCO format)
                skeleton_connections = [
                    ('nose', 'left_eye'), ('nose', 'right_eye'),
                    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
                    ('nose', 'left_shoulder'), ('nose', 'right_shoulder'),
                    ('left_shoulder', 'right_shoulder'),
                    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
                    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
                    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
                    ('left_hip', 'right_hip'),
                    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
                    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
                ]

                # Draw skeleton connections
                for connection in skeleton_connections:
                    kp1 = getattr(player, connection[0], None)
                    kp2 = getattr(player, connection[1], None)

                    if kp1 and kp2 and kp1.confidence > 0.5 and kp2.confidence > 0.5:
                        pt1 = (int(kp1.x), int(kp1.y))
                        pt2 = (int(kp2.x), int(kp2.y))
                        cv2.line(vis_frame, pt1, pt2, (255, 255, 0), 2)

                # Draw keypoints
                for kp in all_keypoints:
                    if kp.confidence > 0.5:
                        cv2.circle(vis_frame, (int(kp.x), int(kp.y)), 4, (0, 255, 255), -1)

        # Add frame to buffer for game state classification
        frame_buffer.append(frame.copy())

        # Classify game state periodically when we have enough frames
        if len(frame_buffer) == 16 and frame_count % classify_interval == 0:
            if ml_manager.game_state_detector is not None:
                try:
                    game_state_result = ml_manager.classify_game_state(list(frame_buffer))
                    game_state = game_state_result.predicted_class
                    game_state_confidence = game_state_result.confidence
                except Exception as e:
                    print(f"Game state classification failed: {e}")

        # Draw game state information
        if game_state != "unknown":
            vis_frame = ml_manager.visualizer.draw_game_state(
                vis_frame,
                game_state,
                confidence=game_state_confidence,
                frame_info=f"Frame: {frame_count}/{total_frames}"
            )

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
