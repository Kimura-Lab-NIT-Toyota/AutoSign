import os
import sys
import glob
import argparse
import warnings
import contextlib
import pickle
from natsort import natsorted
from multiprocessing import Pool, RLock, current_process, Manager

# Suppress logs and warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# warnings.filterwarnings('ignore')

# Context manager for stderr suppression
@contextlib.contextmanager
def suppress_stderr():
    """
    Temporarily redirect stderr to /dev/null.
    Useful for silencing C++ libraries like Mediapipe and TensorFlow.
    """
    stderr_fileno = sys.stderr.fileno()
    # Save original stderr
    with os.fdopen(os.dup(stderr_fileno), 'w') as old_stderr:
        with open(os.devnull, 'w') as devnull:
            sys.stderr.flush()
            os.dup2(devnull.fileno(), stderr_fileno)
            try:
                yield
            finally:
                # Restore stderr
                sys.stderr.flush()
                os.dup2(old_stderr.fileno(), stderr_fileno)

# Import heavy libraries with suppression
with suppress_stderr():
    import cv2
    import mediapipe as mp
    import numpy as np
    from tqdm import tqdm

# Constants
# MSLR expects 86 points:
# 0-21: Right Hand (21)
# 21-42: Left Hand (21)
# 42-61: Mouth (19)
# 61-86: Body (25)

# MediaPipe Indices
# Body: 0-24 (25 points)
# Mouth: 19 points (Outer lips loop, dropping one to match 19)
# Standard Outer Lips (20 points): 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146
MOUTH_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91] # Dropped 146 to get 19

ERROR_LOG_FILE = 'error_log.txt'

def parse_video_metadata(video_path):
    """Parses video path to extract metadata."""
    parent_dir = os.path.basename(os.path.dirname(video_path))
    filename = os.path.basename(video_path)
    filename_stem = os.path.splitext(filename)[0]

    if '-' in parent_dir:
        parts = parent_dir.split('-', 1)
        video_id = parts[0]
        text = parts[1]
    else:
        video_id = "unknown"
        text = parent_dir

    # Use gloss as text for now
    gloss = text.replace("_", " ")
    key = f"{parent_dir}/{filename_stem}"

    return key, gloss, text

def extract_frame_landmarks(results, width, height):
    """Extracts landmarks in the order: RH, LH, Mouth, Body."""

    # 1. Right Hand (21 points)
    rh_points = []
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            rh_points.append([lm.x * width, lm.y * height, lm.z]) # Pixel coords
    else:
        rh_points = [[0, 0, 0]] * 21

    # 2. Left Hand (21 points)
    lh_points = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            lh_points.append([lm.x * width, lm.y * height, lm.z])
    else:
        lh_points = [[0, 0, 0]] * 21

    # 3. Mouth (19 points)
    mouth_points = []
    if results.face_landmarks:
        for idx in MOUTH_INDICES:
            lm = results.face_landmarks.landmark[idx]
            mouth_points.append([lm.x * width, lm.y * height, lm.z])
    else:
        mouth_points = [[0, 0, 0]] * 19

    # 4. Body (25 points)
    body_points = []
    if results.pose_landmarks:
        for i in range(25): # First 25 points of Pose
            lm = results.pose_landmarks.landmark[i]
            body_points.append([lm.x * width, lm.y * height, lm.z])
    else:
        body_points = [[0, 0, 0]] * 25

    return rh_points, lh_points, mouth_points, body_points

def extract_keypoints(video_path, position=0):
    """Extracts keypoints from a single video."""
    mp_holistic = mp.solutions.holistic
    # Fallback to cv2 just for total_frames metadata
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    filename = os.path.basename(video_path)

    frames_data = []

    # Initialize Mediapipe with stderr suppression
    with suppress_stderr():
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True
        )

    import imageio.v3 as iio

    try:
        # Use tqdm on stdout
        pbar = tqdm(
            total=total_frames,
            desc=f"Processing {filename}",
            leave=False,
            unit="frame",
            position=position,
            file=sys.stdout
        )

        # Iterates natively over RGB frames using pyav
        for image_rgb in iio.imiter(video_path, plugin="pyav"):
            height, width = image_rgb.shape[:2]

            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)

            rh, lh, mouth, body = extract_frame_landmarks(results, width, height)

            # Concatenate: RH, LH, Mouth, Body
            # Shape: (86, 3)
            frame_kps = np.concatenate([rh, lh, mouth, body], axis=0)
            frames_data.append(frame_kps)

            pbar.update(1)
            
        pbar.close()
    except Exception as e:
        with open('error_log.txt', 'a') as f:
            f.write(f"Error in {filename}: {str(e)}\n")
    finally:
        holistic.close()

    if not frames_data:
        return None

    return np.array(frames_data) # (T, 86, 3)

def init_worker(id_queue, lock):
    """Initializer for worker processes."""
    tqdm.set_lock(lock)
    try:
        current_process().worker_id = id_queue.get()
    except Exception:
        current_process().worker_id = 0
        
    # Redirect stderr to /dev/null globally for this worker
    sys.stderr.flush()
    devnull = open(os.devnull, 'w')
    os.dup2(devnull.fileno(), sys.stderr.fileno())

def process_video(video_path):
    """Worker function to process a single video."""
    try:
        worker_id = getattr(current_process(), 'worker_id', 0)
        key, gloss, text = parse_video_metadata(video_path)
        
        keypoints = extract_keypoints(video_path, position=worker_id + 1)
        
        if keypoints is None or len(keypoints) == 0:
            return ('error', video_path, "No keypoints detected")
            
        result = {
            "keypoints": keypoints,
            "label": gloss,
            "text": text
        }
        return ('success', key, result)
        
    except Exception as e:
        return ('error', video_path, str(e))


def main():
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./', help='Root directory containing videos')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of videos to process')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of worker processes')
    parser.add_argument('--output', type=str, default='custom_data.pkl', help='Output pickle file path')
    args = parser.parse_args()

    # Find videos
    video_files = natsorted(
        glob.glob(os.path.join(args.root_dir, '**', '*.avi'), recursive=True)
        + glob.glob(os.path.join(args.root_dir, '**', '*.mp4'), recursive=True)
        + glob.glob(os.path.join(args.root_dir, '**', '*.mov'), recursive=True)
    )
    
    print(f"Found {len(video_files)} videos.")
    if args.limit:
        video_files = video_files[:args.limit]
        print(f"Limiting to {args.limit} videos.")
    
    print(f"Processing with {args.workers} workers...")
    
    # Setup workers
    m = Manager()
    id_queue = m.Queue()
    for i in range(args.workers):
        id_queue.put(i)
        
    # Clear error log
    if os.path.exists(ERROR_LOG_FILE):
        os.remove(ERROR_LOG_FILE)
        
    data = {}
    pbar_global = tqdm(total=len(video_files), desc="Total Progress", position=0, leave=True, file=sys.stdout)
    
    with Pool(processes=args.workers, initializer=init_worker, initargs=(id_queue, RLock())) as pool:
        for status, key_or_path, data_or_msg in pool.imap_unordered(process_video, video_files):
            if status == 'success':
                video_id = key_or_path
                entry = data_or_msg
                data[video_id] = entry
            else:
                with open(ERROR_LOG_FILE, 'a') as f:
                    f.write(f"{key_or_path}: {data_or_msg}\n")
            pbar_global.update(1)
            
    pbar_global.close()
    print("\n" * (args.workers + 1))

    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
