from Video import *
import cv2
import numpy as np
from tqdm import trange


def task2_1():
    video = Video("videos/video_test.mp4", 500, 250)
    revideo = video.get_result_video("stabilization")
    previous_frame = None
    acc_t = np.zeros(2)
    acc_list = []
    for i in trange(0, video.num_frames):
        success, frame = video.capture.read()
        frame = cv2.resize(frame, (500, 250), interpolation=cv2.INTER_AREA)
        if not success:
            break

        if i == 0:
            frame_stabilized = frame
        else:
            optical_flow = block_matching(previous_frame, frame, 32, 16,'forward', 'eucl')
            average_optical_flow = - np.array(optical_flow.mean(axis=0).mean(axis=0), dtype=np.float32)
            acc_t += average_optical_flow
            H = np.float32([[1, 0, acc_t[0]], [0, 1, acc_t[1]]])
            frame_stabilized = cv2.warpAffine(frame, H, (500, 250))


        previous_frame = frame
        acc_list.append(acc_t)

        revideo.write(frame_stabilized)

def task2_2(SMOOTHING_RADIUS = 50):
    video = Video("videos/video_test.mp4", 500, 250)
    # The larger the more stable the video, but less reactive to sudden panning


    # Read input video
    cap = video.capture

    # Get frame count
    n_frames = video.num_frames

    # Get width and height of video stream
    w = video.width
    h = video.height


    # Set up output video
    out = video.get_result_video("pointmaching")

    # Read first frame
    _, prev = cap.read()

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in trange(230):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

        # Read next frame
        success, curr = cap.read()
        if not success:
            break

            # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix
        m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)  # will only work with OpenCV-3 or less

        # Extract traslation
        dx = m[0, 2]
        dy = m[1, 2]

        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])

        # Store transformation
        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray

        #print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Create variable to store smoothed trajectory
    smoothed_trajectory = smooth(trajectory)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for i in range(n_frames - 2):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # Write the frame to the file
        frame_out = cv2.hconcat([frame, frame_stabilized])

        # If the image is too big, resize it.
        if (frame_out.shape[1] > 1920):
            frame_out = cv2.resize(frame_out, (int(frame_out.shape[1] / 2), int(frame_out.shape[0] / 2)))

        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(10)
        out.write(frame_out)

    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed
def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=100)

    return smoothed_trajectory

def block_matching(img_prev: np.ndarray, img_next: np.ndarray, block_size, search_area, motion_type, metric):

    if motion_type == 'forward':
        reference = img_prev
        target = img_next
    elif motion_type == 'backward':
        reference = img_next
        target = img_prev
    else:
        raise ValueError(f'Unknown motion type: {motion_type}')

    assert (reference.shape == target.shape)
    height, width = reference.shape[:2]
    motion_field = np.zeros((height, width, 2), dtype=float)
    # Get block in the first image:
    for row in range(0, height - block_size, block_size):

        for col in range(0, width - block_size, block_size):

            # block matching
            dist_min = np.inf
            rowb = max(row - search_area, 0)
            colb = max(col - search_area, 0)
            # Get search area and compare the candidate blocks in the image 2 with the previous block in the image 1
            r = 0
            c = 0
            referenceb = reference[row:row + block_size, col:col + block_size]
            targetb = target[rowb: min(row + block_size + search_area, height),
                      colb: min(col + block_size + search_area, width)]
            for row_s in range(targetb.shape[0]-referenceb.shape[0]):

                for col_s in range(targetb.shape[1]-referenceb.shape[1]):


                    # Compute the distance between blocks
                    dist = 0
                    x1 = referenceb
                    x2 = targetb[row_s:row_s+referenceb.shape[0], col_s:col_s+referenceb.shape[1]]
                    if metric == 'eucl':
                        dist = np.sqrt(np.sum((x1 - x2) ** 2))
                    elif metric == 'mse':
                        dist = np.mean((x1 - x2) ** 2)
                    if dist < dist_min:
                        r = row_s
                        c = col_s
                        dist_min = dist

            # Get the flow
            v = r - (row - rowb)
            u = c - (col - colb)
            motion_field[row:row + block_size, col:col + block_size, :] = [u, v]

    return motion_field

if __name__ == '__main__':
    #task2_1()
    task2_2()
