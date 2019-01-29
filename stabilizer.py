import cv2
import numpy as np

filter_value = 30

# Create a VideoCapture object and read from input file
vid = 'Vibrated.avi'
cap = cv2.VideoCapture(vid)
# load data


# Define the codec and create VideoWriter object
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('Stabilized.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (frame_width, frame_height))

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def frames_difference(cap):
    ret, prev = cap.read()
    prev_grey = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    prev_to_curr_diff_x = []
    prev_to_curr_diff_y = []

    while 1:
        ret, cur = cap.read()

        if not ret:
            break

        cur_grey = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        prev_corner2 = []
        cur_corner2 = []
        cur_corner = np.array([])

        prev_corner = cv2.goodFeaturesToTrack(prev_grey, 200, 0.01, 30)
        cur_corner, status, err = cv2.calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner)

        for i in range(len(status)):
            if status[i]:
                prev_corner2.append(prev_corner[i])
                cur_corner2.append(cur_corner[i])

        cur_corner2 = np.array(cur_corner2)
        prev_corner2 = np.array(prev_corner2)

        # if len (prev_corner2) and len (cur_corner2):
        transform = cv2.estimateAffinePartial2D(prev_corner2, cur_corner2)
        transform = transform[0]

        if transform is None:
            transform = last_transform

        last_transform = transform

        dx = last_transform[0, 2]
        dy = last_transform[1, 2]

        prev_to_curr_diff_x.append(dx)
        prev_to_curr_diff_y.append(dy)

        prev = cur
        prev_grey = cur_grey

    prev_to_curr_difference = np.empty([len(prev_to_curr_diff_x), 2])
    for z in range(len(prev_to_curr_diff_x)):
        prev_to_curr_difference[z] = [prev_to_curr_diff_x[z], prev_to_curr_diff_y[z]]

    return prev_to_curr_difference


# Write an accumulator function to accumulate frame difference for each frame
def accumulator(frame_diff):
    x = 0
    y = 0
    accumulated = np.empty([len(frame_diff), 2])
    for i in range(len(frame_diff)):
        x += frame_diff[i][0]
        y += frame_diff[i][1]
        accumulated[i] = [x, y]

    return accumulated


# Design a low pass filter (smoother)
def low_filter(accumulated):
    global filter_value
    filtered_data = np.empty([len(accumulated), 2])

    for i in range(len(accumulated)):
        sum_x = 0
        sum_y = 0
        count = 0

        for j in range(-filter_value, filter_value + 1):
            if 0 <= i + j < len(accumulated):
                sum_x += accumulated[i + j][0]
                sum_y += accumulated[i + j][1]

                count += 1

        average_x = sum_x / count
        average_y = sum_y / count

        filtered_data[i] = [average_x, average_y]

    return filtered_data


# Write a function to recover frames
def prev_to_current(frame_diff, filtered_data):
    x = 0
    y = 0
    transformed = np.empty([len(frame_diff), 2])
    for i in range(len(frame_diff)):
        x += frame_diff[i][0]
        y += frame_diff[i][1]

        diff_x = filtered_data[i][0] - x
        diff_y = filtered_data[i][1] - y

        dx = frame_diff[i][0] + diff_x
        dy = frame_diff[i][1] + diff_y

        transformed[i] = [dx, dy]

    return transformed


def transform(frame, param):
    ti = cv2.getRotationMatrix2D((0, 0), 0, 1)

    ti[0, 2] += param[0]
    ti[1, 2] += param[1]

    new_frame = cv2.warpAffine(frame, ti, frame.shape[1:-4:-1])

    return new_frame


num = 0
data = frames_difference(cap)
# data = np.loadtxt('Vibrated2.txt', delimiter=',')

cap2 = cv2.VideoCapture(vid)

# Read until video is completed
while (num < frame_count - 1):

    # Capture frame-by-frame
    ret, frame = cap2.read()

    if ret == True:

        # apply transformation
        data1 = accumulator(data)
        data2 = low_filter(data1)
        data3 = prev_to_current(data, data2)

        frame = transform(frame, data3[num])

        num += 1

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Write the frame into the file 'output.avi'
        out.write(frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
cap2.release()

# Closes all the frames
cv2.destroyAllWindows()
exit()
