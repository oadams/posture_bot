import cv2 as cv

cap = cv.VideoCapture(0)

# Define the codec and create VideoWriter object
# fourcc = cv.VideoWriter_fourcc(*'mp4v')
# out = cv.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# fourcc = cv.VideoWriter_fourcc(*'mjpg')
# out = cv.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# fourcc = cv.VideoWriter_fourcc(*'DIVX')
# out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# fourcc = cv.VideoWriter_fourcc(*'X264')
# out = cv.VideoWriter('output.mkv', fourcc, 5.0, (640, 480))

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # out.write(frame)
    cv.imwrite(f"frame_{i}.png", frame)
    i += 1

    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord("q"):
        break

# Release everything if job is finished
cap.release()
# out.release()
cv.destroyAllWindows()
