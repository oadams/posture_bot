import time
from datetime import datetime

import cv2 as cv

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # out.write(frame)
    frame = cv.resize(frame, (640, 360))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv.imwrite(f"data/raw/frame_{timestamp}.png", frame)

    time.sleep(1)

    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord("q"):
        break

# Release everything if job is finished
cap.release()
# out.release()
cv.destroyAllWindows()
