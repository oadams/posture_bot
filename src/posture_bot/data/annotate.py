from pathlib import Path

import cv2

# Get a list of all image files in the data/raw directory
image_files = Path("data/raw/").glob("*.png")  # replace with your image format

for image_file in image_files:
    anno_file = image_file.parent / f"{image_file.stem}.txt"
    if anno_file.exists():
        continue

    # Read and display the image
    image = cv2.imread(str(image_file))
    cv2.imshow("Image", image)

    # Wait for a key press
    key = cv2.waitKey(0)
    if key == ord("q"):
        break

    # Write the pressed key to a text file
    with open(anno_file, "w") as f:
        f.write(chr(key & 0xFF))

    # Close the image window
    cv2.destroyAllWindows()
