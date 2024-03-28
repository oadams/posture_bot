import argparse
from pathlib import Path

from torchvision.io import read_image
from torchvision.models import ResNet50_Weights

from posture_bot.models.train_model import PostureBot

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="lightning_logs/version_22/checkpoints/epoch=9-step=40.ckpt",
)

args = parser.parse_args()

model = PostureBot.load_from_checkpoint(args.model_path)

for image in Path("data/raw/").glob("*.png"):
    preprocess = ResNet50_Weights.DEFAULT.transforms()
    x = preprocess(read_image(str(image))).to(device="mps")
    y_hat = model(x.unsqueeze(0))
    pass

""" cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # out.write(frame)
    frame = cv.resize(frame, (640, 360))

    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #cv.imwrite(f"data/raw/frame_{timestamp}.png", frame)

    cv.imshow("frame", frame)
    if cv.waitKey(1000) == ord("q"):
        break

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

# Release everything if job is finished
cap.release()
# out.release()
cv.destroyAllWindows()
"""
