from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

img = read_image("data/raw/frame_20240325_190554.png")

# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"The model has {num_params} parameters.")
print(
    f"Holding the model in memory alone (assuming f32) requires {num_params * 4 / 1024**2:.1f} MB."
)

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
