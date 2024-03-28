from pathlib import Path

import torch.utils.data
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import lightning as L


class PostureBot(L.LightningModule):
    def __init__(self, weights):
        super().__init__()
        self.model = resnet50(weights=weights)
        # Replace the final layer with a binary classifier
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class PostureDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, preprocess):
        self.root_dir = Path(root_dir)
        self.files = list(self.root_dir.glob("*.png"))
        # Just load all the files into memory and preprocess them. If the dataset
        # gets too big we can change this later to load on the fly.
        self.images = [
            preprocess(read_image(str(file_name))) for file_name in self.files
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 0 currently represents okay posture. We will change this later to load from annotation file.
        return self.images[idx], torch.tensor([0]).float()


weights = ResNet50_Weights.DEFAULT

model = PostureBot(weights)

dataset = PostureDataSet("data/raw", weights.transforms())

train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# train model
trainer = L.Trainer(max_epochs=3)
trainer.fit(model=model, train_dataloaders=train_loader)

x = 1

""" # Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%") """
