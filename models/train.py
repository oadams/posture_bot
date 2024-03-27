from pathlib import Path

import torch.utils.data
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import lightning as L


class PostureBot(L.LightningModule):
    def __init__(self, weights):
        super().__init__()
        self.model = resnet50(weights=weights)
        # Replace the final layer with a binary classifier
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"The model has {num_params} parameters.")
        print(
            f"Holding the model in memory alone (assuming f32) requires {num_params * 4 / 1024**2:.1f} MB."
        )

    def forward(self, x):
        return self.model(x)


"""     def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss """

"""     def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer """


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
        return self.images[idx], 0


weights = ResNet50_Weights.DEFAULT

model = PostureBot(weights)

dataset = PostureDataSet("data/raw", weights.transforms())

x = 1

""" # Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%") """
