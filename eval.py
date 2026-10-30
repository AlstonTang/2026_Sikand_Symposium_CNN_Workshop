import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
model = keras.models.load_model("models/best_model_90_87.keras")

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 255 * x.permute(1, 2, 0))
])

test_dataset = datasets.ImageFolder(root='chest_xray/test', transform=val_transform)
test = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

model.evaluate(test)
