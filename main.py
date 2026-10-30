import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras.applications import EfficientNetV2B0
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 255 * x.permute(1, 2, 0)),
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomVerticalFlip(0.125),
    # transforms.RandomRotation(180)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 255 * x.permute(1, 2, 0))
])

train_dataset = datasets.ImageFolder(root='chest_xray/train', transform=train_transform)
train = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

test_dataset = datasets.ImageFolder(root='chest_xray/test', transform=val_transform)
test = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)

model = keras.Sequential([
    EfficientNetV2B0(include_top=False, input_shape=(224, 224, 3)),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", "precision", "recall"],
)

checkpoint_filepath = 'best_model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='max',
    save_best_only=True
)

model.fit(
    train,
    validation_data=test,
    epochs=100,
    callbacks=[model_checkpoint_callback]
)
