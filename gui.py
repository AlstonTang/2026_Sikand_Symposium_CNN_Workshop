# If you have TensorFlow already you can remove the two lines below.
import os
os.environ["KERAS_BACKEND"] = "torch"

# Setup
# Note that you'll need to change the model path depending on where your model is! (Around Line 58)
import keras
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk

model: keras.Sequential | None = None

# Window
root = tk.Tk()
root.title("Chest X-ray CNN Demonstration")
root.minsize(384, 384)

# Status
status = tk.Label(root, text="Load an image...")
status.pack()

# Image
image = None
raw_image = None

def load_image():
    global image, raw_image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg, *.jpeg")]
    )
    if not file_path:
        status.config(text="Image selection failed.")
    status.config(text="Loading...")
    image = keras.utils.load_img(file_path, target_size=(224, 224))
    raw_image = ImageTk.PhotoImage(image)
    image_label.configure(image=raw_image)
    image = keras.utils.img_to_array(image)
    image = keras.ops.convert_to_tensor(image)
    image = keras.ops.expand_dims(image, axis=0)
    status.config(text="Ready!")

# Model Prediction
result: float | None = None
threshold = 0.5

def get_resultant_text():
    if result is None: return ""
    return f"Predicted probability: {100*result:.2f}% -> {"POSITIVE" if result >= threshold else "NEGATIVE"}"

def run_model():
    global result, model
    if image is None:
        status.config(text="Select an image first.")
        return
    status.config(text="Running...")
    if model is None:
        model = keras.models.load_model("models/best_model_90_87.keras")
    result = model.predict(image)[0].item()
    status.config(text=get_resultant_text())

# Default image
image_label = tk.Label(root, text="Image will show here.")
image_label.pack()

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack()

load_image_button = tk.Button(btn_frame, text="Load Image", command=load_image)
load_image_button.pack(side=tk.LEFT)

run_model_button = tk.Button(btn_frame, text="Run Model", command=run_model)
run_model_button.pack(side=tk.LEFT)

# Confidence
label = tk.Label(root, text=f"Confidence: {threshold}")
label.pack()

def update_threshold(value):
    global threshold
    threshold = float(value)
    label.config(text=f"Confidence: {threshold}")
    if result is None: return
    status.config(text=get_resultant_text())

slider = tk.Scale(root, from_=0, to=1, resolution=0, orient=tk.HORIZONTAL, length=200, showvalue=0, command=update_threshold)
slider.set(0.5)
slider.pack()

root.mainloop()
