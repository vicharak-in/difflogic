import serial
import torch
import time
from datetime import datetime
import torchvision
import torchvision.transforms as transforms
import numpy as np
import mnist_dataset
import os

SERIAL_PORT='/dev/ttyUSB0'
BAUD_RATE=115200
INPUT_BITS=400
OUTPUT_BITS=50
INPUT_BYTES=INPUT_BITS //8
OUTPUT_BYTES=(OUTPUT_BITS + 7) //8  #7
LOG_FILE='uart_log_mnist20x20.txt'

def log_message(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(msg+'\n')

def preprocess_mnist_image(img_tensor):
    """
    Binarize and flatten the image tensor (1x20x20) -> 400 bits -> 50 bytes
    """
    img_tensor = img_tensor.squeeze(0)  # Shape: 20x20
    print("img_tensor: ", img_tensor)
    img_binarized = (img_tensor > 0.5).to(torch.uint8).flatten().tolist()  # List of 0s and 1s
    print("img_binarized: ", img_binarized)
    bitstring = ''.join(map(str, img_binarized))
    byte_array = bytes(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))
    return byte_array


# Set number of bits for your compiled model
num_bits = 64

# Load compiled model
#compiled_model = CompiledLogicNet.load(save_lib_path, 10, num_bits)

# Define transforms
transforms = torchvision.transforms.Compose([
    #torchvision.transforms.Resize((20,20)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.round()),
])

# Load the test set
test_set = mnist_dataset.MNIST('./data-mnist', train=False, transform=transforms, remove_border=True)

# Pick a single image and label (e.g., index 0)
#image, label = test_set[0]
#print("test_set[0]: ", test_set[0])
#print("image: ", image)
#print("label: ", label)
# Flatten and convert to boolean numpy
#input_data_numpy= torch.nn.Flatten()(image).bool().numpy()
#print("input_data_numpy: ", input_data_numpy)
#print("input_data.shape: ", input_data_numpy.shape)
# Run inference
#output = compiled_model.forward(input_data_numpy)  # Reshape to (1, 400)

# === START SERIAL COMMUNICATION ===
try:
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as ser:
        time.sleep(2)  # Wait for UART ready

        #for idx in range(5):  # Send first 5 test images
        image, label = test_set[0]
       # input_data_numpy= torch.nn.Flatten()(image).bool().numpy()
        input_data= preprocess_mnist_image(image) 
        print("input_data_numpy: ", input_data)

            # === SEND DATA ===
        input_bin = ' '.join(f'{byte:08b}' for byte in input_data)
        print(f"\n📤 Sending Image  (Label: {label}, {INPUT_BITS} bits):\n{input_bin}")
        ser.write(input_data)

            # === LOG INPUT ===
        log_message(f"\n[{datetime.now()}] Sent Image (Label: {label}): {INPUT_BITS} bits")
        log_message(input_bin)

            # === RECEIVE DATA ===
        print("📥 Waiting for output...")
        received_data = ser.read(OUTPUT_BYTES)
        print("received_data:", received_data)
        if len(received_data) == OUTPUT_BYTES:
            output_bin = ' '.join(f'{byte:08b}' for byte in received_data)
            print(f"✅ Received ({OUTPUT_BITS} bits):\n{output_bin}")
           # print("output_bin.shape: ", output_bin.shape)
           # predicted_label=output_bin.argmax(-1)[0]
           # print("predicted_label: ", predicted_label)
            print("actual label: ", label)
            log_message(f"Received Output for Image :")
            log_message(output_bin)
        else:
            print(f"❌ Timeout or incomplete data. Got {len(received_data)} bytes.")
            log_message("❌ Timeout or incomplete data received.")

except serial.SerialException as e:
    print(f"Serial error: {e}")
    log_message(f"❌ Serial error: {e}")

##OUTPUT
#print("output: ", output_bin)
#print("output.shape: ", output_bin.shape)
#predicted_label = output_bin.argmax(-1)[0]

# Display results
#print("Actual label:   ", label)
#print("Predicted label:", predicted_label)
