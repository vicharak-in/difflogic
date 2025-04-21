import serial
import torch
import time
from datetime import datetime
import torchvision
import torchvision.transforms as transforms
import numpy as np
import mnist_dataset
import os

SERIAL_PORT='/dev/ttyUSB1'
BAUD_RATE=115200
INPUT_BITS=400
OUTPUT_BITS=50
INPUT_BYTES=INPUT_BITS //8
OUTPUT_BYTES=10
LOG_FILE='uart_log_mnist20x20.txt'

def log_message(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(msg+'\n')

def preprocess_mnist_image(img_tensor):
    """
    Binarize and flatten the image tensor (1x20x20) -> 400 bits -> 50 bytes
    """
    img_tensor = img_tensor.squeeze(0)  # Shape: 20x20
    img_binarized = (img_tensor > 0.5).to(torch.uint8).flatten().tolist()  # List of 0s and 1s
    print("img_binarized: ", img_binarized)
    bitstring = ''.join(map(str, img_binarized))
    byte_array = bytes(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))
    return byte_array

def string2val(string):
    return np.argmax(np.array([int(string[i*5:i*5+5], 2) for i in range(len(string)//5)]))

num_bits = 64

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    ])
test_set = mnist_dataset.MNIST('./data-mnist', train=False, transform=transforms, remove_border=True, download=True)

try:
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=None) as ser:
        image, label = test_set[1]
        input_data = preprocess_mnist_image(image) 
        # === SEND DATA ===
        hex_list = [f"{b:02x}" for b in input_data]
        print(f"Sending Image  (Label: {label}, {INPUT_BITS} bits): {hex_list}")
        ser.write(input_data)
        #log_message(f"\n[{datetime.now()}] Sent Image (Label: {label}): {INPUT_BITS} bits")

        # === RECEIVE DATA ===

        print("Waiting for output...")
        received_data = ser.read(OUTPUT_BYTES)
        print("received_data:", received_data)
        if len(received_data) == OUTPUT_BYTES:
            output_bin = ' '.join(f'{byte:08b}' for byte in received_data)
            print(f"Received ({OUTPUT_BYTES} bytes):\n{output_bin}")
            print("Actual label: ", label)
            print("Inferred Label ", np.argmax(np.array([int(aa) for aa in received_data])))
        else:
            print(f"Timeout or incomplete data. Got {len(received_data)} bytes.")

except serial.SerialException as e:
    print(f"Serial error: {e}")
    log_message(f"Serial error: {e}")

##OUTPUT
#print("output: ", output_bin)
#print("output.shape: ", output_bin.shape)
#predicted_label = output_bin.argmax(-1)[0]

# Display results
#print("Actual label:   ", label)
#print("Predicted label:", predicted_label)
