import serial
import time
from datetime import datetime

# === CONFIGURE UART SETTINGS ===
SERIAL_PORT = '/dev/ttyUSB1'  # Adjust as needed
BAUD_RATE = 115200
INPUT_BITS = 400
OUTPUT_BITS = 50
INPUT_BYTES = INPUT_BITS // 8
OUTPUT_BYTES = (OUTPUT_BITS + 7) // 8

# === GENERATE INPUT DATA (Customize here) ===
input_data = bytes([0b10101010] * INPUT_BYTES)  # Example pattern

# === SET LOG FILE PATH ===
LOG_FILE = 'uart_log.txt'

def log_message(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

try:
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as ser:
        time.sleep(2)  # Wait for UART ready

        # === SEND DATA ===
        input_bin = ' '.join(f'{byte:08b}' for byte in input_data)
        print(f"📤 Sending ({INPUT_BITS} bits):\n{input_bin}")
        ser.write(input_data)

        # Log input
        log_message(f"\n[{datetime.now()}] Sent {INPUT_BITS} bits:")
        log_message(input_bin)

        # === RECEIVE DATA ===
        print("📥 Waiting for output...")
        #print("output bytes: ", OUTPUT_BYTES)
        received_data = ser.read(OUTPUT_BYTES)
        print("received_data: ", received_data)
        if len(received_data) == OUTPUT_BYTES:
            output_bin = ' '.join(f'{byte:08b}' for byte in received_data)
            print(f"✅ Received ({OUTPUT_BITS} bits):\n{output_bin}")

            # Log output
            log_message(f"Received {OUTPUT_BITS} bits:")
            log_message(output_bin)
        else:
            print(f"❌ Timeout or incomplete data. Got {len(received_data)} bytes.")
            log_message("❌ Timeout or incomplete data received.")

except serial.SerialException as e:
    print(f"Serial error: {e}")
    log_message(f"❌ Serial error: {e}")

