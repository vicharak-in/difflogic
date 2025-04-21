import numpy as np

OUTPUT_BYTES = 7  # ceil(output_bits / 8) = ceil(50 / 8) = 7 

def postprocess(input_tensor):
    bytes_list = []
    decimal_list = []
    i = 0
    j = 5

    for _ in range(OUTPUT_BYTES):
        temp=[]
        temp.extend(['0', '0', '0']) #padding -> 8bits
        temp.extend([input_tensor[bit] for bit in range(i, j)])
        i = j
        j = i + 5

        byte_str = ''.join(temp)    
        bytes_list.append(byte_str)

        decimal = int(byte_str, 2)
        decimal_list.append(decimal)
    
    print("Byte strings:", bytes_list)
    print("Decimal values:", decimal_list)
    predicted_class = np.argmax(decimal_list)
    print("Predicted class:", predicted_class)

# Example input_tensor(string)
input_tensor = "01000010000101001000000100100001111001000100101000"
postprocess(input_tensor)

