import numpy as np

def group_sum_50bit(bits, num_classes=10):
    """
    Perform group sum on 50 bits divided across `num_classes` groups using bitwise ripple-carry addition,
    and return the predicted class label (i.e., the class with the maximum sum).

    :param bits: a list or numpy array of 50 booleans (0 or 1)
    :param num_classes: number of output classes
    :return: predicted class label (int)
    """
    bits = np.array(bits, dtype=bool)
    print("bits: ", bits)
    assert bits.shape[0] == 50, "Expected 50 input bits"
    bits_per_class = 50 // num_classes
    print("bitsperclass: ", bits_per_class)
    log2_bits = 5  # max bits to hold sum


    class_scores = []

    for c in range(num_classes):
        out_temp_o = [False] * log2_bits  # initialize output bits for this class

        for a in range(bits_per_class):
            carry = bits[c * bits_per_class + a]
            for d in reversed(range(log2_bits)):
                out_temp_o_d = out_temp_o[d]
                out_temp_o[d] = carry ^ out_temp_o_d
                carry = carry & out_temp_o_d
        print("out_temp_o: ", out_temp_o)
        # Convert bit list to integer score
        score = sum((1 << i) if bit else 0 for i, bit in enumerate(reversed(out_temp_o)))
        print("score: ", score)
        class_scores.append(score)
    
    print("class_scores: ", class_scores)
    print("np.argmax(class_scores): ", np.argmax(class_scores))
    # Return class with highest score
    predicted_label = int(np.argmax(class_scores))
    return predicted_label

# Fake 50-bit output from logic gate net (e.g., np.bool_ or True/False)
#output_bits = np.random.randint(0, 2, 50).astype(bool)

bitstring = "01000010000101001000000100100001111001000100101000"
#bitstring="00011001010101001001010010011000100100010010001001"
bits = np.array([bool(int(b)) for b in bitstring], dtype=bool)
print("bits: ", bits)
#print(output_bits) 
predicted = group_sum_50bit(bits)
print("Predicted label:", predicted)

