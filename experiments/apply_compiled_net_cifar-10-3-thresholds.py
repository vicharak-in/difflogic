import torch
import torchvision

import time
import cifar_dataset
from difflogic import CompiledLogicNet

torch.set_num_threads(1)


dataset = 'cifar-10-3-thresholds'
batch_size = 100
eid =  0
neurons = 128000
layers = 4

# Define 3-threshold transformation as used during training
threshold_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0)),
])

# Load CIFAR-10 test set with 3-threshold preprocessing
test_set = torchvision.datasets.CIFAR10(
    './data-cifar',
    train=False,
    transform=threshold_transform,
    download=True
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True
)

for num_bits in [64]:
    save_lib_path= './saved_files/saved_files/{}_{}_{}_{}_{}.so'.format(eid, num_bits, dataset, neurons, layers)
    compiled_model = CompiledLogicNet.load(save_lib_path, 10, num_bits)
    correct, total = 0, 0
    start_time=time.time()
    for (data, labels) in test_loader:
        data = torch.nn.Flatten()(data).bool().numpy()
        #print("data: ", data)
        output = compiled_model.forward(data)
        #print("output: ", output)
        correct += (output.argmax(-1) == labels).float().sum()
        #print("correct: ", correct)
        total += output.shape[0]
        #print("total: ", total)

    end_time=time.time()
    inference_time=end_time-start_time
    acc3 = correct / total
    print('COMPILED MODEL', num_bits, acc3)
    print(f'time taken for inference:{inference_time:.4f} seconds')


