import torch
import torchvision
import time
import mnist_dataset
from difflogic import CompiledLogicNet

torch.set_num_threads(1)

dataset = 'mnist20x20'
batch_size = 128
eid=0
neurons= 800
layers =6

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.round()),
])
test_set = mnist_dataset.MNIST('./data-mnist', train=False, transform=transforms, download=True, remove_border=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

for num_bits in [
    # 8,
    # 16,
    # 32,
    64
]:
    save_lib_path = './saved_files/{}_{}_{}_{}_{}.so'.format(eid, num_bits, dataset, neurons, layers)
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

