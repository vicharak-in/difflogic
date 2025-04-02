import torch
import torchvision

import mnist_dataset
from difflogic import CompiledLogicNet

torch.set_num_threads(1)

dataset = 'mnist'
batch_size = 1_000
experiment_id = 520000
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.round()),
])
test_set = mnist_dataset.MNIST('./data-mnist', train=False, transform=transforms, remove_border=True)
print(test_set)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
print(test_loader)
for num_bits in [
    # 8,
    # 16,
    # 32,
    64
]:
    save_lib_path = 'lib/{:08d}_{}.so'.format(experiment_id, num_bits)
    print(save_lib_path)
    compiled_model = CompiledLogicNet.load(save_lib_path, 10, num_bits)
    print(compiled_model)
    correct, total = 0, 0
    for (data, labels) in test_loader:
        data = torch.nn.Flatten()(data).bool().numpy()
        print("data: ", data)#bool
        output = compiled_model.forward(data)
        print("output: ", output)#int32
        print("labels: ", labels)
        correct += (output.argmax(-1) == labels).float().sum()    
        print("correct: ", correct)
        total += output.shape[0]
        print("total: ", total)
    acc3 = correct / total
    print(acc3)
    print('COMPILED MODEL', num_bits)
