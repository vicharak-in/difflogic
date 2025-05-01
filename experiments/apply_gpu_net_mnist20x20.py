import torch
import torchvision
import time
import mnist_dataset
from difflogic import PackBitsTensor, LogicLayer, GroupSum


# Parameters
dataset = 'mnist20x20'
batch_size = 128
eid = None
neurons = 800
layers = 6  # Hidden layers (in addition to input)

# Load MNIST-20x20 test set with thresholded inputs
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.round()),
])
test_set = mnist_dataset.MNIST('./data-mnist', train=False, transform=transforms, download=True, remove_border=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,  num_workers=4, pin_memory=True, drop_last=True)

# Define model structure: 1 input + 6 hidden + 1 output
logic_layers = [torch.nn.Flatten(start_dim=1, end_dim=-1)]
logic_layers.append(LogicLayer(in_dim=400, out_dim=neurons, grad_factor=1., connections="unique"))  # input layer
for _ in range(layers-1):  # 6 hidden layers
    logic_layers.append(LogicLayer(in_dim=neurons, out_dim=neurons, grad_factor=1., connections="unique"))
logic_layers.append(GroupSum(10, tau=10))  # output

model = torch.nn.Sequential(*logic_layers)
print(model)
model_path = f'./saved_files/model_{eid}_{dataset}_{neurons}_{layers}.pt'
model.load_state_dict(torch.load(model_path))
model.eval()
model.implementation = 'cuda'
model = model.cuda()

# Inference loop using PackBitsTensor
#correct, total = 0, 0
#start_time = time.time()


# Load a single image from test set
image, label = test_set[0]  # Choose the first image
image = image.flatten().round().bool().unsqueeze(0).to("cuda")  # shape: (1, 400)
label = torch.tensor([label], device="cuda")

# Inference
model.eval()
model.implementation = 'cuda'
model = model.cuda()

with torch.no_grad():
    image_bits = PackBitsTensor(image)
    output = model(image_bits)
    pred = output.argmax(-1)

# Results
correct = (pred == label).float().item()
print(f'Predicted: {pred.item()}, Ground Truth: {label.item()}, Correct: {bool(correct)}')


#with torch.no_grad():
#    for data, labels in test_loader:
#        data = data.flatten(start_dim=1).round().bool().to("cuda")  # boolean input, shape (B, 400)
#        print("data: ", data)
 #       labels = labels.to("cuda")
 #3       print("labels: ", labels)
  #      data_bits = PackBitsTensor(data)
        #print(data_bits)
  #      output = model(data_bits)
  ##      print("output: ", output)

    #    print("output.shape: ", output.shape)
     #   correct += (output.argmax(-1) == labels).float().sum()
      #  print("correct ",correct)
      #  total += labels.size(0)
     #   print("total: ", total)

#end_time = time.time()

# Results
#accuracy = correct / total
#print(f'PackBitsTensor GPU INFERENCE - Accuracy: {accuracy:.4f}')
#print(f'Time taken: {end_time - start_time:.4f} seconds')

