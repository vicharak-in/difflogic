# difflogic
Differential Logic Gate Networks


## Experiments

### [1] MNIST(28*28)

- ~84% test accuracy
- 6 layers
- 1000 neurons
- Implementation: Python
- No. of Iterations: 1000
- Batch Size: 64
- Tau: 10
- 1096 LUTs
- Resource Summary:
```
INPUT  PORTS    : 	784,

OUTPUT PORTS    : 	1000,

EFX_LUT4        : 	1096

   1-2  Inputs  : 	472

   3    Inputs  : 	277

   4    Inputs  : 	347
```
- Model architecture:
```
  Sequential(
  (0): Flatten(start_dim=1, end_dim=-1)
  (1): LogicLayer(784, 1000, train)
  (2): LogicLayer(1000, 1000, train)
  (3): LogicLayer(1000, 1000, train)
  (4): LogicLayer(1000, 1000, train)
  (5): LogicLayer(1000, 1000, train)
  (6): LogicLayer(1000, 1000, train)
  (7): GroupSum(k=10, tau=10.0)
)
```

- Training command: `python3 ./experiments/main.py --dataset mnist --implementation python -bs 64 -t 10 -ni 1000 -ef 1_000 -k 1_000 -l 6 --generate_verilog`

### [2] MNIST(20*20)

- ~68% accuracy
- 4 layers
- 200 neurons
- Implementation: Python
- No. of Iterations: 1000
- Batch Size: 128
- Tau: 10
- 218 LUTs
- Resource Summary:
```
INPUT  PORTS    :   400

OUTPUT PORTS    :   200

EFX_LUT4        :   218

   1-2  Inputs  :   81

   3    Inputs  :   66

   4    Inputs  :   71
```
- Model architecture:
```
  Sequential(
  (0): Flatten(start_dim=1, end_dim=-1)
  (1): LogicLayer(400, 200, train)
  (2): LogicLayer(200, 200, train)
  (3): LogicLayer(200, 200, train)
  (4): LogicLayer(200, 200, train)
  (5): GroupSum(k=10, tau=10.0)
)
```
- Training command: ` python3 ./experiments/main.py -bs 128 -t  10 --dataset mnist20x20 -ni 1_000 -ef 1_000 -k  200 -l 4 --generate_verilog`

