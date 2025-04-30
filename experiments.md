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

- Training command: ` python3 ./experiments/main.py -bs 128 -t  10 --dataset mnist20x20 -ni 1_000 -ef 1_000 -k  200 -l 4 --generate_verilog`
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

- Training command: ` python3 ./experiments/main.py -bs 256 -t  10 --dataset mnist20x20 -ni 200_000 -ef 1_000 -k  8000 -l 6 --implementation CUDA --save_files`
- 96 % accuracy
- 6 layers
- 8000 neurons
- Implementation- CUDA (GTX 1650)
- No. of Iterations 200_000
- Batch size 256
- Tau 10
- 4480 LUTs, Slack: +43.696 ns, 50 mhz
- Resource Summary
  

## [3] CIFAR-10-3-threshold


- Training command: ` python3 ./experiments/main.py -bs 256 -t  10 --dataset cifar-10-3-threshold -ni 200_000 -ef 1_000 -k  8000 -l 6 --implementation CUDA --save_files`
- 49 % accuracy
- 6 layers
- 8000 neurons
- Implementation- CUDA (GTX 1650)
- No. of Iterations 200_000
- Batch size 256
- Tau 10
- ~80,000 LUTs (combinational logic)
- Resource Summary


- Training command: `python3 main.py -bs 100 -t 100 --dataset cifar-10-3-thresholds -ni 200_000 -ef 1_000 -k 12_000 -l 4 --save_files -eid 524000` 
- 46.12% accuracy
- 4 layers
- 12000 neurons
- Implementation- CUDA (GTX 1650)
- No. of iterations 200_000
- batch size 100
- tau 100
- 197 LUTs (sequential logic)
-  Resource Summary
```
INPUT  PORTS    : 	2
OUTPUT PORTS    : 	1

EFX_ADD         : 	13
EFX_LUT4        : 	176
   1-2  Inputs  : 	42
   3    Inputs  : 	51
   4    Inputs  : 	83
EFX_FF          : 	90
EFX_GBUFCE      : 	1
### ### Resource Summary (end) ### ### ###
```

- Architecture:
```
Sequential(
  (0): Flatten(start_dim=1, end_dim=-1)
  (1): LogicLayer(9216, 12000, train)
  (2): LogicLayer(12000, 12000, train)
  (3): LogicLayer(12000, 12000, train)
  (4): LogicLayer(12000, 12000, train)
  (5): GroupSum(k=10, tau=100.0)
)
```



- Training command: `python main.py  -bs 100 -t 100 --dataset cifar-10-3-thresholds  -ni 200_000 -ef 1_000 -k 128_000 -l 4 --save_files`
- 57% accuracy
- No. of iterations: 200_000
- Neurons 128_000
- Layers 4
- Tau 100
- Implementation- CUDA( GTX 1650)
- LUTs (sequential logic)
- Resource summary:

- Generated [Verilog](https://github.com/vicharak-in/difflogic/blob/main/experiments/saved_files/saved_files/compiled_verilog_64_128000.v), [C](https://github.com/vicharak-in/difflogic/blob/main/experiments/saved_files/saved_files/compiled_c_64bits_128000.c)
- Architecture:
```
Sequential(
  (0): Flatten(start_dim=1, end_dim=-1)
  (1): LogicLayer(9216, 128000, train)
  (2): LogicLayer(128000, 128000, train)
  (3): LogicLayer(128000, 128000, train)
  (4): LogicLayer(128000, 128000, train)
  (5): GroupSum(k=10, tau=100.0)
)
```
- CPU Inference time(on CIFAR-10 test dataset of 10k images) (57% accuracy)
   ```
    5.9131 seconds [for 10k images] 
    ~590 microsec [for 1 image]
   ```
