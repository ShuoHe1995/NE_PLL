# NE_PLL
 Code for "Candidate-aware Selective Disambiguation Based On Normalized Entropy for Instance-dependent Partial-label Learning" (ICCV 2023).

Requirements:

python 3.9.12

torch 1.12.1

torchvision 0.15.2

Dataset:

./"dataset_root"

Candidate label generation:

Model checkpoints saved in ./model_path

Run:

sh run.sh 0 cifar10 10

sh run.sh 1 cifar100 100

sh run.sh 2 fmnist 10

sh run.sh 3 kmnist 10

Parameters varied in run.sh.

Log file saved in ./output_log