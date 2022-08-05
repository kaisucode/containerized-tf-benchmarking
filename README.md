
## About
This benchmarking tool uses `ResNet101`, `ResNet152`, and `VGG16` monitor time elapsed and accuracy of TensorFlow on high performance computing clusters. The CIFAR10 dataset is used as the input. Additonal models can be included by modifying the `load_model` function within `main.py`.

## Versions
Developed with `Python 3.7.8` and `TensorFlow 2.6.0`, tested with `Singularity version 3.8.7-1.el7` on `RedHat7`

### How to run locally
`python main.py --num_epochs=1 --num_data=50`
`python main.py --num_epochs=1 --num_data=50 --is_gpu`
`python main.py --num_epochs=1 --num_data=50 --has_two_gpu`

## How to run with Singularity
- Move the desired scripts from `batch_scripts` to the project root
- Run `singularity build nvidia_tf_2_6_0_py3.simg docker://nvcr.io/nvidia/tensorflow:21.12-tf2-py3` to build a container from the NVIDIA NGC catalog
- Duplicate or modify the batch scripts for hyperparameter and hardware configuration tuning, as well as for your HPC cluster
- Run `sbatch scNoGPU.sh` or `sbatch batch_script_name.sh`

