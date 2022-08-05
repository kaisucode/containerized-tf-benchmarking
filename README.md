
## About
This benchmarking tool uses `ResNet101`, `ResNet152`, and `VGG16` to monitor time elapsed and accuracy of TensorFlow models on high performance computing clusters. The CIFAR10 dataset is used as the input. Additonal models can be included by modifying the `load_model` function within `main.py`.

## Versions
- Developed with `Python 3.7.8` and `TensorFlow 2.6.0`
- Tested with `Singularity version 3.8.7-1.el7` on `RedHat7`

## How to run locally (this is only recommmended for development)
- `pip install -r requirements.txt` to install libraries
- CPU benchmark example: `python main.py --num_epochs=1 --num_data=50`
- GPU benchmark example: `python main.py --num_epochs=10 --num_data=10000 --is_gpu`
- Multi-GPU (2) benchmark example: `python main.py --num_epochs=3 --num_data=1000 --has_two_gpu`
  - This uses mirrored strategy to execute distributed training

## How to run with Singularity
- Move the desired scripts from `batch_scripts` to the project root
- Run `singularity build nvidia_tf_2_6_0_py3.simg docker://nvcr.io/nvidia/tensorflow:21.12-tf2-py3` to build a container from the NVIDIA NGC catalog
- Duplicate or modify the batch scripts for hyperparameter and hardware configuration tuning, as well as for your HPC cluster
- Run `sbatch scNoGPU.sh` or `sbatch batch_script_name.sh`

