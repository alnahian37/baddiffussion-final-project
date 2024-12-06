# BadDiffusion
This Repo contains the Codes and Implementation Details of our Final Project to reproduce the paper "How to Backdoor Diffusion Models?" published at CVPR 2023

Paper link: https://arxiv.org/abs/2212.05400

Original Github Link: https://github.com/IBM/BadDiffusion

# Team Members
Mohaiminul Al Nahian (alnahian37)

Rama Al Attar (ramaattar)

# Contribution
- We have both reproduced examples from the paper and have obtained the values of FID and MSE for Model Trained with Different Triggers
- Rama has added a new trigger to the model, not used in the paper and obtained the FID and MSE for comparison with existing triggers.
- Nahian has trained the model with one trigger, then sampled images from the model with different triggers (not seen during training) to observe the behavior of the target output.
- We have both worked equally on the analysis of the outputs and the report preparation needed for the project.


## Usage
Please follow the steps to verify our implementation

### Install Require Packages and Prepare Essential Data

Please run to following command to create necessary directories

```bash
bash install.sh
```

To install the anaconda environment from the provided `yml` file, run the following command

```bash
conda env create -f env_bad.yml
```

### Prepare Training Dataset

`CIFAR10:` It will be downloaded automatically when main code is run

### Prepare FID-Measuring Dataset

`CIFAR10:` Run the following python script, which will download and save images to the folder ``measure/Cifar10``

```bash
python cifar10ToJPG.py
```

## Run BadDiffusion

In order to backdoor a Diffusion Model pre-trained on CIFAR10 with **Grey Box** trigger and **Hat** target and then measure the FID and MSE, we can use the following command

```bash
python baddiffusion.py --mode train+measure --dataset CIFAR10 --batch 128 --epoch 50 --poison_rate 0.1 --trigger BOX_14 --target HAT --ckpt DDPM-CIFAR10-32 --fclip o -o --gpu 0
```


### Generate Samples

If we want to generate the clean samples and backdoor targets from a trained backdoored model checkpoint, use the following command

```bash
python baddiffusion.py --mode sampling --ckpt res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT --fclip o --gpu 0
```


### Generate Samples with Triggers Not Seen During Training

Unfortunately, there is no direct way to do this with the provided codes. If your model was trained with, for example, ``BOX_14``, but you want to generate samples with a different trigger, then open the trained checkpoint folder (For example, `res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT`), then open the file ``args.json`` and replace `"trigger": "BOX_14"` with any of the following: "BOX_14", "STOP_SIGN_14", "BIG_BOX", "SM_BOX" etc. and then run the code for `Generate Samples`

### Train Model with ``Disability Park Sign`` Trigger

```bash
python baddiffusion.py --mode train+measure --dataset CIFAR10 --batch 128 --epoch 50 --poison_rate 0.1 --trigger TRIGGER_DIS_PARK_SIGN_8 --target SHOE --ckpt DDPM-CIFAR10-32 --fclip o -o --gpu 0
```

Choices: "TRIGGER_DIS_PARK_SIGN_8", "TRIGGER_DIS_PARK_SIGN_14", "TRIGGER_DIS_PARK_SIGN_18"

### Arguments
- ``--mode``: Train or test the model, choice: 'train', 'resume', 'sampling`, 'measure', and 'train+measure'
    - ``train``: Train the model
    - ``resume``: Resume the training
    - ``measure``: Compute the FID and MSE score for the BadDiffusion from the saved checkpoint, the ground truth samples will be saved under the 'measure' folder automatically to compute the FID score.
    - ``train+measure``: Train the model and then compute the FID and MSE score
    - ``sampling``: Generate clean samples and backdoor targets from a saved checkpoint
- ``--dataset``: 'CIFAR10'
- ``--batch``: Training batch size. Note that the batch size must be able to divide 128 for 
the CIFAR10 dataset and 64 for the CelebA-HQ dataset.
- ``--eval_max_batch``: Batch size of sampling, default: 512
- ``--epoch``: Training epoch num, default: 50
- ``--learning_rate``: Learning rate, default for 32 * 32 image: '2e-4', default for larger images: '8e-5'
- ``--poison_rate``: Poison rate
- ``--trigger``: Trigger pattern, default: 'BOX_14', choice: 'BOX_18', 'BOX_14', 'BOX_11', 'BOX_8', 'BOX_4', 'STOP_SIGN_18', 'STOP_SIGN_14', 'STOP_SIGN_11', 'STOP_SIGN_8', 'STOP_SIGN_4', 'GLASSES'
- ``--target``: Target pattern, default: 'CORNER', choice: 'TRIGGER', 'SHIFT', 'CORNER', 'SHOE', 'HAT', 'CAT'
- ``--gpu``: Specify GPU device
- ``--ckpt``: Load the HuggingFace Diffusers pre-trained models or the saved checkpoint, default: 'DDPM-CIFAR10-32'
- ``--fclip``: Force to clip in each step or not during sampling/measure, default: 'o'(without clipping)
- ``--result``: Output file path, default: '.'