# RFNet-Edit
This is the repository that contains source code for the RFNet-Edit, which is the image editing version of [Reality-and-Fantasy Network](https://leo81005.github.io/Reality-and-Fantasy/).

 RFNet-Edit is a structured, LLM-assisted framework designed for complex text-guided image editing. Unlike standard diffusion models that struggle with multi-object instructions or background consistency, RFNet-Edit explicitly decomposes free-form text commands into executable, instance-level plans. By separating instruction understanding from visual execution, this framework enables precise multi-object editing while strictly preserving the original background. For more details, please refer to the [report](https://github.com/Terence1219/Reality-and-Fantasy-Edit/blob/dev/RFNet_Edit.pdf)

![Intro Image](https://github.com/Terence1219/Reality-and-Fantasy-Edit/blob/dev/example.jpg)

# Acknowledgement
This work builds upon the codebase of [Reality-and-Fantasy](https://github.com/leo81005/Reality-and-Fantasy).

## Installation
```
pip install -r requirements.txt
```

## Stage 1: LLM-Driven Detail Synthesis
Please refer to this [repo](https://github.com/Chelsea97198/RFNet-Edit-Project) for generating LLM response file, and rename the file to "cache_demo_v0.1_gpt-4.json" and move to the /cache 

## Stage 2: Comprehensive Image Inpainting
This repo contains the code for two-stage generation
1. In-Depth Object Generation: Synthesizes target objects within bounding boxes to ensure semantic alignment and geometric flexibility.

2. Seamless Background Integration: Uses latent freezing during the denoising process to blend the new object with the existing environment without altering the surrounding background.

```
python generate.py --prompt-type demo --model gpt-4 --save-suffix "gpt-4" --repeats 1 --frozen_step_ratio 0.5 --regenerate 1 --force_run_ind 0 --run-model lmd --no-scale-boxes-default --template_version v0.1 --sdxl -- inpaint
```
