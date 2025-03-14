# AnyCam Project

[**Project Page**](https://fwmb.github.io/anycam)

This is the official implementation for the CVPR 2025 paper:

> **AnyCam: Learning to Recover Camera Poses and Intrinsics from Casual Videos**
>
> [Felix Wimbauer](https://fwmb.github.io/)<sup>1,2,3</sup>, [Weirong Chen](https://chiaki530.github.io/)<sup>1,2,3</sup>, [Dominik Muhle](https://dominikmuhle.github.io/)<sup>1,2</sup>, [Christian Rupprecht](https://chrirupp.github.io/)<sup>3</sup>, and [Daniel Cremers](https://cvg.cit.tum.de/members/cremers)<sup>1,2</sup><br>
> <sup>1</sup>Technical University of Munich, <sup>2</sup>MCML, <sup>3</sup>University of Oxford
> 
> [**CVPR 2025** (arXiv, coming soon)](#)

If you find our work useful, please consider citing our paper:
```
@inproceedings{wimbauer2025anycam,
  title={AnyCam: Learning to Recover Camera Poses and Intrinsics from Casual Videos},
  author={Wimbauer, Felix and Chen, Weirong and Muhle, Dominik and Rupprecht, Christian and Cremers, Daniel},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

**<span style="color:red;">WARNING: This is a preliminary code release with no guarantees. The repository is still Work in Progress (WiP) and a lot of cleaning-up and documenting will happen in the future.</span>**

## Setting Up the Environment

To set up the environment, follow these steps individually or see below:

1. Create a new conda environment with Python 3.11:
    ```sh
    conda create -n anycam python=3.11
    ```

2. Activate the conda environment:
    ```sh
    conda activate anycam
    ```

3. Install pytorch according to your CUDA version:
    ```sh
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    ```

4. Install the corresponding cudatoolkit for compilation:
    ```sh
    conda install -c nvidia cuda-toolkit
    ```

5. Install the required packages from `requirements.txt`:
    ```sh
    pip install -r requirements.txt
    ```

6. Compile `knn` operator (necessary for the UniDepth dependency):
    ```sh
    cd knn
    ./compile.sh
    cd ..
    ```

Combined, this yields the following comand. Building might take a few minutes.
```sh
conda create -n anycam python=3.11 -y && \
conda activate anycam && \
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
conda install -c nvidia cuda-toolkit -y && \
pip install -r requirements.txt && \
cd knn && \
./compile.sh && \
cd ..
```

## Note on dependencies

We directly build KNN for UniDepth in here (code is copied), and also use a slightly customized fork of UniMatch.
Furthermore, we use the minipytorch3d variant by VGGSfM.

## Download pretrained checkpoint

To download pretrained models, you can use the `download_checkpoints.sh` script. Follow these steps:

1. Open a terminal and navigate to the root directory of the repository.

2. Run the `download_checkpoints.sh` script with the desired model name. For example, to download the final `anycam_seq8` model, use the following command:
    ```sh
    ./download_checkpoints.sh anycam_seq8
    ```

This will download and unpack the pretrained model into the `pretrained_models` directory. You can then use the downloaded model for evaluation or further training.

## Training

To train the AnyCam model, run the following command:
```sh
python train_anycam.py -cn anycam_training
```

## Evaluation

To evaluate the AnyCam model, run the following command:
```sh
python anycam/scripts/evaluate_trajectories.py -cn evaluate_trajectories ++model_path=pretrained_models/anycam_seq8
```

You can also enable the `with_rerun` option during evaluation to plot the process to rerun.io:
```sh
python anycam/scripts/evaluate_trajectories.py -cn evaluate_trajectories ++model_path=pretrained_models/anycam_seq8 ++fit_video.ba_refinement.with_rerun=true
```

## Visualization

You can use the Jupyter notebook `anycam/scripts/anycam_4d_plot.ipynb` for visualizing the results.

For more details, refer to the individual scripts and configuration files in the repository.