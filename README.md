# EmoVOCA (WACV 202r)

### EmoVOCA: Speech-Driven Emotional 3D Talking Heads

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]([https://arxiv.org/abs/2403.10942](https://arxiv.org/abs/2403.12886))
[![Generic badge](https://img.shields.io/badge/Project%20Page-F76810)](https://fedenoce.github.io/emovoca/)
[![GitHub Stars](https://img.shields.io/github/stars/miccunifi/scantalk?style=social)](https://github.com/miccunifi/scantalk)


This is the **official repository** of the [**WACV 2025 paper**](https://fedenoce.github.io/emovoca/) "*EmoVOCA: Speech-Driven Emotional 3D Talking Heads*" by Federico Nocentini, Claudio Ferrari, Stefano Berretti.

🔥🔥 **[2024/09/10] Our code is now public available! Feel free to explore, use, and contribute!** 


## Overview

The domain of 3D talking head generation has witnessed significant progress in recent years. A notable challenge in this field consists in blending speech-related motions with expression dynamics, which is primarily caused by the lack of comprehensive 3D datasets that combine diversity in spoken sentences with a variety of facial expressions. Whereas literature works attempted to exploit 2D video data and parametric 3D models as a workaround, these still show limitations when jointly modeling the two motions. In this work, we address this problem from a different perspective, and propose an innovative data-driven technique that we used for creating a synthetic dataset, called EmoVOCA, obtained by combining a collection of inexpressive 3D talking heads and a set of 3D expressive sequences. To demonstrate the advantages of this approach, and the quality of the dataset, we then designed and trained an emotional 3D talking head generator that accepts a 3D face, an audio file, an emotion label, and an intensity value as inputs, and learns to animate the audio-synchronized lip movements with expressive traits of the face. Comprehensive experiments, both quantitative and qualitative, using our data and generator evidence superior ability in synthesizing convincing animations, when compared with the best performing methods in the literature.

![assets/teaser.png](assets/idea.png "idea of the method")
We introduce **EmoVOCA**, a novel approach for generating a synthetic 3D Emotional Talking Heads dataset which leverages
speech tracks, intensity labels, emotion labels, and actor specifications. The proposed dataset can be used to surpass the lack of 3D datasets
of expressive speech, and train more accurate emotional 3D talking head generators as compared to methods relying on 2D data as proxy.

![assets/teaser.png](assets/method.png "Architecture of the method")
Overview of our framework. Two distinct encoders process the talking and expressive 3D head
displacements, separately, while a common decoder is trained to reconstruct them. At inference, talking and emotional heads are
combined by concatenating the encoded latent vectors, and the decoder outputs a combination of their displacements.

## Citation
```bibtex
@inproceedings{nocentini2024emovocaspeechdrivenemotional3d,
    title={EmoVOCA: Speech-Driven Emotional 3D Talking Heads}, 
    author={Federico Nocentini and Claudio Ferrari and Stefano Berretti},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year = {2025},
  }
```

<details>
<summary><h2>ScanTalk Installation Guide</h2></summary> 

This guide provides step-by-step instructions on how to set up the ScanTalk environment and install all necessary dependencies. The codebase has been tested on **Ubuntu 20.04.2 LTS** with **Python 3.8**.

## 1. Setting Up Conda Environment

It is recommended to use a Conda environment for this setup.

1. **Create a Conda Environment**
    ```bash
    conda create -n emovoca python=3.8.18
    ```

2. **Activate the Environment**
    ```bash
    conda activate emovoca
    ```

## 2. Install Mesh Processing Libraries

1. **Clone the MPI-IS Repository**
    ```bash
    git clone https://github.com/MPI-IS/mesh.git
    ```

    ```bash
    cd mesh
    ```

2. **Modify line 7 of the Makefile to avoid error**
    ```
    @pip install --no-deps --config-settings="--boost-location=$$BOOST_INCLUDE_DIRS" --verbose --no-cache-dir .
    ```
3. **Run the MakeFile**
    ```bash
    make all
    ```

## 2. Installing PyTorch and Requirements

Ensure you have the correct version of PyTorch and torchvision. If you need a different CUDA version, please refer to the [official PyTorch website](https://pytorch.org/).

1. **Install PyTorch, torchvision, and torchaudio**
    ```bash
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

2. **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```
---

</details>

<details>
<summary><h2>Dataset Installation Guide</h2></summary> 


For training and testing ScanTalk, we utilized three open-source datasets for 3D Talking Heads: [**vocaset**](https://voca.is.tue.mpg.de/), [**BIWI**](https://paperswithcode.com/dataset/biwi-3d-audiovisual-corpus-of-affective), and [**Multiface**](https://github.com/facebookresearch/multiface). The elaborated and aligned datasets, all standardized to the vocaset format, used for both training and testing ScanTalk, can be found [**here**](https://drive.google.com/drive/folders/1KetNagXa9jcgYwnDUAJxDx5UJMx9yLL2?usp=sharing). After downloading, place the `Dataset` folder in the main directory.

</details>

<details>
<summary><h2>Pretrained Models Installation</h2></summary> 

We are releasing two versions of ScanTalk: one named `scantalk_mse.pth.tar`, trained using Mean Square Error Loss, and another named `scantalk_mse_masked_velocity.pth.tar`, which is trained with a combination of multiple loss functions. Both models are available for download [**here**](https://drive.google.com/drive/folders/1iH4ugUI_JoGiejZj3ENltxSIpUnFY4zl?usp=sharing). After downloading, place the `results` folder within the `src` directory.

</details>
<details>
<summary><h2>ScanTalk Training, Testing and Demo</h2></summary> 

The files `scantalk_train.py` and `scantalk_test.py` are used for training and testing, respectively. `scantalk_test.py` generates a directory containing all the ScanTalk predictions for each test set in the datasets. After obtaining the predictions, `compute_metrics.py` is used to calculate evaluation metrics by comparing the ground truth with the model's predictions.

You can use `demo.py` to run a demo of ScanTalk, animating any 3D face that has been aligned with the training set. Both audio and 3D face for the demo are in the  `src/examples` folder.
</details>

## Authors
* [**Federico Nocentini**](https://scholar.google.com/citations?user=EpQCpoUAAAAJ&hl=en)**\***
* [**Claudio Ferrari**](https://scholar.google.com/citations?user=aael17YAAAAJ&hl=en)
* [**Stefano Berretti**](https://scholar.google.com/citations?user=3GPTAGQAAAAJ&hl=en)

**\*** Equal contribution.

## Acknowledgements

This work is  partially supported by "Partenariato FAIR (Future Artificial Intelligence Research) - PE00000013, CUP J33C22002830006", funded by NextGenerationEU through the Italian MUR within the NRRP, project DL-MIG. 
Additionally, this work was partially funded by the ministerial decree n.352 of the 9th April 2022, NextGenerationEU through the Italian MUR within NRRP, and partially supported by Fédération de Recherche Mathématique des Hauts-de-France (FMHF, FR2037 du CNRS).

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />All material is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** that you've made.
