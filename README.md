<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
-->

[![Python package](https://github.com/andrew-garrett/oakd-research/actions/workflows/python-package.yml/badge.svg)](https://github.com/andrew-garrett/oakd-research/actions/workflows/python-package.yml)
[![Deploy static content to Pages](https://github.com/andrew-garrett/oakd-research/actions/workflows/static.yml/badge.svg)](https://github.com/andrew-garrett/oakd-research/actions/workflows/static.yml)


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/andrew-garrett/oakd-research">
    <img src="https://avatars.githubusercontent.com/u/69227803?s=200&v=4" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Iris</h3>

  <p align="center">
    <a href="https://github.com/andrew-garrett/oakd-research/tree/main/docs/iris/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/andrew-garrett/oakd-research/issues">Report Bug</a>
    ·
    <a href="https://github.com/andrew-garrett/oakd-research/issues">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <!-- <li><a href="#roadmap">Roadmap</a></li> -->
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <!-- <li><a href="#contact">Contact</a></li> -->
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

Iris is an Open-Source Package for Deep Computer Vision, Specifically geared towards training and deployment onto cloud and edge environments.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Built With

<!--
This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]
-->

* Docker
* Conda
* Pip
* Weights and Biases

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/andrew-garrett/oakd-research.git
   cd oakd-research
   ```
2. (recommended) Create a new anaconda environment.  Iris is compatible with python 3.8, 3.9, 3.10, and 3.11, but is generally used with 3.10.
    ```sh
    conda create -n iris_env python=3.10
    ```
3. Use pip to install dependencies and the package itself
    ```sh
    pip install -r setup_files/requirements.txt && pip install -e .
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

### Customization:

User can control task, dataset, model, and training/inference hyperparameters from iris.json file.

- Task
  - "task": (str) the desired task, one of ["segmentation", "classification", "multilabel"]
- Dataset
  - "dataset_name": (str) a path-like name of the desired dataset, assumed to be relative to a data_root directory
  - "num_classes": (str) the number of classes in the desired dataset.  For segmentation and single-label classification, this is the number of classes.  For multi-label classification, this is the number of labels.
  - "img_sz": (list[int, int, int]) the desired size of images, given as [C, H, W]
  - "batch_size": (int) the desired batch_size for dataloading
- Model
  - "model_arch": (str) any model available in torchvision.models, but must match the task.
  - "pretrained": (bool) indicate whether or not model is pretrained
- Training/Inference
  - Training-only:
    - "criterion": (str) the desired loss function, either a valid one from torch or one of the custom classes in iris.criteria
    - "epochs": (int) the desired maximum number of epochs to train
    - "lr": (float) the desired initial learning rate
    - "momentum": (float) the desired momentum for SGD (beta_1 for Adam-like optimizers)
    - "nesterov": (bool) indicate whether or not to use nesterov-accelerated optimization
    - "weight_decay": (float) the desired weight decay
    - "optimizer": (str) the desired optimization algorithm, one of those available in torch.optim
    - "scheduler": (bool) when false trains with an adaptive learning rate scheduler based on validation loss (ReduceLRonPlateau), and when true trains with a cosine-annealing with linear warmup scheduler.
    - "ignore_index": (int | list[int]) specifying the indices to ignore in loss computation

### Autocapture:
```
(iris_env) C:\oakd-research\iris>python autocapture.py --help
usage: autocapture.py [-h] [--src {image,dataset,webcam}] [--path PATH]

R&D autocapture feature for iris

options:
  -h, --help                    show this help message and exit
  --src {image,dataset,webcam}  AI pipeline runs on this type of data (default: dataset)
  --path PATH                   If specified, the exact source of data (can be a folder or file) (default: None)
```
Note: If using autocapture with an image/video, be sure to specify the relative path of the image/video


### Training:
```
(iris_env) C:\oakd-research\iris>python train.py --help
usage: train.py [-h] [--debug] [--tune] [--data-root DATA_ROOT] [--n-gpus N_GPUS] [--sweep SWEEP]

Training script for iris

options:
  -h, --help                    show this help message and exit
  --debug                       Run in debug mode (verbose logging, profiling, overfitted training) (default: False)
  --tune                        Tune the model for batch size, learning rate (default: False)
  --data-root DATA_ROOT         The root where the dataset folder is located (default: ./datasets/)
  --n-gpus N_GPUS               Number of GPUs, 0 means cpu, 1 means single gpu, >1 means distributed (default: 1)
  --sweep SWEEP                 The path to a wandb sweep config file (default: None)
```

### Prediction
```
(iris_env) C:\oakd-research\iris>python predict.py --help
usage: predict.py [-h] [--model-root MODEL_ROOT] [--model-arch MODEL_ARCH] [--model-id MODEL_ID] [--model-alias {best,latest,v0}] [--data-root DATA_ROOT] [--n-gpus N_GPUS]

Inference script for iris

options:
  -h, --help                         show this help message and exit
  --model-root MODEL_ROOT            The directory for the model checkpoint, expected to be model.ckpt. (default: None)
  --model-arch MODEL_ARCH            The desired model architecture (default: None)
  --model-id MODEL_ID                The wandb run ID for the model checkpoint (default: None)
  --model-alias {best,latest,v0}     The wandb artifact alias for the model checkpoint (default: best)
  --data-root DATA_ROOT              The root where the dataset folder is located (default: ./tmp/)
  --n-gpus N_GPUS                    Number of GPUs, 0 means cpu, 1 means single gpu, >1 means distributed (default: 1)
```

_For more examples, please refer to the [Documentation](./docs/iris/index.html)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
<!--
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/andrew-garrett/oakd-research/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
<!--
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This readme is created from [this template](https://github.com/othneildrew/Best-README-Template/).  This software is architected and written by Andrew Garrett.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/andrew-garrett/oakd-research.svg
[contributors-url]: https://github.com/andrew-garrett/oakd-research/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/andrew-garrett/oakd-research.svg
[forks-url]: https://github.com/andrew-garrett/oakd-research/network/members
[stars-shield]: https://img.shields.io/github/stars/andrew-garrett/oakd-research.svg
[stars-url]: https://github.com/andrew-garrett/oakd-research/stargazers
[issues-shield]: https://img.shields.io/github/issues/andrew-garrett/oakd-research.svg
[issues-url]: https://github.com/andrew-garrett/oakd-research/issues
[license-shield]: https://img.shields.io/github/license/andrew-garrett/oakd-research.svg
[license-url]: https://github.com/andrew-garrett/oakd-research/blob/main/LICENSE

<!-- 
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
-->