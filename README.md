# Hair[CAM]<a href="http://hair-cam.herokuapp.com"><img src="streamlit-app/imgs/logo.png" alt="hair[cam]" width="30"/>

> Hair type prediction for better hair days

[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://hair-cam.herokuapp.com)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/github/release/Naereen/StrapDown.js.svg)](https://GitHub.com/Naereen/StrapDown.js/releases/)

__What's your hair type, is it straight, wavy, curly or kinky...? How loose are your curls?__

<img src="streamlit-app/imgs/hair_types.png" alt="drawing" height="250"/> <img src="streamlit-app/imgs/hair_types_examples.png" alt="drawing" height="250"/> 

<!-- ## Table of Contents -->

## Project Overview

The global market for hair care products is worth over $80 B with a CAGR of 3.55%. However, navigating this space to find the right products can not only be costly but can also be daunting given the number of factors (e.g. hair type, porosity and hair styling techniques) to consider when formulating a healthy hair regimen. Hair[CAM] aims to make it easy to understand these factors, with current support for hair type recognition. To identify different hair types, Hair[CAM] uses convolutional neural networks trained on a labeled image dataset.

## Getting Started

To run Hair[CAM] locally,

### Clone 

Clone repo to local machine:

`$ git clone https://github.com/reidkr/Hair-CAM.git`

### Install

#### w/ Pip

```bash
$ pip install -r requirements.txt
$ streamlit run hair_cam.py
```

#### \w Docker

```bash
$ ./build-local.sh
$ docker run -d --name hair-cam -p 8501:8501 reidkr/hair-cam:latest
```

## Usage

> Simply upload a photo to the app and Hair[CAM] will determine your hair type

<!-- ![](imgs/hair-cam-screencast.gif) -->

<img src="streamlit-app/imgs/hair-cam-screen.gif" alt="drawing" height="450"/>

#### (Optional)

To run model, independently:

```bash
$ python demo.py
```
<!-- ## Workflow summary -->

## License

[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

- [MIT license](https://opensource.org/licenses/mit-license.php)
- Copyright 2020 © Kemar Reid