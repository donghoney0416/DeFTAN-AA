# DeFTAN-AA: Array geometry agnostic multichannel speech enhancement (INTERSPEECH 2024)

Official page of "DeFT-AA: Array geometry agnostic multichannel speech enhancement", in Proc. Interspeech, 2024"

[![paper](https://img.shields.io/badge/Paper-Pdf-%3CCOLOR%3E.svg)](https://drive.google.com/file/d/1V-At97d8S8PyoD66rHIKIe-IcG5KpiN_/view?usp=drive_link)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://docs.google.com/presentation/d/1hnu4qGVKGEVDufHMNMTuTKUJRM62Ld-5/edit?usp=drive_link&ouid=105609476270770897731&rtpof=true&sd=true)

#### News
- **March 10, 2022:** Training codes are released :fire:
- **June 6, 2022:** Paper accepted at INTERSPEECH 2024 :tada: 
- **August 23, 2021:** Model codes and demos are released! :fire:

> **Abstract:** *We propose an array geometry agnostic multichannel speech enhancement model, which is trained on a single microphone array but can enhance speech in various arrays with different shapes and numbers of microphones. To enable array agnostic processing, the model employs a gated split dense block (GSDB) that separates foreground speech and background noise regardless of array geometry. Furthermore, to design an array-agnostic encoder compatible with different numbers of microphones, we introduce the spatial transformer (ST) that aggregates spatial information by channel-wise self-attention. The proposed space-object cross-attention (SOCA) block alleviates overfitting to a specific array configuration through cross-attention between spatial features and object features. Experimental results demonstrate the efficacy of the proposed model across various array geometries in both simulated and real-world datasets.* 
<hr />

## Network Architecture

<img src = ./figure/Model.png>

## Results and demo clips

Experiments are performed for simulated datasets (spatialized DNS challenge) with various array shapes, number of microphones, and real-world datasets. You can download samples (circular array, rectangular array) if you want to hear demo clips.

<details>
<summary><strong>Simulated dataset</strong> (click to expand) </summary>
<imag src = ./figure/sim_results.PNG>
</details>

<details>
<summary><strong>Real-world experiments</strong> (click to expand) </summary>
<imag src = ./figure/Real_exp.png>
</details>

## Codes
To train the model, you can use the codes below.
Requirements:
```
pip install -r requirements.txt
```
The model code is available in $DeFTAN_AA.py$. You can use this code for training. The training can be done using any training tool (e.g., ESPNet), and other parameters are provided in the paper.

## Citation
If you use DeFTAN-AA, please consider citing:

    @inproceedings{Lee2021DeFTAN-AA,
        title={DeFTAN-AA: Array geometry agnostic multichannel speech enhancement}, 
        author={Dongheon Lee and Jung-Woo Choi},
        booktitle={INTERSPEECH},
        year={2024}
    }
