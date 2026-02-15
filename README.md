# AnyIR
### Any Image Restoration via Efficient Spatial-Frequency Degradation Adaptation

The official PyTorch Implementation of AnyIR for All-in-One Image Restoration

#### [Bin Ren <sup>1,2</sup>](https://amazingren.github.io/)$^\star$, [Eduard Zamfir<sup>4</sup>](https://eduardzamfir.github.io), [Zongwei Wu<sup>4</sup>](https://sites.google.com/view/zwwu/accueil), [Yawei Li<sup>4</sup>](https://yaweili.bitbucket.io/)$^\dagger$, [Yidi Li<sup>3</sup>](https://liyidi.github.io/), [Danda Pani Paudel<sup>3</sup>](https://people.ee.ethz.ch/~paudeld/), [Radu Timofte <sup>4</sup>](https://www.informatik.uni-wuerzburg.de/computervision/), [Ming-Hsuan Yang <sup>7</sup>](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en), [Luc Van Gool <sup>3</sup>](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en), and [Nicu Sebe <sup>2</sup>](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)

$\star$: This work was partially conducted during a visit to INSAIT, $\dagger$: Corresponding author <br>

<sup>1</sup> University of Trento, Italy, <br>
<sup>2</sup> University of Pisa, Italy, <br>
<sup>3</sup> INSAIT Sofia University, "St. Kliment Ohridski", Bulgaria, <br>
<sup>4</sup> University of Würzburg, Germany, <br>
<sup>5</sup> ETH Zürich, Switzerland, <br>
<sup>6</sup> Taiyuan University of Technology, China, <br>
<sup>7</sup> University of California, Merced, USA <br>


[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2407.13372)
<!-- [![project](https://img.shields.io/badge/project-page-brightgreen)](https://eduardzamfir.github.io/daair/) -->

## Latest
- `08/01/2025`: Unfortunately, though this work was recommented `Accept` by the AC from the ACM MM2025, while the PC finally reject this work without reason, so this work is still under review. 
- `07/18/2024`: Repository is created. Our code will be made publicly available upon acceptance. 


## Method
<br>
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Restoring any degraded image efficiently via just one model has become increasingly significant and impactful, especially with the proliferation of mobile devices. Traditional solutions typically involve training dedicated models per degradation, resulting in inefficiency and redundancy. More recent approaches either introduce additional modules to learn visual prompts - significantly increasing the size of the model - or incorporate cross-modal transfer from large language models trained on vast datasets, adding complexity to the system architecture. In contrast, our approach, termed AnyIR, takes a unified path that leverages inherent similarity across various degradations to enable both efficient and comprehensive restoration through a joint embedding mechanism, without scaling up the model or relying on large language models.
Specifically, we examine the sub-latent space of each input, identifying key components and reweighting them first in a gated manner. To fuse intrinsic degradation awareness and contextualized attention, a spatial-frequency parallel fusion strategy is proposed to enhance spatially aware local-global interactions and enrich restoration details from the frequency perspective. Extensive benchmarking in the all-in-one restoration setting confirms AnyIR’s SOTA performance, reducing model complexity by around \textbf{82\%} in parameters and \textbf{85\%} in FLOPs compared to the baseline solution. 
Our code will be available upon acceptance.
</details>


## Installation

### Environments
```
# Step1: Create the virtual environments via micromamba or conda:
micromamba create -n anyir python=3.9 -y
or
conda create -n anyir python=3.9 -y

# Step2: Prepare PyTorch and other libs
pip install -r requirements.txt

# Step3: Set cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8/bin:$PATH

```


### Datasets


### Checkpoints Downloads:


### Visual Results Downloads:


### Training


### Evaluation:
(I). 3-Degradation Setting:

(II). 5-Degradation Setting:

(III). Mix-Degradation Setting:

(IV). Real-World Setting:





## Citation

If you find our work helpful, please consider citing the following paper and/or ⭐ the repo.
```
@misc{ren2025any,
      title={Any Image Restoration via Efficient Spatial-Frequency Degradation Adaptation},
      author={Ren, Bin and Zamfir, Eduard and Wu, Zongwei and Li, Yawei and Li, Yidi and Paudel, Danda Pani and Timofte, Radu and Yang, Ming-Hsuan and Van Gool, Luc and Sebe, Nicu},
      year={2025},
      eprint={2504.14249},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgements

This code is built on [PromptIR](https://github.com/va1shn9v/PromptIR) and [AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet).