# AnyIR
### Any Image Restoration with Efficient Automatic Degradation Adaptation

The official PyTorch Implementation of AnyIR for All-in-One Image Restoration


#### [Bin Ren <sup>1,2</sup>](https://amazingren.github.io/), [Eduard Zamfir<sup>3</sup>](https://eduardzamfir.github.io), [Yawei Li <sup>4</sup>](https://yaweili.bitbucket.io/), [Zongwei Wu<sup>3</sup>](https://sites.google.com/view/zwwu/accueil), [Danda Pani Paudel<sup>4,5</sup>](https://people.ee.ethz.ch/~paudeld/), [Radu Timofte <sup>3</sup>](https://www.informatik.uni-wuerzburg.de/computervision/), [Nicu Sebe <sup>2</sup>](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en), and [Luc Van Gool <sup>4,5</sup>](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en)

<sup>1</sup> University of Pisa, Italy, <br>
<sup>2</sup> University of Trento, Italy, <br>
<sup>3</sup> University of Würzburg, Germany, <br>
<sup>4</sup> ETH Zürich, Switzerland, <br>
<sup>5</sup> INSAIT Sofia University, Bulgaria, <br>

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2407.13372)
<!-- [![project](https://img.shields.io/badge/project-page-brightgreen)](https://eduardzamfir.github.io/daair/) -->


## Latest
- `07/18/2024`: Repository is created. Our code will be made publicly available upon acceptance. 


## Method
<br>
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Reconstructing missing details from degraded low-quality inputs poses a significant challenge. Recent progress in image restoration has demonstrated the efficacy of learning large models capable of addressing various degradations simultaneously. 
With the emergence of mobile devices, there is a growing demand for an efficient model to restore any degraded image for better perceptual quality. However, existing models often require specific learning modules tailored for each degradation, resulting in complex architectures and high computation costs. Different from previous work, in this paper, we propose a unified manner to achieve joint embedding by leveraging the inherent similarities across various degradations for efficient and comprehensive restoration. Specifically, we first dig into the sub-latent space of each input to analyze the key components and reweight their contributions in a gated manner. The intrinsic awareness is further integrated with contextualized attention in an X-shaped scheme, maximizing local-global intertwining. Extensive comparison on benchmarking all-in-one restoration setting validates our efficiency and effectiveness, i.e., our network sets new SOTA records while reducing model complexity by approximately \textbf{-82\%} in trainable parameters and \textbf{-85\%} in FLOPs. Our code will be made publicly available upon acceptance.
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

```


### Datasets




## Citation

If you find our work helpful, please consider citing the following paper and/or ⭐ the repo.
```
@misc{ren2024any,
      title={Any Image Restoration with Efficient Automatic Degradation Adaptation}, 
      author={Bin Ren and Eduard Zamfir and Yawei Li and Zongwei Wu and Danda Pani Paudel and Radu Timofte and Nicu Sebe and Luc Van Gool},
      year={2024},
      eprint={2407.13372},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgements

This code is built on [PromptIR](https://github.com/va1shn9v/PromptIR) and [AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet).