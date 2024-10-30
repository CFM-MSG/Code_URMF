# Introduction

This repository is the implementation code for our paper entitled _"Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion"_ (CVPR 2024) and its extension version (under review). 


![model framework](figures/framework.png)


# Model Training
Please download the BERT pre-trained weights and put corresponding files into the path _prebert/_.
Moreover, the MVSA-Single dataset is organized as follows: 


```
data
|-- MVSA_Single
|   |-- train.jsonl
|   |-- dev.jsonl
|   |-- test.jsonl
|   |-- labelResultAll.txt
```

Run the following scripts to train `URMF` on the MVSA-Single dataset.

```
python train.py
```

The complete implementation on other datasets will be released after the peer review of our extended manuscript. 

# Acknowledgements
The codes are modified from [QMF](https://github.com/QingyangZhang/QMF/tree/main) and [OGM-GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022/tree/main).  
If you find our code is helpful, please consider cite:
```
@inproceedings{gao2024embracing,
  title={Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion},
  author={Gao, Zixian and Jiang, Xun and Xu, Xing and Shen, Fumin and Li, Yujie and Shen, Heng Tao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26876--26885},
  year={2024}
}

@inproceedings{zhang2023provable,
  title={Provable dynamic fusion for low-quality multimodal data},
  author={Zhang, Qingyang and Wu, Haitao and Zhang, Changqing and Hu, Qinghua and Fu, Huazhu and Zhou, Joey Tianyi and Peng, Xi},
  booktitle={International conference on machine learning},
  pages={41753--41769},
  year={2023},
}

@inproceedings{peng2022balanced,
  title={Balanced multimodal learning via on-the-fly gradient modulation},
  author={Peng, Xiaokang and Wei, Yake and Deng, Andong and Wang, Dong and Hu, Di},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={8238--8247},
  year={2022}
}


```