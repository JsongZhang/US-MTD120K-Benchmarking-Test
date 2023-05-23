# Capturing Data’s Own Aspect: A Novel Self-Augmented Unsupervised Learning Paradigm for Medical Images Latent Representation

## Supervised learning benchmarking based on UMTD

### Introduction
This is a PyTorch implementation of supervised learning benchmarking based on UMTD. 
We provide an Ultrasound Multi-tasking Dataset (UMTD), which is the largest known open source dataset for the pre-training of ultrasound image modalities. 
You can get it on [10.5281/zenodo.7947246](https://zenodo.org/record/7947246). The UMTD has a [CC BY-NC 4.0 licence](https://creativecommons.org/licenses/by-nc/4.0/deed.zh)

### Benchmark test on UMTD ,Main Results
We have detailed benchmarking of UMTD in terms of supervised learning.
Including convolutional neural network-based, multi-headed self-attention-based , and hybrid.


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Method</th>
<th valign="center">pretrained</th>
<th valign="center">Top1</th>
<th valign="center">epochs</th>
<th valign="center">Type</th>
<!-- TABLE BODY -->
</tr>
<td align="center">Resnet-18</td>
<td align="center">ImageNet-1k</td>
<td align="center">80.30</td>
<td align="center">300</td>
<th align="center">CNN-based</th>
</tr>
<td align="center">Resnet-50</td>
<td align="center">ImageNet-1k</td>
<td align="center">88.61</td>
<td align="center">300</td>
<th align="center">CNN-based</th>
</tr>
<td align="center">Resnet-101</td>
<td align="center">ImageNet-1k</td>
<td align="center">88.94</td>
<td align="center">300</td>
<th align="center">CNN-based</th>
</tr>
<td align="center">RepVgg-A0</td>
<td align="center">ImageNet-1k</td>
<td align="center">84.88</td>
<td align="center">300</td>
<th align="center">CNN-based</th>
</tr>
<td align="center">RepVgg-plus</td>
<td align="center">ImageNet-1k</td>
<td align="center">85.48</td>
<td align="center">300</td>
<th align="center">CNN-based</th>
</tr>
<td align="center">ViT-small</td>
<td align="center">ImageNet-1k</td>
<td align="center">94.45</td>
<td align="center">200</td>
<th align="center">MSA-based</th>
</tr>
<td align="center">ViT-base</td>
<td align="center">ImageNet-1k</td>
<td align="center">95.00</td>
<td align="center">200</td>
<th align="center">MSA-based</th>
</tr>
<td align="center">Swin-tiny</td>
<td align="center">ImageNet-1k</td>
<td align="center">95.55</td>
<td align="center">200</td>
<th align="center">MSA-based</th>
</tr>
<td align="center">Swin-small</td>
<td align="center">ImageNet-1k</td>
<td align="center">95.30</td>
<td align="center">200</td>
<th align="center">MSA-based</th>
</tr>
<td align="center">Swin-base</td>
<td align="center">ImageNet-1k</td>
<td align="center">95.00</td>
<td align="center">200</td>
<th align="center">MSA-based</th>
</tr>
<td align="center">DNN50</td>
<td align="center">None</td>
<td align="center">87.21</td>
<td align="center">300</td>
<th align="center">hybird</th>

</tbody></table>

### Usage: Preparation

Install the dataset UMTD and load the path into the code

The code has been tested with CUDA 11.7/CuDNN 8.5.0, PyTorch 1.13.1 and timm 0.6.13
### Benchmark Reproducibility

Below are the examples for CDOA pre-training. 

#### CNN-based method with single-GPU training, batch 64 on UMTD

Run:
```
python cnn_train.py
```
##### And if you want to use dnn50 or repvgg to reproduce the benchmark

Run:
```
python repvgg_train.py
```

#### MSA-based method with single-GPU training, batch 64 on UMTD

Run swin or vit [used timm]：
```
python swin_train.py
```

### Model Configs

See the commands listed in [CONFIG.md] for specific model configs, including our recommended hyper-parameters and pre-trained reference models.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](https://creativecommons.org/licenses/by-nc/4.0/deed.zh) for details.

### Citation

None
