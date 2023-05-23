## Supervised learning benchmarking based on UMTD

Here we document the reference commands for the models benchmark testing on UMTD.

### CNN models

With batch 64, the training of all models can fit into 1 RTX 3090 24G GPU. 

<details>
<summary>CNN methods, 300-epoch training hyper-parameters.</summary>

```
--data_aug = RandomResizedCrop, RandomHorizontalFlip, Normalize
--lr = 0.0001
--optim = SGD-M 
--momentum=0.9
```
</details>

<details>
<summary>MSA methods 200-epoch training hyper-parameters.</summary>

```
--data_aug = RandomResizedCrop, RandomHorizontalFlip, Normalize
--lr = 0.0001
--optim AdamW
--weight-decay=5E-2
```

</details>

<details>
<summary>Hybird, 300-epoch training hyper-parameters.</summary>

```
--data_aug = RandomResizedCrop, RandomHorizontalFlip, Normalize
--lr = 0.0005
--optim AdamW
--weight-decay=5E-2
```

</details>

