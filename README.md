# Spiking Compressed Continual Learning
This is the official Pytorch implementation of **\[ISVLSI 2024\]** - [Spiking Compressed Continual Learning](https://www.youtube.com/watch?v=dQw4w9WgXcQ).

**Spiking Neural Networks (SNNs)** are **bio-inspired**, **power efficient** neuron models that specialize in efficiently processing **time series**. 

We experiment **Continual Learning** strategies on this family of models, an approach relatively unexplored in the Spiking-related research field, to make them adapt to evolving environments without forgetting past knowledge.

On a **progressive class learning** task in which we continously train our pre-trained model on a different language, we demonstrate that our model generalizes to the new classes, achieving **+88.2%** top-1 accuracy on each new class, with only **2.2%** accuracy loss on the older ones, while learning on **compressed data** (compression ratio 1:2, **50% training memory saving**).

## Highlights
All experiments were done on the **[Heidelberg](https://zenkelab.org/datasets/) SHD Dataset**.

<div align="center">
    <img width="50%" src="https://github.com/Dequino/Spiking-Compressed-Continual-Learning/blob/main/figure/Figure_5.png?raw=true"/>
</div>

- **Sample incremental** - up to **92.46%** top-1 total accuracy when learning the same classes spoken by a new speaker. With compression and learning on a more shallow layer, we achieved **88.79%** top-1 accuracy using only **160 KB** of memory.
- **Class incremental** - up to **92.05%** top-1 total accuracy when learning a new class. With compression and learning on a more shallow layer, we achieved **85.53%** top-1 accuracy using only **160 KB** of memory.

<div align="center">
    <img width="50%" src="https://github.com/Dequino/Spiking-Compressed-Continual-Learning/blob/main/figure/Figure_6.png?raw=true"/>
</div>
  
- **Progressive class learning** - pretrain on 10 classes in english, progressively add 10 german classes. Compression ratio 1:2. Final accuracy on the full test set: **78.4%**.

## Features

<div align="center">
    <img width="50%" src="https://github.com/Dequino/Spiking-Compressed-Continual-Learning/blob/main/figure/LatentReplaysSNN.png?raw=true"/>
</div>

1. **Latent Replays (LRs) in Spiking Neural Networks** - On a pretrained network, when adding new data we first **freeze** first N layers and train only the last ones. We **replay past latent activations** (spike sequences) to avoid forgetting. On narrow layers, we have **memory saving** compared to raw rehearsal.

