#  Self-Pruning Neural Network (PyTorch)

##  Overview

This project implements a **self-pruning neural network** that dynamically removes less important weights during training. Unlike traditional pruning methods applied after training, this approach integrates pruning directly into the learning process using learnable gating parameters.



##  Key Idea

Each weight is associated with a learnable gate:

* Gate values are passed through a sigmoid → range (0, 1)
* Effective weight = weight × gate
* If gate → 0 → weight is pruned



##  Loss Function

The model is trained using:


Total Loss = Classification Loss + λ × Sparsity Loss

* Classification Loss → CrossEntropy
* Sparsity Loss → Mean of gate values (encourages pruning)



## Implementation Details

* Custom **PrunableLinear layer**
* Sigmoid-based gating mechanism
* Normalized sparsity loss for stable training
* Training on CIFAR-10 dataset
* Gate initialization tuned to avoid collapse



##  Results

| Lambda | Test Accuracy | Sparsity (%) |
| ------ | ------------- | ------------ |
| 5e-5   | ~49%          | ~25%         |
| 1e-4   | 48.33%        | 35.46%       |
| 5e-4   | ~42%          | ~55%         |



## Observations

* Lower λ → higher accuracy, lower sparsity
* Higher λ → increased pruning, reduced accuracy
* Balanced λ (~1e-4) achieves optimal trade-off

The average gate value stabilizes around **0.3**, indicating selective suppression of less important weights rather than complete removal.



##  Key Learnings

* Proper initialization is critical to avoid over-pruning
* Loss scaling significantly impacts training dynamics
* Pruning is a gradual process controlled by epochs
* There exists a clear trade-off between efficiency and performance



##  Future Work

* Extend to CNN architectures for better accuracy
* Implement hard threshold-based pruning
* Deploy as API using FastAPI
* Integrate with LLM pipelines and RAG systems



## Author

Arjun Kaushik
B.Tech | AI Enthusiast
