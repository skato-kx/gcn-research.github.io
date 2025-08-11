---
layout: default
title: My_GCN_Summary_and_Implementation.md
---

<!-- MathJax Configuration -->
<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true
    }
  };
</script>
<!-- MathJax Script -->
<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# My Summary and Implementation of Graph Convolutional Networks

As the next step in my exploration of machine learning techniques for structured data, I decided to tackle the foundational paper on Graph Convolutional Networks (GCNs) for semi-supervised learning and to implement it myself. This model is widely regarded as the starting point for modern GNN research, and its elegance lies in combining graph structure with node features in a computationally efficient way.

While I was able to grasp the overall picture of the paper—including the technical background leading up to GCNs and the core concepts of the model itself—I found certain aspects, especially the derivations behind the formulas and the spectral convolution theory, to be more challenging than in my earlier study of PINNs. These sections require additional prerequisite knowledge, which I’m still working to acquire. My plan is to study related papers on these topics, build up a stronger foundation, and then revisit this work with a deeper understanding.

In this post, I break down the GCN’s key ideas, experimental setup, and limitations, while adding my own reflections on how this knowledge could apply to my future research. I’ve also provided a clear, annotated implementation to make the concepts more approachable for beginners. You can view my implementation here: [View Implementation](https://github.com/skato-kx/skato-kx.github.io-pinn-research)

## 🔗 Paper Info

- **Title**: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- **Authors**: Thomas N. Kipf, Max Welling
- **Keywords**: GCN, Graph Neural Network, Semi-Supervised Learning, Node Classification, Graph Representation Learning

## Research Objective and Challenges of Existing Methods

This paper addresses the problem of semi-supervised classification on graph-structured data, where only a small portion of the nodes are labeled. One of the most prominent existing approaches is Laplacian regularization, which smooths label information across the graph.

$$
L = L_0 + \lambda L_{\mathrm{reg}}, \quad L_{\mathrm{reg}} = \sum_{i,j} A_{ij} \| f(X_i) - f(X_j) \|^2 = f(X)^{\top} \Delta f(X)
$$

This loss function is simpler than it looks.

- \(L_0\) is the standard supervised loss: the difference between predicted and true labels for the labeled nodes.
- \(\lambda\) controls the degree of smoothness enforced—larger values make it harder for differences between connected nodes to remain large, thus encouraging smoother predictions.
- Inside \(L_{\mathrm{reg}}\), \(A_{ij}\) is the adjacency matrix entry for nodes \(i\) and \(j\), where 1 means they are connected and 0 means they are not.
- \(f(X_i)\) and \(f(X_j)\) are the model’s predictions for nodes \(i\) and \(j\), respectively. The squared difference between them is multiplied by \(A_{ij}\), so only connected node pairs contribute to the loss.

This means that the loss explicitly penalizes differences in predicted values between connected nodes, pushing them toward being the same. In other words, it encourages predictions to be smooth across edges in the graph.

However, this approach produces only a single global smoothness score and does not provide detailed guidance on how to adjust specific weights throughout the network. To overcome this limitation, the authors introduce the Graph Convolutional Network (GCN) framework.

## Overview of GCNs

Building on the limitations of Laplacian regularization, the authors propose a new framework called Graph Convolutional Networks (GCNs). The core idea behind GCNs is straightforward: update each node’s feature vector by mixing it with the features of its neighbors. This process allows information to propagate between adjacent nodes layer by layer, enabling the model to infer the labels of unlabeled nodes from the features and labels of nearby nodes.

Mathematically, GCNs are derived from a first-order approximation of spectral graph convolution (Graph Fourier transform). Spectral convolution, originally used in image processing, involves an eigen-decomposition step that is computationally expensive. GCNs simplify this by applying a first-order approximation, greatly reducing computational cost. Additionally, a self-loop is added to the adjacency matrix so that each node also passes its own information to the next layer. The normalized adjacency matrix

$$
\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}
$$

ensures numerical stability and improves training efficiency.

In simpler terms, each GCN layer performs the following steps:
- Gather feature vectors from the node itself and its neighbors.
- Normalize and scale them appropriately.
- Multiply by a learnable weight matrix.
- Apply a non-linear activation function and pass the result to the next layer.

These steps are concisely expressed by the following equation:

$$
H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)
$$

Here, \(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}\) is the normalized adjacency matrix that mixes each node’s features with those of its neighbors (including itself). \(H^{(l)}\) represents the input features at layer \(l\), \(W^{(l)}\) is the learnable weight matrix, and \(\sigma\) is a nonlinear activation function. In essence, multiplying by the normalized adjacency matrix aggregates information from the neighborhood, multiplying by \(W^{(l)}\) applies a trainable transformation, and \(\sigma\) introduces non-linearity—producing the updated features \(H^{(l+1)}\) for the next layer.

While GCNs can, in principle, incorporate information from K-hop neighborhoods in a single layer, the authors choose \(K = 1\). This avoids overemphasizing high-degree nodes (nodes connected to many others) and still allows information to propagate further: stacking two layers reaches two-hop neighbors, three layers reach three-hop neighbors, and so on. This makes \(K = 1\) both efficient and effective.

## ⚖️ Cost Function

The GCN is trained using the cross-entropy loss, but applied only to the labeled nodes in the graph. This ensures that the model learns from known labels while still propagating information to unlabeled nodes through the graph structure.

$$
L = - \sum_{l \in \mathcal{Y}_L} \sum_{f=1}^F Y_{lf} \, \ln Z_{lf}
$$

- \(\mathcal{Y}_L\): set of labeled node indices.
- \(F\): number of classes.
- \(Y_{lf}\): one-hot encoded true label (1 if node \(l\) is in class \(f\), else 0).
- \(Z_{lf}\): predicted probability (softmax output) that node \(l\) belongs to class \(f\).

**How it works**:
- For each labeled node \(l\), the loss penalizes the model when its predicted probability \(Z_{lf}\) for the true class \(f\) is low. Since \(Y_{lf}\) is zero for all incorrect classes, only the correct class contributes to the loss. Minimizing this pushes the predicted probability for the correct class closer to 1.

**Why it works for semi-supervised learning**:
- Even though the loss is computed only on labeled nodes, the GCN’s propagation rule mixes each node’s features with those of its neighbors. This means that labeled nodes influence their neighbors’ feature representations during training, indirectly guiding the classification of unlabeled nodes.

## 🧪 Experiments & Results (in paper)

The authors evaluated GCNs on several benchmark datasets for node classification, including both citation networks (Cora, Citeseer, Pubmed) and synthetic/random graphs.

- **The main task**: semi-supervised node classification with only a small fraction of labeled nodes.

**Setup highlights**:
- Default model: 2-layer GCN (as in Section 3.1).
- Hyperparameters tuned on a validation set (dropout rate, L2 regularization for the first layer, hidden units).
- Optimizer: Adam, learning rate 0.01, up to 200 epochs.
- Early stopping: stop if validation loss does not improve for 10 consecutive epochs.
- For random graphs: hidden size = 32, no regularization.
- Same hyperparameters for Citeseer and Pubmed as tuned on Cora.

**Baselines for comparison**:
- Graph Laplacian–based methods: LP, SemiEmb, ManiReg.
- Graph embedding: DeepWalk (skip-gram based).
- Iterative Classification Algorithm (ICA).
- Planetoid (Yang et al., 2016) – best-performing variant.

**Key results**:
- GCN outperformed all baselines on every dataset tested, often by a large margin.
- Especially strong against Laplacian-regularization methods, showing the benefit of propagating both label and feature information.
- Performance gains came without sacrificing efficiency — GCN was competitive in wall-clock time with simpler methods.
- Deeper GCNs (up to 10 layers) were also tested in Appendix B, but performance often degraded past a few layers due to over-smoothing.

**Takeaway**:
- GCNs effectively leverage both graph structure and node features to spread label information in a principled way, achieving state-of-the-art accuracy for semi-supervised classification at the time.

## Discussion and Conclusion

In this study, the proposed GCN demonstrated performance that significantly outperforms existing representative methods in semi-supervised node classification tasks. Traditional Laplacian regularization-based methods are limited by the assumption that edges merely represent node similarity, which constrains their expressive power. In contrast, GCN overcomes these limitations through a simple yet efficient framework that propagates feature information from neighboring nodes at each layer, achieving both high classification accuracy and computational efficiency. Notably, the proposed normalized propagation model outperforms both the first-order approximation model and higher-order models using Chebyshev polynomials, while requiring fewer parameters and computations.

However, the current implementation has certain limitations. In full-batch training, memory usage grows proportionally with the dataset size, which can exceed GPU memory for large graphs. Furthermore, the current framework is naturally designed to handle only undirected graphs, and does not directly support directed edges or edge features.
