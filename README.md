
# DLBox: New Model Training Framework for Protecting Training Data

Sharing training data for deep learning raises critical concerns about data leakage, as third-party AI developers take full control over the data.
Untrusted AI developers may easily leak and abuse the valuable training data by sending it over the network, exporting it through external disk, or even through a backdoored AI model.
However, current model training frameworks do not provide any protection to prevent such training data leakage.

DLBox is a new model training framework that minimizes the attack vectors raised by untrusted AI developers.
While the complete data protection is infeasible, DLBox forces the developers to obtain a benignly trained model only, such that the data leakage through invalid paths are minimized.
DLBox is implemented on PyTorch with AMD SEV-SNP, and we demonstrated that DLBox eliminates large attack vectors by preventing known ML attacks.

DLBox is published at NDSS 2025 [[paper](https://www.ndss-symposium.org/wp-content/uploads/2025-3001-paper.pdf)]

## Setup

`setup.sh` details the setup procedure, but it may be incomplete.

