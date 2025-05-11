# LEAP
This repository contains the PyTorch implementation of the paper

>  Learning and Editing Universal Graph Prompt Tuning via Reinforcement Learning (LEAP)

## Requirements

The following packages are required under `Python 3.9`.

```
pytorch 2.0.1
torch-geometric 2.3.1
torch-cluster 1.6.3
torch-scatter 2.1.2
torch-sparse 0.6.18
torch-spline-conv 1.2.2
rdkit 2022.3.4
scikit-learn 1.2.0
```

## Experiments

### Graph-level Task

- **Datasets**: All datasets are referenced from the paper Strategies for Pre-training Graph Neural Networks and are available at their repository. Please download the chemistry dataset, unzip it while retaining the `dataset` directory, and place it directly under the `graph_level` folder.

- **Pre-training**: We follow the training steps from the paper Strategies for Pre-training Graph Neural Networks and Graph Contrastive Learning with Augmentations to obtain four pre-trained GIN models, including Deep Graph Infomax, Attribute Masking, Context Prediction and Graph Contrastive Learning strategies.

- **Running**:
```python
# Optional Arguments:
-h, --help              show this help message and exit

# Common Settings
--dataset        	Which dataset to use (default: 'BBBP')
--shot_number           Number of shot to train (default: 50 for few-shot, 1 for full-shot)
--skip_epoch            Number of beginning epochs that does not perform evaluation (default: 20)
--total_epochs        	Number of training epochs (default: 50)
--train_loader_size	Training batch size (default: 8).
--eval_loader_size  	Validation and testing batch size (default: 64)
--emb_dim 		Embedding dimensions (default: 300)
--graph_pooling		Graph level pooling (default: 'mean')
--gnn_layers	        Number of GNN message passing layers (default: 5)

# RL Settings
--actor_d_lr            Discrete actor learning rate (default: 5e-4)
--actor_c_lr            Continuous actor learning rate (default: 5e-4)
--critic_lr             Critic learning rate (default: 5e-4)
--policy_decay        	Policy learning rate decay (static or down) (default: static)
--max_z                 The max value of continuous action each step (default: 1.0)
--batch_size            Batch size for one policy training session (default: 8)
--minibatch_size        Batch size for one policy gradient update (default: 128)
--weight_converge       The coefficient of ECR in reward function (default: 1e-4)

# Basic Prompt Settings
--p_num                 Number of basic prompt (default: 10)

# Projection Head Settings
--head_layers           Number of projection head layers (default: 1)
--tasknet_lr            Task network learning rate (default: 1e-3)
--tasknet_epochs        Number of task network training epochs (default: 1)
--tasknet_decay       	Task network learning rate decay (static or down) (default: static)
```

### Node-level Task

- **Datasets**: All datasets are referenced from the paper Strategies for RELIEF: Reinforcement Learning Empowered Graph Feature Prompt Tuning and are available at their repository. Please download all datasets, unzip it while retaining the `dataset` directory, and place it directly under the `node_level` folder.

- **Pre-training**: We use two edge-level pre-training strategies employed in two pioneering work - GPPT and GraphPrompt, respectively. Pre-trained models are provided in the `pretrained_models` directory.

- **Running**:
```python
# Optional Arguments:
-h, --help              show this help message and exit

# Common Settings
--dataset        	Which dataset to use (default: 'Cora')
--shot_number           Number of shot to train (default: 10 for few-shot, 1 for full-shot)
--skip_epoch            Number of beginning epochs that does not perform evaluation (default: 20)
--total_epochs        	Number of training epochs (default: 50)
--train_loader_size	Training batch size (default: 8).
--eval_loader_size  	Validation and testing batch size (default: 64)
--svd_dim               SVD dimension (default: 100)
--emb_dim 		Embedding dimensions (default: 128)
--graph_pooling		Graph level pooling (default: 'mean')
--gnn_layers	        Number of GNN message passing layers (default: 2)

# RL Settings
--actor_d_lr            Discrete actor learning rate (default: 5e-4)
--actor_c_lr            Continuous actor learning rate (default: 5e-4)
--critic_lr             Critic learning rate (default: 5e-4)
--policy_decay        	Policy learning rate decay (static or down) (default: static)
--max_z                 The max value of continuous action each step (default: 1.0)
--batch_size            Batch size for one policy training session (default: 8)
--minibatch_size        Batch size for one policy gradient update (default: 128)
--weight_converge       The coefficient of ECR in reward function (default: 1e-4)

# Basic Prompt Settings
--p_num                 Number of basic prompt (default: 10)

# Projection Head Settings
--head_layers           Number of projection head layers (default: 1)
--tasknet_lr            Task network learning rate (default: 1e-3)
--tasknet_epochs        Number of task network training epochs (default: 1)
--tasknet_decay       	Task network learning rate decay (static or down) (default: static)
```



## Acknowledgement

The structure of this code is based on GPF and RELIEF. Thank for their works.
