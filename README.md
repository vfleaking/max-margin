# max-margin

Code for the experiments in the paper "Gradient Descent Maximizes the Margin of Homogeneous Neural Networks" (https://arxiv.org/abs/1906.05890)

# How to reproduce the experiments

The experiment results of the paper were produced on Python 3.6.8, Tensorflow 1.13.1, GeForce GTX 1080.

To reproduce the experiments on MNIST, use the following commands:

```bash
		./run.sh exper-final-mnist-cnn-loss-based-lr --gpu <gpu-id>         # CNN without no bias, loss-based learning rate scheduler
		./run.sh exper-final-mnist-cnn-const-lr-1 --gpu <gpu-id>            # CNN without bias, lr = 0.01
		./run.sh exper-final-mnist-biased-cnn-loss-based-lr --gpu <gpu-id>  # CNN with bias, loss-based learning rate scheduler
		./run.sh exper-final-mnist-biased-cnn-const-lr-1 --gpu <gpu-id>     # CNN with bias, lr = 0.01
```

In running the experiments, the working directory is in the form of `./logs/<experiment name>/<date>`, and logs and models will be stored in it. After finishing training the neural network, you need to find out the working directory where the model is stored, and use the following commands to perform adversarial attacks:

```bash
		./eval.sh eval-final-mnist-l2 <working directory> --gpu <gpu-id>  # L2 attack on the training set 
		./eval.sh eval-final-mnist-test-l2 <working directory> --gpu <gpu-id>  # L2 attack on the training set
```

Note that this will only attack the model at epoch 10000. If you want to attack models at different epochs, you can change the list of epoch IDs corresponding to the key `eid` in `eval-final-mnist-l2.py` and `eval-final-mnist-test-l2.py`. For example, you can change it to `[100, 1000, 10000]` to attack the models at epoch 100, 1000, 10000 (if they are stored in the working directory).

To reproduce the experiments on CIFAR-10, use the following commands:

```bash
		./run.sh exper-final-cifar-vgg-loss-based-lr --gpu <gpu-id>         # VGGNet-16 without no bias, loss-based learning rate scheduler
		./run.sh exper-final-cifar-vgg-const-lr-1 --gpu <gpu-id>            # VGGNet-16 without bias, lr = 0.1
		./run.sh exper-final-cifar-biased-vgg-loss-based-lr --gpu <gpu-id>  # VGGNet-16 with bias, loss-based learning rate scheduler
		./run.sh exper-final-cifar-biased-vgg-const-lr-1 --gpu <gpu-id>     # VGGNet-16 with bias, lr = 0.1
```

# Code structure

The python files with name `exper-*` or `eval-*` can be seen as configuration files for every experiment or every attack. Running `./run.sh` and `./eval.sh` will start these python files, and they will pass some flag arugments to a `Core` object defined in `core.py`. So the code in `core.py` actually controls how an experiment or an attack should be done.

If you are only interested in the implementation of the loss-based learning rate scheduler, please see the class `LossBasedLRv1` in `models/lr_scheduler.py` and check the `train` method of `Core` to see how it is invoked during training. To understand how our experiments avoid numerical issues, please see the `hp_sparse_softmax_cross_entropy_with_logits_v2` method in `Model` defined in `models/base_model.py`.

Good luck & have fun!

# License
MIT License
