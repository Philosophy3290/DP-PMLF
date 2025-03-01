# Enhancing Differentially Private Stochastic Gradient Descent via Per-Sample Momentum and Low-Pass Filtering

## Abstract
Differentially Private Stochastic Gradient Descent (DPSGD) is widely used to train deep neural networks with formal privacy guarantees. However, the addition of differential privacy (DP) perturbation and per-sample gradient clipping often degrades model accuracy by introducing both noise and bias. Existing techniques typically address only one of these issues, as reducing DP noise can exacerbate clipping bias and vice versa. In this paper, we propose a novel DPSGD method, DP-PMLF, which integrates per-sample momentum with a low-pass filtering strategy to simultaneously mitigate DP noise and clipping bias. Our approach uses per-sample momentum to smooth gradient estimates prior to clipping, thereby reducing sampling variance, and employs a post-processing low-pass filter to attenuate high-frequency DP noise without consuming additional privacy budget. We provide a theoretical analysis demonstrating an improved convergence rate under rigorous DP guarantees, and our empirical evaluations reveal that DP-PMLF significantly enhances the privacy-utility trade-off compared to several state-of-the-art DPSGD variants.
## Environment
Numpy == 1.21.5 \
Pytorch == 2.2.2 \
Torchvision == 0.17.2 \
Opacus == 1.5.2 \
Timm = 1.0.11 \
fastDP == 2.0.0

## Usage
To use the tool, simply run the following command:
```python DPPMLF.py [OPTIONS]```

| Option | Description |
| --- | --- | 
| `--data` | Dataset Name |
| `--model` | Trained Model | 
| `--bs` | Batch Size |
| `--lr` | Learning Rate | 
| `--epsilon` | Privacy Budget &#949; |
| `--epoch` | Number of Epoch | 
| `--momentum_length` | Per-sample Momentum Length | 
| `--inner_momentum` | Per-sample Momentum Weight | 
| `--coef_file` | Low-pass Filter Coefficients | 


Here's an example of how to use the tool:

```python DPPMLF.py --data MNIST --model cnn5 --bs 1000 --mnbs 200 --lr 0.5 --epoch 25 --epsilon 1 --momentum_length 2 --inner_momentum 0.1 --coef_file ./coefs/a9b1.csv```

## Acknowledgements
This work utilizes portions of the code from **Zhang et al. (2024)**, as described in *"DOPPLER: Differentially Private Optimizers with
Low-pass Filter for Privacy Noise Reduction"*. We appreciate their contribution to the open research community.
