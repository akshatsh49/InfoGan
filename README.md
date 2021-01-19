# InfoGan
Pytorch implementation of [InfoGAN: Interpretable Representation Learning byInformation Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) on the MNIST dataset.


## Introduction
Infogan uses an information theoretic approach for unsupervised disentangled representation learning.
It does so by maximizing a mutual information objective between a subset of the generator input and the generated distribution. Generator input has 2 parts : z , the source of incompressible noise and c which we hope will learn different semantic features of the data.
Mutual information is maximized between the generator output distribution : G(z,c) and the variable distribution (c). Using variational maximization we are able to obtain a lower bound over the mutual information. A net is required to model the posterior distribution p(c|x). This net shares most parameters from the discriminator to reduce computational costs. 

## Results 
<table align='center'>
<tr align='center'>
</tr>
<tr>
<td><img src="https://github.com/akshatsh49/InfoGan/blob/master/samples/Epoch_1.png" width=1000" />
<td> <img src="https://github.com/akshatsh49/InfoGan/blob/master/samples/Epoch_25.png" width="1000" />
<td> <img src="https://github.com/akshatsh49/InfoGan/blob/master/samples/Epoch_50.png" width="1000" />
</tr>
</table>


### Discriminator and Generator Loss
<table align='center'>
<tr align='center'>
</tr>
<tr>
<td><img src="https://github.com/akshatsh49/InfoGan/blob/master/track_loss/Discriminator_Loss.png" width =1000 />
<td> <img src="https://github.com/akshatsh49/InfoGan/blob/master/track_loss/Generator_Loss.png" width="1000" />
</tr>
</table>


### L1 loss for infogan and vanilla gan 
<img src="https://github.com/akshatsh49/InfoGan/blob/master/track_loss/L1_Loss.png" width="600" style='vertical-align:middle'/>
Vanilla gan L1 values are capped at (-15) for visibility.This shows that the infogan architecture increases mutual information between c and G(z,c) better than the vanilla gan architecture. The vanilla gan may use the input noise in a highly entangled fashion and still produce sharp samples.

## Latent Space Interpolations
<table align='center'>
<tr align='center'>
  <th> Hand-picked linear space interpolation </th>
  <th> Interpolation GIF </th>
</tr>
<tr>
<td><img src='https://github.com/akshatsh49/InfoGan/blob/master/Space_interpolation/1.png'>
<td> <img src='https://github.com/akshatsh49/InfoGan/blob/master/Space_interpolation/ani.gif' width='400' />
</tr>
</table>

## Factor Interpolations
Right to Left shows interpolation in the single categorical variable.

<img src='https://github.com/akshatsh49/InfoGan/blob/master/Factor_interpolation/fi.png'>

## Additional Comments 
The file [auxiliary.py](https://github.com/akshatsh49/InfoGan/blob/master/auxiliary.py) implements several helper functions for linear/spherical space interpolations , factor interpolations , and functions that make gifs of training over time and interpolations.
