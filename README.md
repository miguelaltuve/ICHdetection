# ICHdetection

Code of the paper Altuve, M., & Perez, A. (2022). Intracerebral Hemorrhage Detection on Computed Tomography Images Using a Residual Neural Network

**Citation of the code**: Altuve, M. (2022) Intracerebral Hemorrhage Detection on Computed Tomography Images Using a Residual Neural Network [Source Code].

## Abstract
Intracerebral hemorrhage (ICH) is a high mortality rate, critical medical injury, produced by the rupture of a blood vessel of the vascular system inside the skull. ICH can lead to paralysis and even death. Therefore, it is considered a clinically dangerous disease and needs to be treated quickly. Thanks to the advancement in machine learning and the computing power of today's microprocessors, deep learning has become an unbelievably valuable tool for detecting diseases, in particular from medical images. In this work, we are interested in differentiating computer tomography (CT) images of healthy brains and ICH using the ResNet-18, a deep residual convolutional neural network. In addition, the gradient-weighted class activation mapping (Grad-CAM) technique was employed to visually explore and understand the network’s decisions. A 10-iteration Monte Carlo cross-validation was used, by splitting the data set into 80\% for training and 20\% for testing, to assure the generalizability of the detector. In a database of 100 CT images of brains with ICH and another 100 without ICH, the detector yielded, on average, an accuracy of 96\%, a specificity of 97.11\%, a sensitivity of 95\% and a precision of 97\%, with an average computing time of 208.15 s to train it (on 160 images) and 2.11 s to test 40 images, all that by visually inspecting the decisions made by the network during the classification task. Although these results are comparable with the state of the art, our detector is simplest and with a lower computational load than those found in the literature. Our detector could assist physicians in their medical decision, in resource optimization and in reducing the time and error in ICH diagnosis.

## MATLAB APP
We have developed an app in MATLAB that can be used to detect ICH on CT images using the trained ResNet-18 network, and where the Grad-Cam is plotted to visualize the network's decisions.

