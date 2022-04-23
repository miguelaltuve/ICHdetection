# ICHdetection

Code of the paper Altuve, M., & Perez, A. (2022). Intracerebral Hemorrhage Detection on Computed Tomography Images Using a Residual Neural Network

**Citation of the code**: Altuve, M. (2022) Intracerebral Hemorrhage Detection on Computed Tomography Images Using a Residual Neural Network [Source Code].

## Abstract
Intracerebral hemorrhage (ICH) is a high mortality rate, critical medical injury, produced by the rupture of a blood vessel of the vascular system inside the skull. ICH can lead to paralysis and even death. Therefore, it is considered a clinically dangerous disease and needs to be treated quickly. Thanks to the advancement in machine learning and the computing power of today's microprocessors, deep learning has become an unbelievably valuable tool for detecting diseases, in particular from medical images. In this work, we are interested in differentiating computer tomography (CT) images of healthy brains and ICH using the ResNet-18, a deep residual convolutional neural network. In addition, the gradient-weighted class activation mapping (Grad-CAM) technique was employed to visually explore and understand the network’s decisions. A 10-iteration Monte Carlo cross-validation was used, by splitting the data set into 80\% for training and 20\% for testing, to assure the generalizability of the detector. In a database of 100 CT images of brains with ICH and another 100 without ICH, the detector yielded, on average, an accuracy of 96\%, a specificity of 97.11\%, a sensitivity of 95\% and a precision of 97\%, with an average computing time of 208.15 s to train it (on 160 images) and 2.11 s to test 40 images, all that by visually inspecting the decisions made by the network during the classification task. Although these results are comparable with the state of the art, our detector is simplest and with a lower computational load than those found in the literature. Our detector could assist physicians in their medical decision, in resource optimization and in reducing the time and error in ICH diagnosis.

## Head CT database
In this work, CT images from Kaggle's Head CT-Hemorrhage database were considered, which contains 100 images of normal brains (without pathology) and 100 images of brains with hemorrhage. 

![TC_HIC (1)](https://user-images.githubusercontent.com/8375111/164104556-9e88d16b-ee19-4b8e-91e5-95d42c09bb2e.jpg)

CT image of a brain with ICH.

## ResNet-18 based ICH detector
The proposed ICH detector is based on a ResNet-18 network, a residual convolutional neural network with eighteen layers deep, which, thanks to a transfer learning strategy~\cite{torrey2010transfer}, identifies the presence of hemorrhages in non-contrast CT images of the brain.

![ResNet](https://user-images.githubusercontent.com/8375111/164103301-93cb7b44-cc76-460f-a63f-4bf15539ff09.png)

ResNet-18 Network Architecture Block Diagram.

ResNet-18 has four residual blocks. The residual block, shown in figure~\ref{fig:residualblock}, is what makes ResNet particularly attractive and efficient. In the residual block, its input is added to the output before the final ReLU activation function.

![residualblock](https://user-images.githubusercontent.com/8375111/164103555-14e87c47-3717-4e59-ba46-a585376826a4.png)

Block diagram of the residual block of a ResNet.

## Gradient-Weighted Class Activation Mapping
Grad-Cam, a generalization of the class activation mapping technique, is a visualization technique used to understand the classification decisions taken by a deep learning network. 

![Screenshot 2022-04-19 162143](https://user-images.githubusercontent.com/8375111/164104092-0e50b7e4-9be7-4b08-b59b-c0e3e87e16ca.jpg)

Examples of ICH CT images with visual explanations of the network using Grad-CAM technique.

## Classification performance over 100 iterations

![errorbar](https://user-images.githubusercontent.com/8375111/164942617-4246dbcf-8953-42d6-96f4-702a7f301acb.png)


## MATLAB APP
We have developed an app in MATLAB that can be used to detect ICH on CT images using the trained ResNet-18 network, and where the Grad-Cam is plotted to visualize the network's decisions. The app (ICHdetection.mlappinstall) can be installed and accessed from the apps gallery in MATLAB. 

![Screenshot 2022-04-19 163646](https://user-images.githubusercontent.com/8375111/164105996-5dd42d34-64fb-474c-bdbf-46fc7bbaa233.jpg)

