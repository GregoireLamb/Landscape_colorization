# Landscape colorization using CNN

This project is developed in the context of the lecture "194.077 Applied Deep Learning" form the TU Wien. 

**Author**: Grégoire de Lambertye   
**Mat.number**: 12202211

## Description

This project aims to use convolutional neural network to do landscape image colorization. 

### Introduction

Image colorization is a popular topic in the field of image processing, and despite the substantial 
amount of effort invested in research, there always seems to be room for improvement.

The paper **Colorful Image Colorization** from Zhang et al [[1]](#references) showed a great improvement in image 
colorization using a clever idea of color categorization and re-balancing classes. It inspired a 
lot Chen et al for the paper **Image Colorization Algorithm Based on Deep Learning** [[2]](#references) where they 
improved the neuron network architecture and used different activation functions. In practise they 
focused on faces colorization what inspired this project with the idea of using deep learning to 
train a colorization cnn specialized in landscapes. We will at first try to implement the CU-net network and train it
on the landscape dataset describe bellow. Then we will try to improve it by using different activation functions or 
convolutional layers.  


### Dataset 

The dataset chosen for this project is the 'Landscape Pictures from Rougetet Arnaud' dataset,
available on Kaggle [here](https://www.kaggle.com/datasets/arnaud58/landscape-pictures). 
This comprehensive dataset comprises 7 folders, totaling 4300 high-quality landscape pictures. 
These folders encompass various landscapes such as general scenery, mountains, deserts, seas, beaches, 
islands, and specific scenes from Japan.

Please note that for computational reasons, there might be a reduction in picture quality and shape. 
This process will be thoroughly documented in the final report.


###  Work-breakdown structure

This project will be held in 6 phases: 
+ Gathering information (4h)
+ Gathering and preparing data (2h)
+ Development and training of the CNN  (20h)
+ Workaround to improve the CNN (12h)
+ Evaluation of the final CNN including a user comparison to Zhang's CNN output (6h)
+ Redaction of a report (6h) 

### Additional information 

**Project type:** *beat the stars:* The aim will 
be to beat the CNN from the Colorful Image Colorization paper 
on landscape colorization.

## References

**Paper 1:** {#paper1} Zhang, R., Isola, P., Efros, A.A. (2016). Colorful Image Colorization. In: Leibe, B., Matas, J., Sebe, N., Welling, M. (eds) Computer Vision – ECCV 2016. ECCV 2016. Lecture Notes in Computer Science(), vol 9907. Springer, Cham. https://doi.org/10.1007/978-3-319-46487-9_40

**Paper 2:** {#paper2} Wang, N.; Chen, G.-D.; Tian, Y. Image Colorization Algorithm Based on Deep Learning. Symmetry 2022, 14, 2295. https://doi.org/10.3390/sym14112295

**Dataset:** Rougetet Arnaud, Landscape Pictures (2020), https://www.kaggle.com/datasets/arnaud58/landscape-pictures







