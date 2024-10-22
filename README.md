# Semantic Segmentation of Land Use
The full text together with a short presentation is available in [this release](https://github.com/SchiffFlieger/semantic-segmentation-master-thesis/releases/tag/v255).

### Abstract
The automated identification of emergency landing fields is a complex challenge which requires to process huge quantities of information.
This thesis explores the use of convolutional neural networks to perform a classification of land use through semantic segmentation to support the identification process.
For that, three popular reference architectures of [U-Net](https://arxiv.org/abs/1505.04597), [FC-DenseNet](https://arxiv.org/abs/1611.09326) and [W-Net](https://arxiv.org/abs/1711.08506) are implemented and applied to this challenge.
The experiments show that U-Net and FC-DenseNet achieve adequate segmentation results, while the unsupervised learning process of W-Net fails to learn a proper class differentiation.
In a second step, several spectral vegetation indices are investigated whether they are applicable to further narrow down the number of suitable emergency landing fields.
It is demonstrated that with the given dataset, the indices do not provide any meaningful information.
Overall, this thesis offers a valuable contribution to the improvement of the automatic identification of emergency landing fields.

### Segmetation Results
| ![image](https://github.com/SchiffFlieger/semantic-segmentation-master-thesis/blob/master/latex/images/segmentation_discussion/images/1.png) | ![label](https://github.com/SchiffFlieger/semantic-segmentation-master-thesis/blob/master/latex/images/segmentation_discussion/labels/1.png) | ![unet](https://github.com/SchiffFlieger/semantic-segmentation-master-thesis/blob/master/latex/images/segmentation_discussion/unet/1.png) | ![densenet](https://github.com/SchiffFlieger/semantic-segmentation-master-thesis/blob/master/latex/images/segmentation_discussion/densenet/1.png) |
|:---:|:---:|:---:|:---:|
| Original Image | Ground Truth | U-Net | FC-DenseNet |

### References
1. M. Abadi, A. Agarwal, P. Barham, E. Brevdo, et al. _TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems_. 2015. [[Online]](https://tensorflow.org).
1. P. Arbeláez, M. Maire, C. Fowlkes, J. Malik. _Contour Detection and Hierarchical Image Segmentation_. IEEE TPAMI: vol. 33, no. 5, pp. 898 - 916, 2011. [DOI: 10.1109/TPAMI.2010.161](https://doi.org/10.1109/tpami.2010.161).
1. I. Arganda-Carreras, S. Seung, A. Cardona, J. Schindelin. _ISBI Challenge: Segmentation of neuronal structures in EM stacks_. 2012. [[Online]](http://brainiac2.mit.edu/isbi_challenge/).
1. I. Arganda-Carreras, S. C. Turaga, D. R. Berger, D. Ciresan, A. Giusti. _Crowdsourcing the creation of image segmentation algorithms for connectomics_. Front. Neuroanat: vol. 9, no. 142. 2015. [DOI: 10.3389/fnana.2015.00142](https://doi.org/10.3389/fnana.2015.00142).
1. G. J. Brostow, J. Shotton, J. Fauqueur, R. Cipolla. _Segmentation and Recognition Using Structure from Motion Point Clouds_. In ECCV: pp. 44 - 57, 2008.
1. J. Canny. _A Computational Approach to Edge Detection_. IEEE TPAMI: vol. 8, no. 6, pp. 679 - 698. 1986. [DOI: 10.1109/TPAMI.1986.4767851](https://doi.org/10.1109/TPAMI.1986.4767851). 
1. L. Chen, G. Papandreou, I. Kokkinos, K. Murphy, A. L. Yuille. _DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs_. 2017. [arXiv:1606.00915](https://arxiv.org/abs/1606.00915).
1. F. Chollet. _Xception: Deep Learning with Depthwise Separable Convolutions_. 2017. [arXiv:1610.02357](https://arxiv.org/abs/1610.02357).
1. Y. Cui, M. Jia, T. Lin, Y. Song, S. Belongie. _Class-Balanced Loss Based on Effective Number of Samples_. 2019. [arXiv:1901.05555](https://arxiv.org/abs/1901.05555).
1. J. Deng, W. Dong, R. Socher, L. Li, K. Li, L. Fei-Fei. _Imagenet: A large-scale hierarchical image database_. CVPR: pp. 248 - 255, 2009. [DOI: 10.1109/CVPR.2009.5206848](https://ieeexplore.ieee.org/document/5206848/citations)
1. L. Deng, D. Yu. _Deep Learning: Methods and Applications_. FTSP: vol. 7, pp. 197 - 387, 2014. [DOI: 10.1561/2000000039](http://dx.doi.org/10.1561/2000000039)
1. V. Dumoulin, F. Visin. _A guide to convolution arithmetic for deep learning_. 2018. [arXiv:1603.07285](https://arxiv.org/abs/1603.07285).
1. M. Everingham, L. V. Gool, C. K. I. Williams, J. Winn, A. Zisserman. _The PASCAL Visual Object Classes (VOC) Challenge_. 2012. [[Online]](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
1. M. Everingham, S. M. A. Eslami, L. V. Gool, C. K. I. Williams, J. Winn, A. Zisserman. _The Pascal Visual Object Classes Challenge: A Retrospective_. IJCV: vol. 111, pp. 98 - 136, 2015. [DOI: 10.1007/s11263-014-0733-5](https://doi.org/10.1007/s11263-014-0733-5)
1. A. Géron. _Praxiseinstieg: Machine Learning mit Scikit-Learn und TensorFlow_. O'Reilly dpunkt.verlag, Heidelberg. 2017. Translated by K. Rother. ISBN 978-3-96009-061-8.
1. R. Girshick, J. Donahue, T. Darrell, J. Malik. _Rich feature hierarchies for accurate object detection and semantic segmentation_. 2014. [arXiv:1311.2524](https://arxiv.org/abs/1311.2524).
1.  GISGeography. _What is NDVI (Normalized Difference Vegetation Index)_. 2020. [[Online]](https://gisgeography.com/ndvi-normalized-difference-vegetation-index/).
1. I. Goodfellow, Y. Bengio, A. Courville. _Deep Learning_. MIT Press. 2016. [[Online]](http://www.deeplearningbook.org).
1. FernUniversität in Hagen. _Emergency Landing Field Identification (ELFI)_. 2020. [[Online]](https://www.fernuni-hagen.de/rechnerarchitektur/forschung/fas-elfi.shtml).
1. FernUniversität in Hagen. _Flugassistenzsysteme - Forschungsbereich Notlandeassistenzsysteme_. 2020. [[Online]](https://www.fernuni-hagen.de/rechnerarchitektur/forschung/fas.shtml).
1. H. He, E. A. Garcia. _Learning from Imbalanced Data_. IEEE TKDE: vol. 21, no. 9, pp. 1263 - 1284, 2009. [DOI: 10.1109/TKDE.2008.239](https://doi.org/10.1109/TKDE.2008.239)
1. K. He, G. Gkioxari, P. Dollár, R. Girshick. _Mask R-CNN_. 2018. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870).
1. G. Hinton, N. Srivastava, K. Swersky. _rmsprop: Divide the gradient by a running average of its recent magnitude_. Lecture Notes. 2014. [[Online]](https://www.cs.toronto.edu/~tijmen/csc321/).
1. N. Horning, D. C. Russell. _Global Land Vegetation - An Electronic Textbook_. 2003. [[Online]](http://www.ccpo.odu.edu/SEES/veget/vg_class.htm).
1. G. Huang, Z. Liu, L. v. d. Maaten, K. Q. Weinberger. _Densely Connected Convolutional Networks_. 2018. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993).
1. A. Huete. _A soil-adjusted vegetation index (SAVI)_. Remote Sensing of Environment: vol. 25, no. 3, pp. 295 - 309, 1988. [DOI: 10.1016/0034-4257(88)90106-X](https://doi.org/10.1016/0034-4257(88)90106-X)
1. A. Huete, K. Didan, T. Miura, E. P. Rodriguez, X. Gao, L. G. Ferreira. _Overview of the radiometric and biophysical performance of the MODIS vegetation indices_. Remote Sensing of Environment: vol. 83, no. 1, pp. 195 - 213, 2002. [DOI: 10.1016/S0034-4257(02)00096-2](https://doi.org/10.1016/S0034-4257(02)00096-2)
1. Ministerium des Innern des Landes Nordrhein-Westfalen. _GEOportal.NRW_. [[Online]](https://www.geoportal.nrw/).
1. S. Jégou, M. Drozdzal, D. Vazquez, A. Romero, Y. Bengio. _The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation_. 2017. [arXiv:1611.09326](https://arxiv.org/abs/1611.09326).
1. Bundesamt für Kartographie und Geodäsie. _Digital Basic Landscape Model (Basic-DLM)_. 2020. [[Online]](https://sg.geodatenzentrum.de/web_public/gdz/dokumentation/eng/basis-dlm_eng.pdf).
1. M. Klein, A. Klos, W. Schiffmann. _A Smart Flight Director for Emergency Landings with Dynamical Recalculation of Stable Glide Paths_. AIAA Forum. 2020. [DOI: 10.2514/6.2020-3098](https://doi.org/10.2514/6.2020-3098).
1. A. Klos, J. Lenhardt, M. Klein, W. Schiffmann. _Multi-Modal Image Processing Pipeline for a Reliable Emergency Landing Field Identification_. CEAS GNC. 2019. [[Online]](https://www.researchgate.net/publication/335950509_Multi-Modal_Image_Processing_Pipeline_for_a_Reliable_Emergency_Landing_Field_Identification).
1. A. Klos, M. Rosenbaum, W. Schiffmann. _Ensemble Transfer Learning for Emergency Landing Field Identification on Moderate Resource Heterogeneous Kubernetes Cluster_. ISCNC. 2020. [arxiv:2006.14887](https://arxiv.org/abs/2006.14887).
1. K. Köhler. _Semantic Image Segmentation - Code for Preprocessing, Model Training and Deployment_. 2020. [DOI: 10.5281/zenodo.4065143](https://doi.org/10.5281/zenodo.4065143)
1. Bezirksregierung Köln. _TIM-Online_. [[Online]](https://www.tim-online.nrw.de/tim-online2/).
1. A. Krizhevsky, I. Sutskever, G. Hinton. _ImageNet Classification with Deep Convolutional Neural Networks_. NIPS: vol. 25, 2012. [DOI: 10.1145/3065386](https://www.researchgate.net/deref/http%3A%2F%2Fdx.doi.org%2F10.1145%2F3065386).
1. Y. LeCun, C. Cortes. _MNIST handwritten digit database_. 2010. [[Online]](http://yann.lecun.com/exdb/mnist/).
1. T. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C. L. Zitnick, P. Dollár. _Microsoft COCO: Common Objects in Context_. 2015. [arXiv:1405.0312](https://arxiv.org/abs/1405.0312).
1. Google LLC. _TensorFlow - Case Studies and Mentions_. [[Online]](https://www.tensorflow.org/about/case-studies).
1. Google LLC. _TensorFlow - Serving Models_. [[Online]](https://www.tensorflow.org/tfx/guide/serving).
1. J. Long, E. Shelhamer, T. Darrell. _Fully Convolutional Networks for Semantic Segmentation_. 2015. [arXiv:1411.4038](https://arxiv.org/abs/1411.4038).
1. S. Minaee, Y. Boykov, F. Porikli, A. Plaza, N. Kehtarnavaz, D. Terzopoulos. _Image Segmentation Using Deep Learning: A Survey_. 2020. [arXiv:2001.05566](https://arxiv.org/abs/2001.05566).
1. Y. Nesterov. _A method of solving a convex programming problem with convergence rate O(1/sqrt(k))_. Soviet Mathematics Doklady: vol. 27, pp. 372 - 376, 1983.
1. Bezirksregierung Köln (Abteilung Geobasis NRW). _Topographische Bildinformationen - Luftbildmaterial von Nordrhein-Westfalen_. 2016. [Brochure with Product Information](https://docplayer.org/20323511-Topographische-bildinformationen-luftbildmaterial-von-nordrhein-westfalen.html).
1. C. Nwankpa, W. Ijomah, A. Gachagan, S. Marshall. _Activation Functions: Comparison of trends in Practice and Research for Deep Learning_. 2018. [arXiv:1811.03378](https://arxiv.org/abs/1811.03378).
1. G. Papandreou, L. Chen, K. Murphy, A. L. Yuille. _Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation_. 2015. [arXiv:1502.02734](https://arxiv.org/abs/1502.02734).
1. J. Qi, A. Chehbouni, A. Huete, Y. Kerr, S. Sorooshian. _A Modified Soil Adjusted Vegetation Index_. Remote Sensing of Environment: vol. 48, no. 2, pp. 119 - 126, 1994. [DOI: 10.1016/0034-4257(94)90134-1](https://doi.org/10.1016/0034-4257(94)90134-1)
1. Sh. Ren, K. He, R. Girshick, J. Sun. _Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks_. 2016. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497).
1. R. Rojas. _Neural Networks - A Systematic Introduction_. Springer-Verlag, Berlin. 1996. ISBN 978-3-642-61068-4.
1. O. Ronneberger, P. Fischer, T. Brox. _U-Net: Convolutional Networks for Biomedical Image Segmentation_. 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597).
1. Ch. Szegedy, A. Toshev, D. Erhan. _Deep Neural Networks for Object Detection_. NIPS Proceedings: vol. 26, pp. 2553 - 2561, 2013. [DOI: 10.5555/2999792.2999897](https://dl.acm.org/doi/10.5555/2999792.2999897)
1. E. Tiu. _Metrics to Evaluate your Semantic Segmentation Model_. 2019. [[Online]](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2).
1. Stanford University. _Convolutional Neural Networks for Visual Recognition_. Lecture Notes. 2020. [[Online]](https://cs231n.github.io/).
1. J. Weier, D. Herring. _Measuring Vegetation (NDVI \& EVI)_. 2000. [[Online]](https://earthobservatory.nasa.gov/features/MeasuringVegetation).
1. X. Xia, B. Kulis. _W-Net: A Deep Model for Fully Unsupervised Image Segmentation_. 2017. [arXiv:1711.08506](https://arxiv.org/abs/1711.08506).
1. Y. Xu, R. Goodacre. _On Splitting Training and Validation Set: A Comparative Study of Cross-Validation,  Bootstrap and Systematic Sampling for Estimating the Generalization Performance of Supervised Learning_. Journal of Analysis and Testing: vol. 2, pp. 249 - 262, 2018. [DOI: 10.1007/s41664-018-0068-2](https://doi.org/10.1007/s41664-018-0068-2).
