# SMO-SVM
a python implementation of libsvm

libsvm.py : The SMO version propoesd in this paper:
R.-E. Fan, P.-H. Chen, and C.-J. Lin. Working set selection using second order information for training SVM. Journal of Machine Learning Research 6, 1889-1918, 2005
It is also the idea adopted by 2016 libsvm 2.8+ version

oldsvm.py is a SMO version with working set selected by old method.

It should be noticed that both of them are very slow when it comes to big training set!!