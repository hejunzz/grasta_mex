# GRASTA_MEX
A Mex version of GRASTA (Grassmannian Robust Adaptive Subspace Tracking Algorithm) https://sites.google.com/site/hejunzz/grasta

GRASTA is an efficient online algorithm for low rank subspace tracking, which is robust to both highly incomplete information and sparse corruption by outliers. This project provides the C++ source code and its mex interface for Matlab.

The main dependency of our code is Armadillo (http://arma.sourceforge.net/). So you should first download the latest version of Armadillo and install it properly according to the instructions of Armadillo. Then open Matlab and locate into our grast_mex directory, run make_mex.m script which is a simple compile line like this:

mex -O -I/usr/local/include grasta_mex.cpp grasta.cpp admm_solvers.cpp  
(You may change "/usr/local/include" to your path of Armadillo, for example -I./armadillo.4.2.3)

Once you compile the mex file successfully, you can run the demo.m to test the robust subspace recovery problem. 


#References
[1] Jun He, Laura Balzano, and John C.S. Lui. Online robust subspace tracking from partial information. Preprint available at http://arxiv.org/pdf/1109.3827v2., 2011.

[2] Jun He, Laura Balzano, and Arthur Szlam. Incremental gradient on the grassmannian for online foreground and background separation in subsampled video. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2012. 
