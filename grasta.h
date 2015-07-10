//
//  grasta.h
// 
//
//  Created by He Jun on 14-10-10.
//

#ifndef GRASTA_H
#define GRASTA_H

#include "mex.h"
#include "matrix.h"
#include <vector>
#include <list>
#include <string.h>
#include <math.h>
#include "armadillo"

using namespace arma;
using namespace std;

#define PRINTF mexPrintf
#define L1_NORM     1
#define L21_NORM    2

struct ADMM_OPT{
    double rho;
    double alpha;
    double TOL;
    double lambda;
    
    int    MAX_ITER;
    ADMM_OPT()
    {
        rho = 1.8; alpha = 0.8;
        MAX_ITER = 10;
        TOL = 1e-10;;
        lambda = 0.0;
    }
};

struct GRASTA_OPT {
    double  OMEGA;  // for sigmod function
    double  FMIN;   // for sigmod function
    double  FMAX;   // for sigmod function
    
    double  rho;      // for ADMM
    int     MIN_ITER; // for ADMM
    int     MAX_ITER; // for ADMM
    double  lambda;   // for ADMM regularizer 
    
    int     DIM;      // for grasta
    int     RANK;     // for grasta
    
    double     MIN_MU;   // for multi-level step-size
    double     MAX_MU;   // for multi-level step-size
    int     MAX_LEVEL;
    double  STEP_SCALE;
    double  MAX_STEPSIZE;
    
    bool    ADAPTIVE; // flag of adaptive / diminishing step-size
    
    int     maxCycles; // for grasta cycling around the dataset
    int     QUIET;
    
    int     convergeLevel;
    
    int     NORM_TYPE;
    
    mat     GT_mat;     // for debug
    double  TOL;        // for debug
    
    GRASTA_OPT()
    {
        rho = 1.8;
        maxCycles = 20;
        QUIET = 0;
        
        OMEGA = 0.1;
        FMIN  = -1;
        FMAX  = 0.5;
        
        MIN_ITER = 10;
        MAX_ITER = 30;
        
        lambda = 0.0;
        
        DIM = 0;
        RANK = 0;
        
        MIN_MU = 1;
        MAX_MU = 15;
        MAX_LEVEL = 20;
        STEP_SCALE = 0.1;
        MAX_STEPSIZE = 1.0;
        
        ADAPTIVE = true;
        
        NORM_TYPE = L1_NORM;
        
        convergeLevel = 20;
        
        TOL = 1e-7;
    }
};

struct STATUS {
    int     init;
    double  last_mu;
    double  step_scale;
    double  grasta_t;
    double  SCALE;
    int     level;
    int     curr_iter;
    mat     last_w;
    mat     last_gamma;
    vector<double>  hist_rel;
    
    STATUS()
    {
        init = 0;
        last_mu = 1.0;
        step_scale = 0.0;
        grasta_t = 1.0;
        SCALE = 1.0;
        level = 0;
        curr_iter = 0;
        last_w = zeros<mat>(1, 1);
        last_gamma = zeros<mat>(1, 1);
    }
};


// [ s, w, y ] = ADMM_L1( U, v, OPTS)
// solving min \| v- Uw \|_1 by ADMM
//
// ---Input
// U: Rank-d matrix spanning the low rank subspace
// v: observed incomplete data vector
// pOPTs: options for ADMM, for example tol for each iteration, MAX_ITER for maximum iteration
//
// ---Output
// s: estimated sparse signal
// w: estimated weights
// y: dual vector
bool ADMM_L1(const mat &U,
        const mat &v,
        const ADMM_OPT &pOPTs,
        mat &s,
        mat &w,
        mat &y
        );


// ---Input
// Uhat: Current estimated subspace, DIM*RANK matrix
// w : the regression weight from ADMM
// dual: the dual from ADMM
// idx: the index of the observed 
// status: current running status of GRASTA
// options: running options of GRASTA
//
// ---Output
//  Unew: updated subspace by taking a gradient step over Grassmannian
//  statusnew: updated running status
// 
// return the estimated step-size for simplely debug
double GRASTA_update(mat &Uhat, 
        struct STATUS &status,
        const mat &w,
        const mat &dual,
        const uvec &idx,
        const struct GRASTA_OPT &options
        );

// D [in]: Incomplete and corrupted data matrix DIM*N size
// Uhat [in,out]: estimated subspace, DIM*RANK matrix
// status [in,out]: running status of GRASTA
// options [in]: running options of GRASTA
// W [out]: The regression weight matrix
// Outlier [out]: D = U*W + E (Outlier)
void GRASTA_training(const mat &D,
        mat &Uhat,
        struct STATUS &status,
        const struct GRASTA_OPT &options,
        mat &W,
        mat &Outlier
        ); 

#endif
