//
//  admm_solvers.cpp
//
//  Created by He Jun on 2014-10-10.
//

#include "grasta.h"
#define PI 3.1415926535897

#define MAX(a,b) (((a) > (b)) ? (a) : (b))

inline mat shrinkage_max(const mat & a)
{
    mat b(a.n_rows,a.n_cols);
    b.zeros();    
    for (int i=0; i<a.n_rows; i++)
    {
        if (a(i,0)>0) b(i,0)=a(i,0);
    }
    return b;
};

inline mat shrinkage(const mat &a, double kappa)
{
    mat y;
    
    // as matlab :y = max(0, a-kappa) - max(0, -a-kappa);
    y= shrinkage_max( a-kappa) - shrinkage_max(-a-kappa);
    
    return y;
};

// [ s, w, y] = ADMM_L1_private( U, v, OPTS)
bool ADMM_L1(const mat &U, 
                 const mat &v, 
                 const ADMM_OPT &OPTs,
                 mat &s, 
                 mat &w, 
                 mat &y
                 )
{

    double rho, mu, TOL,lambda;   
    int MAX_ITER = 100;
    
    TOL = OPTs.TOL;
    MAX_ITER = OPTs.MAX_ITER;
    lambda = OPTs.lambda;    
    rho  = OPTs.rho;
    
    mat  UtU = trans(U) * U;
    mat  regUtU = lambda/rho*eye<mat>(UtU.n_rows, UtU.n_cols) + UtU;
    mat  P, Uw_hat,h;
    
    
    w = zeros<mat>(U.n_cols, 1);
    s = zeros<mat>(U.n_rows, 1);
    y = zeros<mat>(U.n_rows, 1);


    mu = 1.25/norm(v,2);
    
    // main algorithm    
    bool bRet = solve(P, regUtU , trans(U));
    if (!bRet) 
    {
        return false;
    }
    int k;
    for (k=0; k < MAX_ITER; k++)
    {
        // w update
        w =  P * (v -s - y/mu); // 1/(2*lambda +rho) * P * ((2*lambda+rho)*v - (y+rho*s));
        
        // s update
        Uw_hat = U*w;
        s  = shrinkage( v-Uw_hat - y/mu, 1/mu);
        
        // y update
        h = Uw_hat + s - v;
        y  = y + mu*h;
        
        mu = mu * rho;
        
        if (norm(h,2) < TOL)
        {
            return true;
        }
    }
    
 
    return false;
}
