//
//  grasta.cpp
//
//  Created by He Jun on 2014-10-10.
//

#include "grasta.h"

#define PI 3.1415926535897

#define MAX(a,b) (((a) > (b)) ? (a) : (b))

double subspace(mat A, mat B)
{
    //Check rank and swap
    mat tmp;
    if (A.n_cols < B.n_cols){
        tmp = A; A = B; B = tmp;
    }
    //Compute the projection the most accurate way, according to [1].
    for (int k=0; k< A.n_cols; k++)
        B = B - A.col(k)*( trans(A.col(k)) * B );

    //Make sure it's magnitude is less than 1.
    double theta = asin(min(1.0,norm(B)));
    
    return theta;         
    
}

inline mat orth(const mat & A)
{
    mat U,V;
    vec s;
    int m,n;
    
    svd_econ(U,s,V,A);
    m = A.n_rows; n = A.n_cols;
    
    double tol = MAX(m,n) * max(s) * math::eps();
    
    int r = 0;
    for (int i=0; i<s.n_elem; i++)
    {
        if (s(i) > tol)
        {
            r++;
        }
    }
    return U.cols(0, r-1);
};

inline double sigmoid(double x, double omega, double FMAX, double FMIN)
{
    double fval = FMIN + (FMAX - FMIN)/(1 - (FMAX/FMIN)*exp(-x/omega));          

    return fval;
}

inline void adjust_level(struct STATUS & status, const struct GRASTA_OPT &options)
{
//     const double ROOT                = exp((log(5)-log(options.MAX_MU))/options.MAX_LEVEL);
    const double MAX_MU              =  options.MAX_MU; // pow(ROOT, status.level)* options.MAX_MU;
    const double DEFAULT_MU_HIGH     = (MAX_MU-1)/2;
    const double DEFAULT_MU_LOW      = (MAX_MU-1)/2; //options.MIN_MU + 2;
    
    if (status.last_mu <= options.MIN_MU) {
        if (status.level > 1) {
            status.level = status.level - 1;
            status.curr_iter  = 0;
        }            
        status.last_mu    = DEFAULT_MU_LOW;        
    }
    else if (status.last_mu > MAX_MU){
        if (status.level < options.MAX_LEVEL){
            status.level = status.level + 1;
                        
            status.curr_iter  = 0;
            status.last_mu    = DEFAULT_MU_HIGH;
        }
        else
            status.last_mu    = MAX_MU;
        
    }
   
    return;
}

double estimate_step_size(const mat &Uhat, const mat & gamma_grad,const mat & w, double sG,struct STATUS & status, const struct GRASTA_OPT &options)
{
    const double    LEVEL_FACTOR        = 2;
    
    mat DL_prev_gamma;
    int newlevel = 0;
    double t = 0.0;
    
    if  (fabs(status.step_scale) < 0.0000000000001) {
        status.step_scale = options.STEP_SCALE*(1+options.MIN_MU)/sG;
        
        if (!options.QUIET) PRINTF("Estimated step-scale %.2e, sigmoid :[%.2f, %.2f, %.2f]\n",
                status.step_scale,options.OMEGA, options.FMAX, options.FMIN);
    }
    if (options.ADAPTIVE) {
        DL_prev_gamma = status.last_gamma - Uhat*(trans(Uhat)*status.last_gamma);
        
        //double grad_ip = trace(status.last_w * (trans(DL_prev_gamma) * gamma_grad) * trans(w));
        double grad_ip = trace(status.last_w * (trans(status.last_gamma) * gamma_grad) * trans(w));
        
        //double normalization = norm(DL_prev_gamma* trans(status.last_w), "fro") * norm(gamma_grad * trans(w), "fro");
        double normalization = norm(status.last_gamma * trans(status.last_w),"fro") * norm(gamma_grad * trans(w),"fro");
        
        double grad_ip_normalization = 0.0;
        
        if (fabs(normalization) > 0.00001 )
            grad_ip_normalization = grad_ip/normalization;
        
        status.last_mu = max(status.last_mu + sigmoid(-grad_ip_normalization, options.OMEGA, options.FMAX, options.FMIN) , options.MIN_MU);                
        
        t = status.step_scale * pow((double)LEVEL_FACTOR, (double)(-status.level)) * sG; // (1+status.last_mu);
    }
    else{
        status.last_mu = status.last_mu +1;
        
        t = status.step_scale * sG / (status.last_mu);
        
    }   
            

    if (t > options.MAX_STEPSIZE)
        t= options.MAX_STEPSIZE;
    
    if (options.ADAPTIVE)       
        adjust_level(status, options);
    
    return t;
}

double GRASTA_update(mat &Uhat, 
        struct STATUS &status,
        const mat &w,
        const mat &dual,
        const uvec &idx,
        const struct GRASTA_OPT &options
        )
{
    
    double sG, w_norm, gamma_norm, sG_mean , t;
    
    mat U_Omega;
    mat gamma_grad, gamma_1, gamma_2, gamma, UtDual_omega;
    U_Omega = zeros<mat>(idx.n_elem, Uhat.n_cols);
    for (int i=0; i<idx.n_elem; i++)
        U_Omega.row(i) = Uhat.row(idx(i));
    
    gamma_1         = dual;
    UtDual_omega    = trans(U_Omega) * gamma_1;
    gamma_2         = Uhat * UtDual_omega;
    gamma           = zeros<mat>(Uhat.n_rows, 1);
    gamma.elem(idx) = gamma_1;
    gamma           = gamma - gamma_2;
    
    gamma_grad      = gamma;
    
    gamma_norm = norm(gamma);
    w_norm     = norm(w);
    sG = gamma_norm * w_norm;
    
    t = estimate_step_size(Uhat, gamma_grad, w, sG, status, options);
    
//     t = status.step_scale * atan(norm(dual)/norm(w));
    
    
    // Take the gradient step along Grassmannian geodesic.
    mat  alpha = w/w_norm;
    mat  beta  = gamma/gamma_norm;

    mat  step = (cos(t)-1)*Uhat*(alpha*trans(alpha))  - sin(t)*beta*trans(alpha);
    Uhat = Uhat + step;

    status.curr_iter ++;
    status.last_gamma  = gamma_grad;
    status.last_w      = w;
    status.grasta_t    = t;

    return t;
}

void GRASTA_training(const mat &D,
        mat &Uhat,
        struct STATUS &status,
        const struct GRASTA_OPT &options,
        mat &W,
        mat &Outlier
        )
{
    int rows, cols;
    rows = D.n_rows; cols = D.n_cols;
    
    if ( !status.init ){
        status.init         = 1;
        status.curr_iter    = 0;
        
        status.last_mu      = options.MIN_MU;
        status.level        = 0;
        status.step_scale   = 0.0;
        status.last_w       = zeros(options.RANK, 1);
        status.last_gamma   = zeros(options.DIM, 1);        
        
        if (!Uhat.is_finite()){
            Uhat = orth(randn(options.DIM, options.RANK));        
        }
    }
    
    Outlier      = zeros<mat>(rows, cols);
    W            = zeros<mat>(options.RANK, cols);
    
    mat         U_Omega, y_Omega, y_t, s, w, dual, gt;
    uvec        idx, col_order;
    ADMM_OPT    admm_opt;
    double      SCALE, t, rel;
    bool        bRet;
    
    admm_opt.lambda = options.lambda;
    //if (!options.QUIET) 
    int maxIter = options.maxCycles * cols; // 20 passes through the data set
    status.hist_rel.reserve( maxIter);
                
    // Order of examples to process
    arma_rng::set_seed_random();
    col_order = conv_to<uvec>::from(floor(cols*randu(maxIter, 1)));
    
    for (int k=0; k<maxIter; k++){
        int iCol = col_order(k);
        //PRINTF("%d / %d\n",iCol, cols);
        
        y_t     = D.col(iCol);
        idx     = find_finite(y_t);
                
        y_Omega = y_t.elem(idx);
        
        SCALE = norm(y_Omega);
        y_Omega = y_Omega/SCALE;
        
        // the following for-loop is for U_Omega = U(idx,:) in matlab
        U_Omega = zeros<mat>(idx.n_elem, Uhat.n_cols);
        for (int i=0; i<idx.n_elem; i++)
            U_Omega.row(i) = Uhat.row(idx(i));
        
        // solve L-1 regression
        admm_opt.MAX_ITER = options.MAX_ITER;
        
        if (options.NORM_TYPE == L1_NORM)
            bRet = ADMM_L1(U_Omega, y_Omega, admm_opt, s, w, dual);
        else if (options.NORM_TYPE == L21_NORM){
            w = solve(U_Omega, y_Omega);
            s = y_Omega - U_Omega*w;
            dual = -s/norm(s, 2);
        }
        else {
            PRINTF("Error: norm type does not support!\n");
            return;
        }
        
        vec tmp_col = zeros<vec>(rows);
        tmp_col.elem(idx) = SCALE * s;
        
        Outlier.col(iCol) = tmp_col;
        
        W.col(iCol) =  SCALE * w;

        // take gradient step over Grassmannian
        t = GRASTA_update(Uhat, status, w, dual, idx, options);
        
        if (!options.QUIET){
            rel = subspace(options.GT_mat, Uhat);
            status.hist_rel.push_back(rel);
            
            if (rel < options.TOL){
                PRINTF("%d/%d: subspace angle %.2e\n",k,maxIter, rel);
                break;
            }
        }
        
        if (k % cols ==0){
            
            if (!options.QUIET) PRINTF("Pass %d/%d: step-size %.2e, level %d, last mu %.2f\n",
                    k % cols, options.maxCycles, t, status.level, status.last_mu);
        }
        if (status.level >= options.convergeLevel){
            // Must cycling around the dataset twice to get the correct regression weight W
            if (!options.QUIET) PRINTF("Converge at level %d, last mu %.2f\n",status.level,status.last_mu);           
            break;
        }        
    }
}
