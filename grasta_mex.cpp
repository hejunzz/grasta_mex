//
//  grasta_mex.cpp
//
//  Created by He Jun on 14-12-20.
//  Mex Function for 
//      [Uhat, W, Outlier, status] = grasta_mex( D, Uhat, status, options)
//


#include "grasta.h"
#include "armaMex.hpp"


// D is a dense matrix, entries with inf mean missing data
// function [Uhat, W, Outlier, status] = grasta_mex( D, Uhat, status, options)
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // Get D, U, status, and options fro prhs[]
#define D_IN                prhs[0]
#define U_IN                prhs[1]
#define STATUS_IN           prhs[2]
#define OPTIONS_IN          prhs[3]
    
#define U_OUT               plhs[0]
#define W_OUT               plhs[1]
#define OUTLIER_OUT         plhs[2]
#define STATUS_OUT          plhs[3]


    
    /* Check for proper number of input and output arguments */    
    if (nrhs != 4 || nlhs!=4) 
    {
        mexErrMsgTxt("Usage: [Uhat, W, Outlier, status] = grasta_mex( D, Uhat, status, options)");
    } 
        
    mat     D_cpp = armaGetPr(D_IN);    
    mat     Uin_cpp = armaGetPr(U_IN);  
        
    struct GRASTA_OPT    OPTIONS;
    struct STATUS     	 status;    
    
    int      ndim = 2, dims[2] = {1, 1};
    int      status_fields = 10;
    const char *status_field_names[] = {"init", "last_mu", "step_scale","grasta_t", 
                "SCALE","level", "curr_iter","last_w" ,"last_gamma", "hist_rel"};    
    
            
    // Parse GRASTA_OPT 
    int number_of_fields, field_num;    
    const char *fld_name;
    
    number_of_fields = mxGetNumberOfFields(OPTIONS_IN);
    
    for (field_num=0; field_num<number_of_fields; field_num++)
    {
        mxArray *pa;
        pa = mxGetFieldByNumber(OPTIONS_IN, 0, field_num); // only one struct
        fld_name = mxGetFieldNameByNumber(OPTIONS_IN, field_num);
        
        // we only parse some important fields
        if (strcasecmp(fld_name,"maxCycles") ==0 )
        {
            OPTIONS.maxCycles = (int)mxGetScalar(pa);
        }
        else if (strcasecmp(fld_name,"MIN_ITER") ==0 )
        {
            OPTIONS.MIN_ITER = (int)mxGetScalar(pa);
        }
        else if (strcasecmp(fld_name,"MAX_ITER") ==0 )
        {
            OPTIONS.MAX_ITER = (int)mxGetScalar(pa);
        }
        else if (strcasecmp(fld_name,"DIM") ==0 )
        {
            OPTIONS.DIM = (int)mxGetScalar(pa);
        }
        else if (strcasecmp(fld_name,"RANK") ==0 )
        {
            OPTIONS.RANK = (int)mxGetScalar(pa);
        }
        else if (strcasecmp(fld_name,"ADAPTIVE") ==0 )
        {
            OPTIONS.ADAPTIVE = (int)mxGetScalar(pa);
        }
        else if(strcasecmp(fld_name,"STEP_SCALE") ==0 )
        {
            OPTIONS.STEP_SCALE = mxGetScalar(pa);
        }                
        else if (strcasecmp(fld_name,"NORM_TYPE") ==0 )
        {
            OPTIONS.NORM_TYPE = (int)mxGetScalar(pa);
        }
        else if (strcasecmp(fld_name,"convergeLevel") ==0 )
        {
            OPTIONS.convergeLevel = (int)mxGetScalar(pa);
        }
        else if (strcasecmp(fld_name,"MAX_LEVEL") ==0 )
        {
            OPTIONS.MAX_LEVEL = (int)mxGetScalar(pa);
        }        
        else if (strcasecmp(fld_name,"MIN_MU") ==0 )
        {
            OPTIONS.MIN_MU =  mxGetScalar(pa);
        }        
        else if (strcasecmp(fld_name,"MAX_MU") ==0 )
        {
            OPTIONS.MAX_MU =  mxGetScalar(pa);
        } 
        else if (strcasecmp(fld_name,"QUIET") ==0 )
        {
            OPTIONS.QUIET =  (int)mxGetScalar(pa);
        }         
        else if (strcasecmp(fld_name,"GT_mat") ==0 )
        {
            OPTIONS.GT_mat = armaGetPr(pa); // matrix type       
        }
        else if(strcasecmp(fld_name,"MAX_STEPSIZE") ==0 )
        {
            OPTIONS.MAX_STEPSIZE = mxGetScalar(pa);    
        }
        else if(strcasecmp(fld_name,"LAMBDA") ==0 )
        {
            OPTIONS.lambda = mxGetScalar(pa);    
        }        
        else if(strcasecmp(fld_name,"TOL") ==0 )
        {
            OPTIONS.TOL = mxGetScalar(pa);    
        }        
    }
    
    // parse status    
    number_of_fields = mxGetNumberOfFields(STATUS_IN);
    
    for (field_num=0; field_num<number_of_fields; field_num++)
    {
        mxArray *pa;
        pa = mxGetFieldByNumber(STATUS_IN, 0, field_num); // only one struct
        fld_name = mxGetFieldNameByNumber(STATUS_IN, field_num);
        
        if (strcasecmp(fld_name,"init") ==0 )
        {
            status.init = (int)(mxGetScalar(pa));
        }
        else if (strcasecmp(fld_name,"last_mu") ==0 )
        {
            status.last_mu = mxGetScalar(pa);
        }
        else if (strcasecmp(fld_name,"step_scale") ==0 )
        {
            status.step_scale = mxGetScalar(pa);
        }
        else if (strcasecmp(fld_name,"grasta_t") ==0 )
        {
            status.grasta_t = mxGetScalar(pa);
        }
        else if (strcasecmp(fld_name,"SCALE") ==0 )
        {
            status.SCALE = mxGetScalar(pa);
        }
        else if (strcasecmp(fld_name,"level") ==0 )
        {
            status.level = mxGetScalar(pa);
        }
        else if (strcasecmp(fld_name,"curr_iter") ==0 )
        {
            status.curr_iter = mxGetScalar(pa);
        }
        else if (strcasecmp(fld_name,"last_w") ==0 )
        {
            status.last_w = armaGetPr(pa); // matrix type            
        }
        else if (strcasecmp(fld_name,"last_gamma") ==0 )
        {
            status.last_gamma = armaGetPr(pa); // matrix type            
        }        
    }
    
    // main framwork
    mat  W, S, Uhat;
    Uhat = Uin_cpp;
    
    GRASTA_training(D_cpp, Uhat, status, OPTIONS, W, S);
    
    // prepare U for matlab
    U_OUT = armaCreateMxMatrix(Uhat.n_rows, Uhat.n_cols); //mxCreateDoubleMatrix(Uhat.n_rows, Uhat.n_cols, mxREAL); 
    armaSetPr(U_OUT,Uhat);

    // prepare W for matlab
    W_OUT = armaCreateMxMatrix(W.n_rows, W.n_cols); 
    armaSetPr(W_OUT, W);

    // prepare Outlier for matlab
    OUTLIER_OUT = armaCreateMxMatrix(S.n_rows, S.n_cols); 
    armaSetPr(OUTLIER_OUT, S);
    
    // prepare STATUS struct for matlab
    STATUS_OUT = mxCreateStructArray(ndim, dims, status_fields, status_field_names);

    mxArray *field_value;

    field_value = mxCreateDoubleScalar(status.init);
	mxSetField(STATUS_OUT,0,"init",field_value);

    field_value = mxCreateDoubleScalar(status.last_mu);
	mxSetField(STATUS_OUT,0,"last_mu",field_value);
    
    field_value = mxCreateDoubleScalar(status.step_scale);
	mxSetField(STATUS_OUT,0,"step_scale",field_value);
    
    field_value = mxCreateDoubleScalar(status.grasta_t);
	mxSetField(STATUS_OUT,0,"grasta_t",field_value);
    
    field_value = mxCreateDoubleScalar(status.SCALE);
	mxSetField(STATUS_OUT,0,"SCALE",field_value);
    
    field_value = mxCreateDoubleScalar(status.level);
	mxSetField(STATUS_OUT,0,"level",field_value);
    
    field_value = mxCreateDoubleScalar(status.curr_iter);
	mxSetField(STATUS_OUT,0,"curr_iter",field_value);

    field_value = armaCreateMxMatrix(status.last_w.n_rows, status.last_w.n_cols);
    armaSetPr(field_value, status.last_w);
	mxSetField(STATUS_OUT,0,"last_w",field_value);

    field_value = armaCreateMxMatrix(status.last_gamma.n_rows, status.last_gamma.n_cols);
    armaSetPr(field_value, status.last_gamma);
	mxSetField(STATUS_OUT,0,"last_gamma",field_value);
    
    // convert hist_rel from vector<double> to matlab matrix
    vec hist_rel(status.hist_rel.size());
    for (int i=0; i< status.hist_rel.size(); i++)
        hist_rel(i) = status.hist_rel[i];        
    field_value = armaCreateMxMatrix(hist_rel.n_rows, hist_rel.n_cols);
    armaSetPr(field_value, hist_rel);
    mxSetField(STATUS_OUT,0,"hist_rel",field_value);
    
    return;
}
