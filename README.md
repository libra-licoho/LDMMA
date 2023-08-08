# LDMMA

python and Matlab code for experiments for a novel algorithm for hyperparameter optimization problems names as LDMMA. The algorithm and models are presented in the article "Lower-level Duality Based Reformulation and Majorization Minimization Algorithm for Hyperparameter Optimization" 

# Code Execution

The Python code employs the cvxpy package with open-source solvers ECOS and SCS to solve hyperparameter optimization problems. The code also utilizes several existing algorithms for hyperparameter optimization as competitors, including VF-iDCA, grid search, random search, Implicit Differentiation and TPE. To execute the code, it is necessary to add the appropriate pathways within the code before running it. Furthermore, you can run the files 'ElasticNet_Experiments.py', 'SVM_CV_Experiments.py', 'SGL_Experiments.py' directly for experiment results and other imformation. Our code is based on anaconda(pandas, numpy, cvxpy, matplotlab). 

The Matlab code uses the MOSEK solver to address hyperparameter optimization problems with the algorithm LDMMA. To execute the code, it is necessary to add the appropriate pathways within the code before running it. Furthermore, you can run the files 'demo.m' directly for experiment， and you can conduct experiments on different problems by adjusting the 'experiment' parameter within it, where 1 represents 'elastic net', 2 represents 'sparse group lasso', and 3 represents 'support vector machine'. Our code is based on optimization solver MOSEK.


