function [a_hat] = lasso_func(lasso_mat,y,lambdas)

save('lasso_values','lasso_mat', 'y', 'lambdas');
lasso_py = py.importlib.import_module('lasso');
lasso_py.lasso();
load('lasso_result');
delete('lasso_values.mat');
delete('lasso_result.mat')

end

