function [W,Y,b] = do_PCA(num_principal_components,X)
    %------ W: EigenVector
    %------ Y: Projection on the principal component
    %------ b: mean of each dimension

    X = double(X);
    n = size(X,2); % number of sample
    b = mean(X,2); 
    
    Xmb = X - b;
    S = 1/n*(Xmb*Xmb'); % Scatter matrix
    
    % sorted eigenvectors, eigenvalues by descend
    [W, lambda] = eig(S);
    % disp(lambda)
    [~,sort_mat] = sort(diag(lambda), 'descend');

    W = -W(:, sort_mat(1:num_principal_components));
    Y = W'*Xmb;
    
end