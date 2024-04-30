function [W,Y,u] = do_LDA(num_lda_components, X, num_classes)

    num_dim = size(X,1);           % Dimensionality
    num_sample = size(X,2);         % number of sample 
    
    N_i = num_sample/num_classes;   % number of sample in class
    N = num_classes;                % number of class
    
    %------ Get mean of class : u_i, u
    u_i = zeros(num_dim,N); % mean of sample in class, u_i = [u_1 u_2 u_3 ...]
    u = zeros(num_dim,1);   % mean of class, u
    for idx = 1:N
        u_i(:,idx) = mean(X(:,(idx-1)*N_i+1:idx*N_i),2);
        u = u + N_i*u_i(:,idx);
    end
    u = u/N;
    
    %------ Within Scatter matrix
    %------ S_W = sum(S_i) => S_W = S_1+S_2+...
    S_W = 0;
    for idx = 1:N
        x_tmp = X(:,(idx-1)*N_i+1:idx*N_i)-u_i(:,idx); % x-u_i
        S_i = 1/N_i*(x_tmp*x_tmp'); % S_i = sum((x-u_i)*(x-u_i)'))
        S_W = S_W + S_i;
    end
    
    %------ Between Scatter matrix
    %------ S_B = sum(N_i*(u_i-u)*(u_i-u))'): outer product
    S_B = 0;
    for idx = 1:N
        S_B = S_B + N_i*(u_i(:,idx)-u)*(u_i(:,idx)-u)';
    end
    
    %------ inv(S_W)*S_B*W = lambda*W
    %------ sorting eigenvector, eigenvalue by descend
    [W, lambda] = eig(inv(S_W)*S_B); % EVD
    [~,sort_indices] = sort(diag(lambda), 'descend');
    
    W = W(:, sort_indices(1:num_lda_components));
    Y = W'*X;
end