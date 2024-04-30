close all 
clear
clc

%% get image 
path = 'ORL_database\';
expression_per_person = 10;
num_person = 40;
[X, row, col] = get_image(path,num_person,expression_per_person);

%% divide training/test image
% training image:test image = 9:1
X_train = [];
X_test = [];

select_num_person = num_person*0.1;
rand_num = sort(randperm(num_person,select_num_person)); % 랜덤으로 n개의 class 선택
rand_V = [];
for idx = 1: select_num_person
    range = (rand_num(idx)-1)*expression_per_person+1:rand_num(idx)*expression_per_person;
    rand_V = [rand_V range]; % test에 포함되지 않은 나머지 항목들은 train에 저장하기 위함
    X_test = [X_test X(:,range)]; 
end
X_train = X(:, setdiff(1:size(X,2), rand_V));

%% perfom PCA
% find 'Eigenface' from training image
num_principal_components = size(X_train,2)-1;
num_principal_components = 80;
num_principal_components = 150;
% num_principal_components = 300;
[W, ~, b] = do_PCA(num_principal_components,X_train);
S
for k = 1 : size(X_train,2)/expression_per_person
    % EigenFace visualization
    figure(1);
    subplot(4,9,k);
    show_img = reshape(W(:,k),row,col);
    imshow(show_img,[])
    title(['W= ',num2str(k)]);
end

% reconstructed images for the test data
Ytest = W'*(X_test-b);

for k = 1 : size(Ytest,2)
    figure(2);
    subplot(5,8,k);
    x_reconst = W*Ytest+b;
    show_img = reshape(x_reconst(:,k),row,col);
    imshow(show_img,[])
    title(['Img ',num2str(k)]);
end

% show the 3d points in a scatterplot
num_principal_components = 3; % 3차원으로 차원 감소
[W, Y, b] = do_PCA(num_principal_components,X_train);
Ytest_3d = W'*(X_test-b);        % X_test를 projection

figure(3);
title('Scatter plot using test image')
hold on
grid on
for idx = 0:size(Ytest_3d,2)-1
    class = fix(idx/expression_per_person);
    if class == 0  
        scatter3(Ytest_3d(1,idx+1), Ytest_3d(2,idx+1), Ytest_3d(3,idx+1),'or','LineWidth',1.5);
        text(Ytest_3d(1,idx+1), Ytest_3d(2,idx+1), Ytest_3d(3,idx+1), ['  data',int2str(idx+1)])    
    elseif class == 1
        scatter3(Ytest_3d(1,idx+1), Ytest_3d(2,idx+1), Ytest_3d(3,idx+1),'og','LineWidth',1.5);
        text(Ytest_3d(1,idx+1), Ytest_3d(2,idx+1), Ytest_3d(3,idx+1), ['  data',int2str(idx+1)])
        
    elseif class == 2
        scatter3(Ytest_3d(1,idx+1), Ytest_3d(2,idx+1), Ytest_3d(3,idx+1),'ob','LineWidth',1.5);
        text(Ytest_3d(1,idx+1), Ytest_3d(2,idx+1), Ytest_3d(3,idx+1), ['  data',int2str(idx+1)])
    elseif class == 3
        scatter3(Ytest_3d(1,idx+1), Ytest_3d(2,idx+1), Ytest_3d(3,idx+1),'ok','LineWidth',1.5);
        text(Ytest_3d(1,idx+1), Ytest_3d(2,idx+1), Ytest_3d(3,idx+1), ['  data',int2str(idx+1)])
    else
        disp("less than sample!")
        return 
    end
end
xlabel('x1')
ylabel('x2')
zlabel('x3')
lgd = legend();
lgd.NumColumns = 4;

%% classification
% divide gallery/query image from test image
% gallery image:query image = 8:2

gallery_img = struct('class',[],'img',[]);
query_img = struct('class',[],'img',[]);

c_rand_num = sort(randperm(expression_per_person,2));
c_rand_V = [];
for n = 1:select_num_person
    query_img.img = [query_img.img Ytest_3d(:,c_rand_num)];
    query_img.class = [query_img.class repmat(n,1,2)];
    c_rand_V = [c_rand_V c_rand_num];
    c_rand_num = c_rand_num + 10;
end
gallery_img.img = Ytest_3d(:,setdiff(1:size(Ytest_3d,2), c_rand_V));
for m = 1:num_person
    gallery_img.class = [gallery_img.class repmat(m,1,8)];
end

% k-NN & posterior
knn = struct('class',[],'img',[]);
posterior = [];

num_classes_test = size(Ytest_3d,2)/expression_per_person;
unique_num = 1:num_classes_test;
counts = zeros(size(unique_num));

K = 5;
for idx = 1:size(query_img.img,2)
     % get knn
    knn(idx) = get_kNN(K, gallery_img, query_img.img(:,idx));

    % caculate posterior
    for cnt = 1:length(unique_num)
        counts(cnt) = sum(knn(idx).class==unique_num(cnt));
    end

    posterior = [posterior (counts/K)'];
end

disp('Confusion Matrix')
disp('----------------')
disp([int8(query_img.class); int8(posterior*100)])