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

%% perfom LDA
% find 'Fisherface' from training image
num_principal_components = size(X_train,2)/expression_per_person-1;
[Wpca, Ypca, b] = do_PCA(num_principal_components,X_train);

num_classes = size(X_train,2)/expression_per_person;
% num_lda_components = 3;
% num_lda_components = 10;
% num_lda_components = 20;
num_lda_components = 30;
[Wlda, Ylda, ~] = do_LDA(num_lda_components,Ypca,num_classes);
W = Wpca*Wlda;
FisherFace = W*W'*(X_train-b)+b;

for k = 1 : num_classes
    % Fisher visualization
    figure(1);
    subplot(4,9,k);
    show_img = reshape(FisherFace(:,k),row,col);
    imshow(show_img,[])
    title(['Img ',num2str(k)]);
end

% show the 3d points in a scatterplot
Ypca_test = Wpca'*(X_test-b);
Ylda_test = Wlda'*Ypca_test;

figure(3);
title('Scatter plot using test image')
hold on
grid on
for idx = 0:size(Ylda_test,2)-1
    class = fix(idx/expression_per_person);
    if class == 0  
        scatter3(Ylda_test(1,idx+1), Ylda_test(2,idx+1), Ylda_test(3,idx+1),'or','LineWidth',1.5);
        text(Ylda_test(1,idx+1), Ylda_test(2,idx+1), Ylda_test(3,idx+1), ['  data',int2str(idx+1)])    
    elseif class == 1
        scatter3(Ylda_test(1,idx+1), Ylda_test(2,idx+1), Ylda_test(3,idx+1),'og','LineWidth',1.5);
        text(Ylda_test(1,idx+1), Ylda_test(2,idx+1), Ylda_test(3,idx+1), ['  data',int2str(idx+1)])
        
    elseif class == 2
        scatter3(Ylda_test(1,idx+1), Ylda_test(2,idx+1), Ylda_test(3,idx+1),'ob','LineWidth',1.5);
        text(Ylda_test(1,idx+1), Ylda_test(2,idx+1), Ylda_test(3,idx+1), ['  data',int2str(idx+1)])
    elseif class == 3
        scatter3(Ylda_test(1,idx+1), Ylda_test(2,idx+1), Ylda_test(3,idx+1),'ok','LineWidth',1.5);
        text(Ylda_test(1,idx+1), Ylda_test(2,idx+1), Ylda_test(3,idx+1), ['  data',int2str(idx+1)])
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
    query_img.img = [query_img.img Ylda_test(:,c_rand_num)];
    query_img.class = [query_img.class repmat(n,1,2)];
    c_rand_V = [c_rand_V c_rand_num];
    c_rand_num = c_rand_num + 10;
end
gallery_img.img = Ylda_test(:,setdiff(1:size(Ylda_test,2), c_rand_V));
for m = 1:num_person
    gallery_img.class = [gallery_img.class repmat(m,1,8)];
end

% k-NN & posterior
knn = struct('class',[],'img',[]);
posterior = [];

num_classes_test = size(Ylda_test,2)/expression_per_person;
unique_num = 1:num_classes_test;
counts = zeros(size(unique_num));

K = 5; % number of neighbor = 5
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