% read image
%------ 각 얼굴사진의 데이터를 vectorization하고 사진들의 집합을 배열로 만듦
%------ input: image folder path, number of person, expression/person
%------ output: X, image row pixel, image col pixel
%------ X행렬의 행:이미지의 dimension, 열: 각 이미지 번호
function [X, img_row, img_col] = get_image(path, num_person,expression_per_person)    
    img = 1;
    X = [];
    for i = 1:num_person % 사람의 수 만큼 반복
        for j = 1:expression_per_person % 한 사람당 표정 수만큼 반복
            path_temp = strcat(path,int2str(img),'_',int2str(i),'.jpg'); % 이미지 번호 ex) 123_1.jpg / '_' 앞: image number /'_' 뒤: 사람 번호
            img_temp = imread(path_temp);
            [img_row, img_col] = size(img_temp);
            X_tmp = reshape(img_temp,img_row*img_col,1);
            X = [X double(X_tmp)];
    
            img = img+1;
        end
    end
end