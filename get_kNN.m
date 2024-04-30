function k_nearest_neighbor = get_kNN(K, gallery_img, query_img)
    k_nearest_neighbor = struct('class', [], 'img', []);

    % k-NN
    num_img = size(gallery_img.img,2);
    distance = zeros(1,num_img);
    for idx = 1:num_img
        distance(idx) = norm(gallery_img.img(:,idx)-query_img);
    end
    
    % sorting by ascending order
    [~, dis_sort] = sort(distance);
    k_nearest_neighbor.img = gallery_img.img(:,dis_sort(1:K));
    k_nearest_neighbor.class = gallery_img.class(dis_sort(1:K));
end