function u_l_i = u_l_update_foursquare( trainset,rank_gamma_p,rank_delta,l_u,matrix_feature,lamda,i )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%提取第user_unique_m(i_test)个用户的gamma
    temp_gamma_old=rank_gamma_p(:,trainset(i==trainset(:,1),16));
    temp_gamma=repmat(temp_gamma_old',[1 1 matrix_feature]);
    %提取第user_unique_m(i_test)个用户的delta
    temp_delta=repmat(rank_delta(:,trainset(i==trainset(:,1),16))',[1 1 matrix_feature]);
    temp_l_u_pos=l_u(trainset(i==trainset(:,1),3),:,:);
    temp_l_u_neg=l_u(trainset(i==trainset(:,1),12),:,:);
    temp_l_u=(temp_l_u_pos-temp_l_u_neg);
    temp_up=sum(temp_gamma .* temp_delta .* temp_l_u,1);
    
    
    temp_dowm=lamda .* (sum(temp_gamma_old,2));
    temp_dowm_new=repmat(temp_dowm',[1 1 matrix_feature]);
%     temp_u_l_new=temp_up ./ temp_dowm_new;
    
    u_l_i=temp_up ./ temp_dowm_new;

end

