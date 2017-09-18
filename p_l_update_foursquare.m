function p_l_i = p_l_update_foursquare( trainset,rank_gamma_p,rank_delta,l_p,matrix_feature,lamda,i )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%提取第user_unique_m(i_test)个用户的gamma
    temp_gamma_old=rank_gamma_p(:,trainset(i==trainset(:,2),16));
    temp_gamma=repmat(temp_gamma_old',[1 1 matrix_feature]);
    %提取第user_unique_m(i_test)个用户的delta
    temp_delta=repmat(rank_delta(:,trainset(i==trainset(:,2),16))',[1 1 matrix_feature]);
    temp_l_p_pos=l_p(trainset(i==trainset(:,2),3),:,:);
    temp_l_p_neg=l_p(trainset(i==trainset(:,2),12),:,:);
    temp_l_p=(temp_l_p_pos-temp_l_p_neg);
    temp_up=sum(temp_gamma .* temp_delta .* temp_l_p,1);
   
    
    temp_dowm=lamda .* (sum(temp_gamma_old,2));
    temp_dowm_new=repmat(temp_dowm',[1 1 matrix_feature]);
%     temp_u_l_new=temp_up ./ temp_dowm_new;
    
    p_l_i=temp_up ./ temp_dowm_new;

end

