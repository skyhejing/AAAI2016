function [Myresult_1,Myresult_5,Myresult_10,Myresult_20,Myresult_30,Myresult_40,Myresult_50,Myresult_60,Myresult_70,Myresult_80,Myresult_90,Myresult_100,right_distance,right_i_1,right_i_5,right_i_10,right_i_20] = Myprediction_foursquare_camera_ready( testset,l_u,u_l,l_p,p_l,rank_alpha_p_testset,rho,location_unique_test,distance_matrix_frac,i)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    u_l_predict=u_l(testset(i,1),:,:);
    p_l_predict=p_l(testset(i,2),:,:);
    
    l_u_test=l_u(location_unique_test(:,1),:,:);
    l_p_test=l_p(location_unique_test(:,1),:,:);
    
    u_l_tensor=repmat(u_l_predict,[size(l_u_test,1) 1 1]);
    p_l_tensor=repmat(p_l_predict,[size(l_p_test,1) 1 1]);
    
    tensor_result=u_l_tensor .* l_u_test + p_l_tensor .* l_p_test;
    tensor_result_sum=sum(tensor_result,3);
   
    
    prelocation_line=location_unique_test(location_unique_test(:,1)==testset(i,2),4);
    distance_test_frac=distance_matrix_frac(prelocation_line,:);
    
    rho_distance=rho * distance_test_frac;
    
    tensor_rho_result=tensor_result_sum' + rho_distance;
    
    alpha_result=bsxfun(@times,tensor_rho_result,rank_alpha_p_testset(:,i));
    
    %This is the wrong code.
    %alpha_result=rank_alpha_p_testset' .* tensor_result_sum;
    alpha_result_sum=sum(alpha_result,1);
    
    [~, index_sum]=sort(alpha_result_sum,'descend');
    index_sum_top1=index_sum(1);
    index_sum_top5=index_sum(1:5);
    index_sum_top10=index_sum(1:10);
    index_sum_top20=index_sum(1:20);
    index_sum_top30=index_sum(1:30);
    index_sum_top40=index_sum(1:40);
    index_sum_top50=index_sum(1:50);
    index_sum_top60=index_sum(1:60);
    index_sum_top70=index_sum(1:70);
    index_sum_top80=index_sum(1:80);
    index_sum_top90=index_sum(1:90);
    index_sum_top100=index_sum(1:100);
    
    if any(testset(i,3)==location_unique_test(index_sum_top1,1))
        Myresult_1=1;
        right_i_1=i;
    else
        Myresult_1=0;
        right_i_1=-1;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top5,1))
        Myresult_5=1;
        right_i_5=i;
    else
        Myresult_5=0;
        right_i_5=-1;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top10,1))
        Myresult_10=1;
        right_distance=testset(i,8);
        right_i_10=i;
    else
        Myresult_10=0;
        right_distance=-1;
        right_i_10=-1;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top20,1))
        Myresult_20=1;
        right_i_20=i;
    else
        Myresult_20=0;
        right_i_20=-1;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top30,1))
        Myresult_30=1;
    else
        Myresult_30=0;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top40,1))
        Myresult_40=1;
    else
        Myresult_40=0;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top50,1))
        Myresult_50=1;
    else
        Myresult_50=0;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top60,1))
        Myresult_60=1;
    else
        Myresult_60=0;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top70,1))
        Myresult_70=1;
    else
        Myresult_70=0;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top80,1))
        Myresult_80=1;
    else
        Myresult_80=0;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top90,1))
        Myresult_90=1;
    else
        Myresult_90=0;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top100,1))
        Myresult_100=1;
    else
        Myresult_100=0;
    end
end