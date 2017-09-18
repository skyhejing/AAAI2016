pattern_num=6; %the number of pattern

lamda=1;    %init the lamda
matrix_feature=64;    %the number of matrix feature


%to be done. We can use struct to replace tensor.20150724
%init tensor parameter
u_l=normrnd(0,2/lamda,[size(user_unique,1),pattern_num,matrix_feature]);
l_u=normrnd(0,2/lamda,[size(location_unique,1),pattern_num,matrix_feature]);

p_l=normrnd(0,2/lamda,[size(location_unique,1),pattern_num,matrix_feature]);
l_p=normrnd(0,2/lamda,[size(location_unique,1),pattern_num,matrix_feature]);

%init new tensor parameter
u_l_new=normrnd(0,2/lamda,[size(user_unique,1),pattern_num,matrix_feature]);
l_u_new=normrnd(0,2/lamda,[size(location_unique,1),pattern_num,matrix_feature]);

p_l_new=normrnd(0,2/lamda,[size(location_unique,1),pattern_num,matrix_feature]);
l_p_new=normrnd(0,2/lamda,[size(location_unique,1),pattern_num,matrix_feature]);

%alpha=pattern_num * the number of features.
alpha=normrnd(0,2/lamda,pattern_num,3);

%The coefficient of distance
rho=normrnd(0,2/lamda,pattern_num,1);


[row_trainset, col_trainset]=size(trainset);
[row_testset, col_testset]=size(testset);
%useless.
%rank_expection=zeros(pattern_num,row_trainset);


%deal with softmax
rank_alpha=zeros(pattern_num,row_trainset);
%It will use the new rank_alpha when calculate the objective value and the prediction accuracy rate.
rank_alpha_new=zeros(pattern_num,row_trainset);
rank_alpha_testset=zeros(pattern_num,row_testset);

rank_alpha_p=zeros(pattern_num,row_trainset);
rank_alpha_p_new=zeros(pattern_num,row_trainset);
rank_alpha_p_testset=zeros(pattern_num,row_testset);

%not vector program
% tic
% for i_1=1:row_trainset
%     for i_2=1:pattern_num
%         rank_alpha(i_2,i_1)=exp(alpha(i_2,:)*trainset(i_1,4:5)');
%     end
% end
% toc
iteration_i=1;
%The dataset exist distance=0.
postive_vs_negative_distance=1 ./ (trainset(:,8)+10000) - 1./ (trainset(:,15)+10000);


%20151125.camera ready
distance_all=testset(:,8);
Mylength(1)= length(find(distance_all>=0&distance_all<5));
Mylength(2)= length(find(distance_all>=5&distance_all<10));
Mylength(3)= length(find(distance_all>=10&distance_all<20));
Mylength(4)= length(find(distance_all>=20&distance_all<50));
Mylength(5)= length(find(distance_all>=50&distance_all<100));
Mylength(6)= length(find(distance_all>=100&distance_all<200));
Mylength(7)= length(find(distance_all>=200&distance_all<500));
Mylength(8)= length(find(distance_all>=500&distance_all<800));
Mylength(9)= length(find(distance_all>=800&distance_all<1200));
Mylength(10)= length(find(distance_all>=1200&distance_all<1600));
Mylength(11)= length(find(distance_all>=1600&distance_all<2000));
Mylength(12)= length(find(distance_all>=2000&distance_all<2500));
Mylength(13)= length(find(distance_all>=2500&distance_all<3000));
Mylength(14)= length(find(distance_all>=3000&distance_all<4000));
Mylength(15)= length(find(distance_all>=4000&distance_all<5000));
Mylength(16)= length(find(distance_all>=5000));

while 1
%as same as the last code block, but this have used vector program
tic
rank_alpha=exp(alpha * trainset(:,9:11)');
toc

%as same as the last code block, but this have used vector program
tic
rank_alpha_p=bsxfun(@rdivide,rank_alpha,sum(rank_alpha,1));
rank_alpha_p(rank_alpha_p<0.00001)=0.00001;
toc

%deal with tensor
rank_tenosr=zeros(pattern_num,row_trainset);
rank_tenosr_sigma=zeros(pattern_num,row_trainset);

%distance.20150817
rank_distance=zeros(pattern_num,row_trainset);

%The fourthe version. This version consumes less time than the third version.
tic
u_l_trainset=u_l(trainset(:,1),:,:);
l_u_trainset_pos=l_u(trainset(:,3),:,:);
l_u_trainset_neg=l_u(trainset(:,12),:,:);
l_p_trainset_pos=l_p(trainset(:,3),:,:);
l_p_trainset_neg=l_p(trainset(:,12),:,:);
p_l_trainset=p_l(trainset(:,2),:,:);

u_l_sum=sum(u_l_trainset .* (l_u_trainset_pos -l_u_trainset_neg),3);
u_l_sum(isnan(u_l_sum))=700;
u_l_sum(u_l_sum<0.00001)=0.00001;
p_l_sum=sum((l_p_trainset_pos-l_p_trainset_neg) .*p_l_trainset,3);
p_l_sum(isnan(p_l_sum))=700;
p_l_sum(p_l_sum<0.00001)=0.00001;
rank_tenosr=u_l_sum + p_l_sum;
toc


rank_rho= rho * postive_vs_negative_distance';

rank_tensor_rho=rank_tenosr' + rank_rho;
%The threshold sets to 700, because exp(700) is the biggest exp() value.
rank_tensor_rho(rank_tensor_rho>700)=700;
rank_tensor_rho(rank_tensor_rho<0.00001)=0.00001;

rank_tenosr_sigma=exp(rank_tensor_rho) ./ (1 +exp(rank_tensor_rho));

%deal with tensor*softmax
rank_gamma=zeros(pattern_num,row_trainset);
rank_gamma_p=zeros(pattern_num,row_trainset);
rank_gamma=rank_tenosr_sigma .* rank_alpha_p;
rank_gamma_p=bsxfun(@rdivide,rank_gamma,sum(rank_gamma,1));

%M step

rank_delta=zeros(pattern_num,row_trainset);
rank_delta=1 ./ (1 +exp(rank_tensor_rho));
rank_delta(rank_delta<0.00001)=0.00001;



%test=rank_gamma_p .* rank_delta .* (l_u(trainset(:,3),:,:) -l_u(trainset(:,6),:,:));

%index the line number
trainset_index=1:row_trainset;
trainset(:,16)=trainset_index';

user_unique_m=unique(trainset(:,1),'stable');
location_unique_pre_m=unique(trainset(:,2),'stable');
location_unique_pos_m=unique(trainset(:,3),'stable');
location_unique_neg_m=unique(trainset(:,12),'stable');

tic
%update alpha. the first feature.
alpha_dowm=lamda .* (sum(rank_gamma_p,2));
%alpha_dowm_new=repmat(temp_dowm',[1 1 matrix_feature]);

trainset_feature_1=repmat(trainset(:,9)',[6 1]);

alpha_up_1=rank_gamma_p .* trainset_feature_1 .* (1-rank_alpha_p);

alpha_up_1_sum=sum(alpha_up_1,2);

alpha(:,1)=alpha_up_1_sum ./ alpha_dowm;
toc

%update alpha. the second feature.
trainset_feature_2=repmat(trainset(:,10)',[6 1]);

alpha_up_2=rank_gamma_p .* trainset_feature_2 .* (1-rank_alpha_p);

alpha_up_2_sum=sum(alpha_up_2,2);

alpha(:,2)=alpha_up_2_sum ./ alpha_dowm;

%update alpha. the third feature.
trainset_feature_3=repmat(trainset(:,11)',[6 1]);

alpha_up_3=rank_gamma_p .* trainset_feature_3 .* (1-rank_alpha_p);

alpha_up_3_sum=sum(alpha_up_3,2);

alpha(:,3)=alpha_up_3_sum ./ alpha_dowm;

alpha(alpha<0.00001)=0.00001;

%update the rho.
trainset_postive_vs_negative_distance=repmat(postive_vs_negative_distance',[6 1]);
rho_up=rank_gamma_p .* rank_delta .* trainset_postive_vs_negative_distance;
rho_up_sum=sum(rho_up,2);
rho=rho_up_sum ./ alpha_dowm;
rho(rho<0.00001)=0.00001;


%The eighth revise. This is OK. update u_l
tic
u_l_new_cell=arrayfun(@(i) u_l_update_foursquare(trainset,rank_gamma_p,rank_delta,l_u,matrix_feature,lamda,i),user_unique_m,'UniformOutput', false);
%It can be used when testing.
%test=u_l_update(trainset,rank_gamma_p,rank_delta,l_u,matrix_feature,lamda,user_unique_m(1));
n_u_l=1:size(user_unique_m,1);
u_l_new(user_unique_m,:,:)=cell2mat(u_l_new_cell(n_u_l));
u_l_new(isnan(u_l_new))=700;
u_l_new(u_l_new<0.00001)=0.00001;
toc

%update l_u_pos and l_p_pos
tic
[l_u_pos_new_cell, l_p_pos_new_cell]=arrayfun(@(i) l_u_p_pos_update_foursquare( trainset,rank_gamma_p,rank_delta,u_l,p_l,matrix_feature,lamda,i ),location_unique_pos_m,'UniformOutput', false);
n_l_u_pos=1:size(location_unique_pos_m,1);
l_u_new(location_unique_pos_m,:,:)=cell2mat(l_u_pos_new_cell(n_l_u_pos));
l_u_new(isnan(l_u_new))=700;
l_u_new(l_u_new<0.00001)=0.00001;
l_p_new(location_unique_pos_m,:,:)=cell2mat(l_p_pos_new_cell(n_l_u_pos));
l_p_new(isnan(l_p_new))=700;
l_p_new(l_p_new<0.00001)=0.00001;
toc

%update l_u_neg and l_p_neg
tic
[l_u_neg_new_cell, l_p_neg_new_cell]=arrayfun(@(i) l_u_p_neg_update_foursquare( trainset,rank_gamma_p,rank_delta,u_l,p_l,matrix_feature,lamda,i ),location_unique_neg_m,'UniformOutput', false);
n_l_u_neg=1:size(location_unique_neg_m,1);
l_u_new(location_unique_neg_m,:,:)=cell2mat(l_u_neg_new_cell(n_l_u_neg));
l_u_new(isnan(l_u_new))=700;
l_u_new(l_u_new<0.00001)=0.00001;
l_p_new(location_unique_neg_m,:,:)=cell2mat(l_p_neg_new_cell(n_l_u_neg));
l_p_new(isnan(l_p_new))=700;
l_p_new(l_p_new<0.00001)=0.00001;
toc

% %update p_l
tic
p_l_new_cell=arrayfun(@(i) p_l_update_foursquare(trainset,rank_gamma_p,rank_delta,l_p,matrix_feature,lamda,i),location_unique_pre_m,'UniformOutput', false);
n_p_l=1:size(location_unique_pre_m,1);
p_l_new(location_unique_pre_m,:,:)=cell2mat(p_l_new_cell(n_p_l));
p_l_new(isnan(p_l_new))=700;
p_l_new(p_l_new<0.00001)=0.00001;
toc


%prediction
tic
rank_alpha_testset=exp(alpha * testset(:,9:11)');
rank_alpha_p_testset=bsxfun(@rdivide,rank_alpha_testset,sum(rank_alpha_testset,1));

[result_vector_1,result_vector_5,result_vector_10,result_vector_20,result_vector_30,result_vector_40,result_vector_50,result_vector_60,result_vector_70,result_vector_80,result_vector_90,result_vector_100,right_distance,right_i_1,right_i_5,right_i_10,right_i_20]=arrayfun(@(i) Myprediction_foursquare_camera_ready( testset,l_u_new,u_l_new,l_p_new,p_l_new,rank_alpha_p_testset,rho,location_unique_test,distance_matrix_frac,i),1:row_testset);
%Myprediction_foursquare( testset,l_u_new,u_l_new,l_p_new,p_l_new,rank_alpha_p_testset,rho,location_unique_test,distance_matrix_frac,22);
disp(['top1=£º' num2str(sum(result_vector_1)./ row_testset)]);
disp(['top5=£º' num2str(sum(result_vector_5)./ row_testset)]);
disp(['top10=£º' num2str(sum(result_vector_10)./ row_testset)]);
disp(['top20=£º' num2str(sum(result_vector_20)./ row_testset)]);
disp(['top30=£º' num2str(sum(result_vector_30)./ row_testset)]);
disp(['top40=£º' num2str(sum(result_vector_40)./ row_testset)]);
disp(['top50=£º' num2str(sum(result_vector_50)./ row_testset)]);
disp(['top60=£º' num2str(sum(result_vector_60)./ row_testset)]);
disp(['top70=£º' num2str(sum(result_vector_70)./ row_testset)]);
disp(['top80=£º' num2str(sum(result_vector_80)./ row_testset)]);
disp(['top90=£º' num2str(sum(result_vector_90)./ row_testset)]);
disp(['top100=£º' num2str(sum(result_vector_100)./ row_testset)]);
eval(['save datav2',num2str(iteration_i)]);
toc
%end %if  0==mod(iteration_i,50)

disp(['iteration£º' num2str(iteration_i)]);
iteration_i=iteration_i+1;

%20151125. camera ready
right_length(1)= length(find(right_distance>=0&right_distance<5));
right_length(2)= length(find(right_distance>=5&right_distance<10));
right_length(3)= length(find(right_distance>=10&right_distance<20));
right_length(4)= length(find(right_distance>=20&right_distance<50));
right_length(5)= length(find(right_distance>=50&right_distance<100));
right_length(6)= length(find(right_distance>=100&right_distance<200));
right_length(7)= length(find(right_distance>=200&right_distance<500));
right_length(8)= length(find(right_distance>=500&right_distance<800));
right_length(9)= length(find(right_distance>=800&right_distance<1200));
right_length(10)= length(find(right_distance>=1200&right_distance<1600));
right_length(11)= length(find(right_distance>=1600&right_distance<2000));
right_length(12)= length(find(right_distance>=2000&right_distance<2500));
right_length(13)= length(find(right_distance>=2500&right_distance<3000));
right_length(14)= length(find(right_distance>=3000&right_distance<4000));
right_length(15)= length(find(right_distance>=4000&right_distance<5000));
right_length(16)= length(find(right_distance>=5000));

disp(['distance5=£º' num2str(right_length(1)./ Mylength(1))]);
disp(['distance10=£º' num2str(right_length(2)./ Mylength(2))]);
disp(['distance20=£º' num2str(right_length(3)./ Mylength(3))]);
disp(['distance50=£º' num2str(right_length(4)./ Mylength(4))]);
disp(['distance100=£º' num2str(right_length(5)./ Mylength(5))]);
disp(['distance200=£º' num2str(right_length(6)./ Mylength(6))]);
disp(['distance500=£º' num2str(right_length(7)./ Mylength(7))]);
disp(['distance800=£º' num2str(right_length(8)./ Mylength(8))]);
disp(['distance1200=£º' num2str(right_length(9)./ Mylength(9))]);
disp(['distance1600=£º' num2str(right_length(10)./ Mylength(10))]);
disp(['distance2000=£º' num2str(right_length(11)./ Mylength(11))]);
disp(['distance2500=£º' num2str(right_length(12)./ Mylength(12))]);
disp(['distance3000=£º' num2str(right_length(13)./ Mylength(13))]);
disp(['distance4000=£º' num2str(right_length(14)./ Mylength(14))]);
disp(['distance5000=£º' num2str(right_length(15)./ Mylength(15))]);
disp(['distance5000+=£º' num2str(right_length(16)./ Mylength(16))]);

%in the end, update the tensor parameters.
u_l=u_l_new;
l_u=l_u_new;
p_l=p_l_new;
l_p=l_p_new;

end %while 1


