close all;
clear all;
clc;

%% Files initiated
train_user_file = 'train_triplets.txt';
test_user_file = 'year1_test_triplets_hidden.txt';
valid_user_file = 'year1_valid_triplets_hidden.txt';
song_file = 'song_data.csv';

%% Data created numerically
[train_numerical_data,train_user_unique,train_user_unique_idx,train_song_unique,train_song_unique_idx] = read_files(10000,train_user_file,song_file);
[test_numerical_data,test_user_unique,test_user_unique_idx,test_song_unique,test_song_unique_idx] = read_files(5000,test_user_file,song_file);
[valid_numerical_data,valid_user_unique,valid_user_unique_idx,valid_song_unique,valid_song_unique_idx] = read_files(5000,valid_user_file,song_file);

%% Create User-Song or Utility Matrix (will be fed into SVD matrix factorization algorithm AND collaborative filtering algorithm)
training_utility_matrix = userSongMatrix(train_user_unique,train_song_unique,train_numerical_data);
%save('training_utility_matrix_10000','training_utility_matrix');

testing_utility_matrix = userSongMatrix(test_user_unique,test_song_unique,test_numerical_data);
%save('testing_utility_matrix_10000','testing_utility_matrix');

validation_utility_matrix = userSongMatrix(valid_user_unique,valid_song_unique,valid_numerical_data);
%save('validation_utility_matrix_10000','validation_utility_matrix');

% Save all the required matrices to avoid expensive run for later
% save('train_numerical_data_10000_whole','train_numerical_data');
% save('test_numerical_data_10000_whole','test_numerical_data');
% save('train_user_unique_10000_whole','train_user_unique_idx');
% save('train_song_unique_10000_whole','train_song_unique_idx');
% save('test_user_unique_10000_whole','test_user_unique_idx');
% save('test_song_unique_10000_whole','test_song_unique_idx');
% save('valid_numerical_data_10000_whole','valid_numerical_data');
% save('valid_user_unique_10000_whole','valid_user_unique_idx');
% save('valid_song_unique_10000_whole','valid_song_unique_idx');

%% Learning Algorithms %%

test_numerical_data = popularity(test_song_unique,test_numerical_data);
valid_numerical_data = popularity(valid_song_unique,valid_numerical_data);

% First prepare label for training and testing data
% User-based algorithms
Y_train_userbased = train_numerical_data(:,3);
Y_test_userbased = test_numerical_data(:,3);
Y_valid_userbased = valid_numerical_data(:,3);
% Item-based algorithms
Y_train_itembased = train_numerical_data(:,3);         
Y_test_itembased = test_numerical_data(:,3);
Y_valid_itembased = valid_numerical_data(:,3);



%% 1) SVD Matrix Factorization Algorithm
[U_train,S_train,V_train] = nnmf(training_utility_matrix,70); % non-negative matrix factoriztion
[U_test,S_test,V_test] = nnmf(testing_utility_matrix,70); % non-negative matrix factoriztion
[U_valid,S_valid,V_valid] = nnmf(validation_utility_matrix,70); % non-negative matrix factoriztion

X_train_MF_unique = U_train*S_train*V_train';
X_test_MF_unique = U_test*S_test*V_test';
X_valid_MF_unique = U_valid*S_valid*V_valid';

X_train_MF = zeros(size(train_user_unique_idx,1),size(X_train_MF_unique,2));
X_train_MF(1:size(train_user_unique_idx,1),:) = X_train_MF_unique(train_user_unique_idx(1:size(train_user_unique_idx,1)),:); 

X_test_MF = zeros(size(test_user_unique_idx,1),size(X_test_MF_unique,2));
X_test_MF(1:size(test_user_unique_idx,1),:) = X_test_MF_unique(test_user_unique_idx(1:size(test_user_unique_idx,1)),:); 

X_valid_MF = zeros(size(valid_user_unique_idx,1),size(X_valid_MF_unique,2));
X_valid_MF(1:size(valid_user_unique_idx,1),:) = X_valid_MF_unique(valid_user_unique_idx(1:size(valid_user_unique_idx,1)),:); 


% Same number of features
feat_num_MF = min(size(X_train_MF,2),size(X_test_MF,2));
feat_num_MF = min(feat_num_MF,size(X_valid_MF,2));
X_train_MF(:,(feat_num_MF+1):end)=[];
X_test_MF(:,(feat_num_MF+1):end)=[];
X_valid_MF(:,(feat_num_MF+1):end)=[];

% 1.1) KNN fit and predict for Matrix Factorization
score_knn_MF = zeros(10,1);
for k =1:10 % go over multiple k values and see the result on validation set then choose the best k
    knn_mdl_MF = fitcknn(X_train_MF,Y_train_userbased,'NumNeighbors',k,'Distance','euclidean','Standardize',0);
    label_result_knn_MF = predict(knn_mdl_MF,X_valid_MF);
    % Evaluation Error Calculation
    score_knn_MF(k) = evaluate_labels(valid_user_unique,valid_user_unique_idx,valid_numerical_data,Y_valid_userbased,label_result_knn_MF);
end;
[~,score_knn_MF_idx] = max(score_knn_MF); % K = score_knn_MF_idx provides the best score with validation set
knn_mdl_MF = fitcknn(X_train_MF,Y_train_userbased,'NumNeighbors',score_knn_MF_idx,'Distance','euclidean','Standardize',0);
label_result_knn_MF = predict(knn_mdl_MF,X_test_MF);
score_knn_MF = evaluate_labels(test_user_unique,test_user_unique_idx,test_numerical_data,Y_test_userbased,label_result_knn_MF);

% 1.2) SVM fit and predict for Matrix Factorization
svm_mdl_MF = fitcecoc(X_train_MF,Y_train_userbased);
label_result_svm_MF = predict(svm_mdl_MF,X_test_MF);
score_svm_MF = evaluate_labels(test_user_unique,test_user_unique_idx,test_numerical_data,Y_test_userbased,label_result_svm_MF);

% 1.3) Ensemble fit and predict for Matrix Factorization
%ens_bagging_mdl_user = fitensemble(X_train_userbased,Y_train_userbased,'Bag',4,'tree','Type','classification');
ens_boost_mdl_MF = fitensemble(X_train_MF,Y_train_userbased,'AdaBoostM2',4,'tree','Type','classification');
label_result_ens_MF = predict(ens_boost_mdl_MF,X_test_MF);
score_ensemble_MF = evaluate_labels(test_user_unique,test_user_unique_idx,test_numerical_data,Y_test_userbased,label_result_ens_MF);


%% 2) User-based Collaborative Filtering

% User-based similarity (Cosine or Pearson)

%X_train_userbased_unique = corr(training_utility_matrix'); % matlab built-in, default is Pearson Correlation
%X_test_userbased_unique = corr(testing_utility_matrix'); % matlab built-in, default is Pearson Correlation
[~,X_train_userbased_unique] = correlation_calculation(training_utility_matrix,0.7,0.1,'pearson'); % type can be Pearson Correlation OR Cosine Correlation OR default in the paper
[~,X_test_userbased_unique] = correlation_calculation(testing_utility_matrix,0.7,0.1,'pearson'); % type can be Pearson Correlation OR Cosine Correlation OR default in the paper
[~,X_valid_userbased_unique] = correlation_calculation(validation_utility_matrix,0.7,0.1,'pearson');

X_train_userbased = zeros(size(train_user_unique_idx,1),size(X_train_userbased_unique,2));
X_train_userbased(1:size(train_user_unique_idx,1),:) = X_train_userbased_unique(train_user_unique_idx(1:size(train_user_unique_idx,1)),:); 

X_test_userbased = zeros(size(test_user_unique_idx,1),size(X_test_userbased_unique,2));
X_test_userbased(1:size(test_user_unique_idx,1),:) = X_test_userbased_unique(test_user_unique_idx(1:size(test_user_unique_idx,1)),:); 

X_valid_userbased = zeros(size(valid_user_unique_idx,1),size(X_valid_userbased_unique,2));
X_valid_userbased(1:size(valid_user_unique_idx,1),:) = X_valid_userbased_unique(valid_user_unique_idx(1:size(valid_user_unique_idx,1)),:); 


% Same number of features
feat_num_user = min(size(X_train_userbased,2),size(X_test_userbased,2));
feat_num_user = min(feat_num_user,size(X_valid_userbased,2));
X_train_userbased(:,(feat_num_user+1):end)=[];
X_test_userbased(:,(feat_num_user+1):end)=[];
X_valid_userbased(:,(feat_num_user+1):end)=[];

% 2.1) KNN fit and predict for User-User similarity
score_knn_user_based = zeros(10,1);
for k=1:10
    knn_mdl_user = fitcknn(X_train_userbased,Y_train_userbased,'NumNeighbors',k,'Distance','euclidean','Standardize',0);
    label_result_knn_userbased = predict(knn_mdl_user,X_valid_userbased);
    % Evaluation Error Calculation
    score_knn_user_based(k) = evaluate_labels(valid_user_unique,valid_user_unique_idx,valid_numerical_data,Y_valid_userbased,label_result_knn_userbased);
end;
[~,score_knn_userbased_idx] = max(score_knn_user_based); % K = score_knn_userbased_idx provides the best score over validation set
knn_mdl_user = fitcknn(X_train_userbased,Y_train_userbased,'NumNeighbors',score_knn_userbased_idx,'Distance','euclidean','Standardize',0);
label_result_knn_userbased = predict(knn_mdl_user,X_test_userbased);
% Evaluation Error Calculation
score_knn_user_based = evaluate_labels(test_user_unique,test_user_unique_idx,test_numerical_data,Y_test_userbased,label_result_knn_userbased);


% 2.2) SVM fit and predict for User-User similarity
svm_mdl_user = fitcecoc(X_train_userbased,Y_train_userbased);
label_result_svm_userbased = predict(svm_mdl_user,X_test_userbased);
score_svm_user_based = evaluate_labels(test_user_unique,test_user_unique_idx,test_numerical_data,Y_test_userbased,label_result_svm_userbased);

% 2.3) Ensemble fit and predict for User-User similarity
%ens_bagging_mdl_user = fitensemble(X_train_userbased,Y_train_userbased,'Bag',4,'tree','Type','classification');
ens_boost_mdl_user = fitensemble(X_train_userbased,Y_train_userbased,'AdaBoostM2',4,'tree','Type','classification');
label_result_ens_userbased = predict(ens_boost_mdl_user,X_test_userbased);
score_ensemble_userbased = evaluate_labels(test_user_unique,test_user_unique_idx,test_numerical_data,Y_test_userbased,label_result_ens_userbased);


%% 3) Item-based Collaborative Filtering

% Item-based similarity (Cosine or Pearson)
%X_train_itembased_unique = corr(training_utility_matrix); % matlab built-in, default is Pearson Correlation
%X_test_itembased_unique = corr(testing_utility_matrix); % matlab built-in, default is Pearson Correlation
[~,X_train_itembased_unique] = correlation_calculation(training_utility_matrix',1,0,'pearson'); % type can be Pearson Correlation OR Cosine Correlation
[~,X_test_itembased_unique] = correlation_calculation(testing_utility_matrix',1,0,'pearson'); % type can be Pearson Correlation OR Cosine Correlation
[~,X_valid_itembased_unique] = correlation_calculation(validation_utility_matrix',1,0,'pearson'); % type can be Pearson Correlation OR Cosine Correlation

X_train_itembased = zeros(size(train_song_unique_idx,1),size(X_train_itembased_unique,2));
X_train_itembased(1:size(train_song_unique_idx,1),:) = X_train_itembased_unique(train_song_unique_idx(1:size(train_song_unique_idx,1)),:); 

X_test_itembased = zeros(size(test_song_unique_idx,1),size(X_test_itembased_unique,2));
X_test_itembased(1:size(test_song_unique_idx,1),:) = X_test_itembased_unique(test_song_unique_idx(1:size(test_song_unique_idx,1)),:); 

X_valid_itembased = zeros(size(valid_song_unique_idx,1),size(X_valid_itembased_unique,2));
X_valid_itembased(1:size(valid_song_unique_idx,1),:) = X_valid_itembased_unique(valid_song_unique_idx(1:size(valid_song_unique_idx,1)),:); 

% Same number of features
feat_num_item = min(size(X_train_itembased,2),size(X_test_itembased,2));
feat_num_item = min(feat_num_item,size(X_valid_itembased,2));
X_train_itembased(:,(feat_num_item+1):end)=[];
X_test_itembased(:,(feat_num_item+1):end)=[];
X_valid_itembased(:,(feat_num_item+1):end)=[];         

% 3.1) KNN fit and predict for Item-based similarity
score_knn_item_based = zeros(10,1);
for k = 1:10
    knn_mdl_item = fitcknn(X_train_itembased,Y_train_itembased,'NumNeighbors',k,'Distance','cosine','Standardize',0);
    label_result_knn_itembased = predict(knn_mdl_item,X_valid_itembased);
    % Evaluation Error Calculation
    score_knn_item_based(k) = evaluate_labels(valid_user_unique,valid_user_unique_idx,valid_numerical_data,Y_valid_itembased,label_result_knn_itembased);
end;
[~,score_knn_itembased_idx] = max(score_knn_item_based); % K = score_knn_itembased_idx provides the best score over validation set
knn_mdl_item = fitcknn(X_train_itembased,Y_train_itembased,'NumNeighbors',score_knn_itembased_idx,'Distance','cosine','Standardize',0);
label_result_knn_itembased = predict(knn_mdl_item,X_test_itembased);
% Evaluation Error Calculation
score_knn_item_based = evaluate_labels(test_user_unique,test_user_unique_idx,test_numerical_data,Y_test_itembased,label_result_knn_itembased);

% 3.2) SVM fit and predict for Item-based similarity
svm_mdl_item = fitcecoc(X_train_itembased,Y_train_itembased);
label_result_svm_itembased = predict(svm_mdl_item,X_test_itembased);
score_svm_item_based = evaluate_labels(test_user_unique,test_user_unique_idx,test_numerical_data,Y_test_itembased,label_result_svm_itembased);

% 3.3) Ensemble fit and predict for Item-based similarity
%ens_bagging_mdl_item = fitensemble(X_train_itembased,Y_train_itembased,'Bag',4,'tree','Type','classification');
ens_boost_mdl_item = fitensemble(X_train_itembased,Y_train_itembased,'AdaBoostM2',2,'tree','Type','classification');
label_result_ens_itembased = predict(ens_boost_mdl_item,X_test_itembased);
score_ensemble_itembased = evaluate_labels(test_user_unique,test_user_unique_idx,test_numerical_data,Y_test_itembased,label_result_ens_itembased);

%% 4) Popularity of Songs and Artists -- Weakest Recommender

score_popularity = evaluate_labels_unique(test_user_unique,test_user_unique_idx,Y_test_userbased,test_numerical_data(:,4)); % labels are just the popularity of songs


%% Evaluation results

% Save the scores
save('final_scores','score_ensemble_itembased','score_ensemble_MF','score_ensemble_userbased','score_knn_item_based','score_knn_itembased_idx','score_knn_MF','score_knn_MF_idx','score_knn_user_based','score_knn_userbased_idx','score_popularity','score_svm_item_based','score_svm_MF','score_svm_user_based');

%% Calculate Precision, Recall and Accuracy for each approach

cf_knn_userbased = confusionmat(Y_test_userbased,label_result_knn_userbased);
precision_knn_userbased(1:size(cf_knn_userbased,1))=cf_knn_userbased(1:size(cf_knn_userbased,1),1:size(cf_knn_userbased,1))/sum(cf_knn_userbased(:,1:size(cf_knn_userbased,1)));
Precision_knn_userbased = sum(precision_knn_userbased)/size(cf_knn_userbased,1);
recall_knn_userbased(1:size(cf_knn_userbased,1))=cf_knn_userbased(1:size(cf_knn_userbased,1),1:size(cf_knn_userbased,1))/sum(cf_knn_userbased(1:size(cf_knn_userbased,1),:));
Recall_knn_userbased = sum(recall_knn_userbased)/size(cf_knn_userbased,1);
accuracy_knn_userbased = sum(diag(cf_knn_userbased)) / size(label_result_knn_userbased,1);

cf_knn_itembased = confusionmat(Y_test_itembased,label_result_knn_itembased);
precision_knn_itembased(1:size(cf_knn_itembased,1))=cf_knn_itembased(1:size(cf_knn_itembased,1),1:size(cf_knn_itembased,1))/sum(cf_knn_itembased(:,1:size(cf_knn_itembased,1)));
Precision_knn_itembased = sum(precision_knn_itembased)/size(cf_knn_itembased,1);
recall_knn_itembased(1:size(cf_knn_itembased,1))=cf_knn_itembased(1:size(cf_knn_itembased,1),1:size(cf_knn_itembased,1))/sum(cf_knn_itembased(1:size(cf_knn_itembased,1),:));
Recall_knn_itembased = sum(recall_knn_itembased)/size(cf_knn_itembased,1);
accuracy_knn_itembased = sum(diag(cf_knn_itembased)) / size(label_result_knn_itembased,1);

cf_knn_MF = confusionmat(Y_test_userbased,label_result_knn_MF);
precision_knn_MF(1:size(cf_knn_MF,1))=cf_knn_MF(1:size(cf_knn_MF,1),1:size(cf_knn_MF,1))/sum(cf_knn_MF(:,1:size(cf_knn_MF,1)));
Precision_knn_MF = sum(precision_knn_MF)/size(cf_knn_MF,1);
recall_knn_MF(1:size(cf_knn_MF,1))=cf_knn_MF(1:size(cf_knn_MF,1),1:size(cf_knn_MF,1))/sum(cf_knn_MF(1:size(cf_knn_MF,1),:));
Recall_knn_MF = sum(recall_knn_MF)/size(cf_knn_MF,1);
accuracy_knn_MF = sum(diag(cf_knn_MF)) / size(label_result_knn_MF,1);

% Save the accuracy results from knn classifier of every approach
save('accuracy_results','accuracy_knn_userbased','accuracy_knn_itembased','accuracy_knn_MF');
