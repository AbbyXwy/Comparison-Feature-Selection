function train_contrast_14task_181server(train_data,train_label,test_data,test_label,j,N_of_sub)
per = 0.02:0.02:0.1;
for m = 1:5                    %percentage of features
    varia_order = calculate_anova(train_data,train_label,m,j,N_of_sub);
    w_order = calculate_kendall(train_data,train_label,m,j,N_of_sub);
    toll = size(train_data,2);
    train_data_anova = train_data(:,varia_order(1:toll*per(m)));
    test_data_anova = test_data(:,varia_order(1:toll*per(m)));
    train_data_kendall = train_data(:,w_order(1:toll*per(m)));
    test_data_kendall = test_data(:,w_order(1:toll*per(m)));
    anova = softmax_ten(train_data_anova, train_label, test_data_anova, test_label);
    acc_soft_anova(m) = anova;
    kendall = softmax_ten(train_data_kendall, train_label, test_data_kendall, test_label);
    acc_soft_kendall(m) = kendall;
end
    string = ['hcp_14task_mean_' num2str(N_of_sub)];
    cd(string)
    str2 = ['accuracy_anova_10fold_hcp_repetition' num2str(j) '_14task'];    %including per2-per10
    acc = acc_soft_anova;
    save(str2,'acc')            
    str2 = ['accuracy_kendall_10fold_hcp_repetition' num2str(j) '_14task'];    %%including per2-per10
    acc = acc_soft_kendall;
    save(str2,'acc')
end
    
function [b] = calculate_anova(train_data,train_label,percentage,internal,N_of_sub)
varia = [];    
mid = (train_label==1);
train_emotion1 = train_data(mid==1,:);
mid = (train_label==2);
train_emotion2 = train_data(mid==1,:);
mid = (train_label==3);
train_gambling1 = train_data(mid==1,:);
mid = (train_label==4);
train_gambling2 = train_data(mid==1,:);
mid = (train_label==5);
train_language1 = train_data(mid==1,:);
mid = (train_label==6);
train_language2 = train_data(mid==1,:);
mid = (train_label==7);
train_social1 = train_data(mid==1,:);
mid = (train_label==8);
train_social2 = train_data(mid==1,:);
mid = (train_label==9);
train_relational1 = train_data(mid==1,:);
mid = (train_label==10);
train_relational2 = train_data(mid==1,:);
mid = (train_label==11);
train_wm1 = train_data(mid==1,:);
mid = (train_label==12);
train_wm2 = train_data(mid==1,:);
mid = (train_label==13);
train_wm3 = train_data(mid==1,:);
mid = (train_label==14);
train_wm4 = train_data(mid==1,:);
for k = 1:59412
    pre = [train_emotion1(:,k),train_emotion2(:,k),train_gambling1(:,k),train_gambling2(:,k),train_language1(:,k),train_language2(:,k),train_social1(:,k),train_social2(:,k),train_relational1(:,k),train_relational2(:,k),train_wm1(:,k),train_wm2(:,k),train_wm3(:,k),train_wm4(:,k)];
    [P,tb] = anova1(pre,[],'nodisplay');
    varia = [varia,cell2mat(tb(2,5))];
end
[~,b] = sort(varia,'descend');
str = ['hcp_features_anova_percentage' num2str(percentage*2) '_internal' num2str(internal)];
str2 = ['hcp_14task_mean_' num2str(N_of_sub)];
cd(str2)
save(str, 'varia', 'b')
end

function [b] = calculate_kendall(train_data,train_label,percentage,internal,N_of_sub)
mid = (train_label==1);
train_emotion1 = train_data(mid==1,:);
mid = (train_label==2);
train_emotion2 = train_data(mid==1,:);
mid = (train_label==3);
train_gambling1 = train_data(mid==1,:);
mid = (train_label==4);
train_gambling2 = train_data(mid==1,:);
mid = (train_label==5);
train_language1 = train_data(mid==1,:);
mid = (train_label==6);
train_language2 = train_data(mid==1,:);
mid = (train_label==7);
train_social1 = train_data(mid==1,:);
mid = (train_label==8);
train_social2 = train_data(mid==1,:);
mid = (train_label==9);
train_relational1 = train_data(mid==1,:);
mid = (train_label==10);
train_relational2 = train_data(mid==1,:);
mid = (train_label==11);
train_wm1 = train_data(mid==1,:);
mid = (train_label==12);
train_wm2 = train_data(mid==1,:);
mid = (train_label==13);
train_wm3 = train_data(mid==1,:);
mid = (train_label==14);
train_wm4 = train_data(mid==1,:);
for k = 1:59412
    pre = [train_emotion1(:,k),train_emotion2(:,k),train_gambling1(:,k),train_gambling2(:,k),train_language1(:,k),train_language2(:,k),train_social1(:,k),train_social2(:,k),train_relational1(:,k),train_relational2(:,k),train_wm1(:,k),train_wm2(:,k),train_wm3(:,k),train_wm4(:,k)];
    n_task = 14;
    Ri = sum(pre)/size(train_data,1);
    R = sum(Ri)/14.*ones(1,14); 
    SSR = sum((R-Ri).^2);
    w_ori(k) = 12*SSR/(size(train_data,1)^2 * (n_task^3-n_task));
end
    w_ori = w_ori*size(train_data,1)*(n_task-1)/n_task*1e7;
    %w(k) = log(w_ori(k));

[~,b] = sort(w_ori,'descend');
str = ['hcp_features_kendall_percentage' num2str(percentage*2) '_internal' num2str(internal)];
str2 = ['hcp_14task_mean_' num2str(N_of_sub)];
cd(str2)
save(str,'b','w_ori')
end

function [acc] = softmax_ten(traindata, trainlabel, testdata, testlabel)
order = size(trainlabel,2);
order = randperm(order);
trainData = traindata(order,:);
trainLabel = trainlabel(order);
ord = randperm(size(testlabel,2));
testLabel = testlabel(ord);
testData = testdata(ord,:);
cd('softmax')
inputSize = size(trainData,2);
lambda = 1e-4; % Weight decay parameter 
numClasses = 14;
inputData = trainData;
theta = 0.005 * randn(numClasses * inputSize, 1);   

[cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData', trainLabel);                                    
options.maxIter = 200;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData', trainLabel, options);
parameter = softmaxModel.optTheta;               

inputData = testData;
[pred] = softmaxPredict(softmaxModel, inputData');
acc = mean(testLabel(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc* 100);    
end
 
