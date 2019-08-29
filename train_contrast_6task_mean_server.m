function train_contrast_6task_mean_server(train_data,train_label,test_data,test_label,j,N_of_sub)
per = 0.02:0.02:0.1;
for m = 1:5                      %percentage of features
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
    string = ['hcp_6task_mean_' num2str(N_of_sub)];
    cd(string)
    str2 = ['accuracy_anova_10fold_hcp_repetition' num2str(j) '_6task'];    %including per2-per10
    acc = acc_soft_anova;
    save(str2,'acc')            
    str2 = ['accuracy_kendall_10fold_hcp_repetition' num2str(j) '_6task'];    %%including per2-per10
    acc = acc_soft_kendall;
    save(str2,'acc')
end
    
function [b] = calculate_anova(train_data,train_label,percentage,internal,N_of_sub)
varia = [];    
mid = (train_label==1);
train_emotion = train_data(mid==1,:);
mid = (train_label==2);
train_gambling = train_data(mid==1,:);
mid = (train_label==3);
train_language = train_data(mid==1,:);
mid = (train_label==4);
train_social = train_data(mid==1,:);
mid = (train_label==5);
train_relational = train_data(mid==1,:);
mid = (train_label==6);
train_wm = train_data(mid==1,:);

for k = 1:59412
    na = size(train_emotion,1);
    ma = size(train_wm,1);
    pre = [train_emotion(:,k);train_gambling(:,k);train_language(:,k);train_social(:,k);train_relational(:,k);train_wm(:,k)];
    group = [ones(1,na) 2*ones(1,na) 3*ones(1,na) 4*ones(1,na) 5*ones(1,na) 6*ones(1,ma)];
    [P,tb] = anova1(pre,group,'nodisplay');
    varia = [varia,cell2mat(tb(2,5))];
end
[~,b] = sort(log(varia),'descend');
str = ['hcp_features_anova_percentage' num2str(percentage*2) '_internal' num2str(internal)];
str2 = ['hcp_6task_mean_' num2str(N_of_sub)];
cd(str2)
save(str, 'varia', 'b')
end

function [b] = calculate_kendall(train_data,train_label,percentage,internal,N_of_sub)
mid = (train_label==1);
train_labelmid = zeros(size(train_label));
x = find(mid==1);
for i = 1:size(x,2)/2   
    x = find(mid==1);
    train_emotion1 = train_data(x(2*i-1),:);
    train_emotion2 = train_data(x(2*i),:);
    train_emotion(i,:) = mean([train_emotion1;train_emotion2]);
end
mid = (train_label==2);
x = find(mid==1);
for i = 1:size(x,2)/2
    x = find(mid==1);
    train_gambling1 = train_data(x(2*i-1),:);
    train_gambling2 = train_data(x(2*i),:);
    train_gambling(i,:) = mean([train_gambling1;train_gambling2]);
end
mid = (train_label==3);
x = find(mid==1);
for i = 1:size(x,2)/2
    x = find(mid==1);
    train_language1 = train_data(x(2*i-1),:);
    train_language2 = train_data(x(2*i),:);
    train_language(i,:) = mean([train_language1;train_language2]);
end
mid = (train_label==4);
x = find(mid==1);
for i = 1:size(x,2)/2
    x = find(mid==1);
    train_social1 = train_data(x(2*i-1),:);
    train_social2 = train_data(x(2*i),:); 
    train_social(i,:) = mean([train_social1;train_social2]);
end
mid = (train_label==5);
x = find(mid==1);
for i = 1:size(x,2)/2
    x = find(mid==1);
    train_relational1 = train_data(x(2*i-1),:);
    train_relational2 = train_data(x(2*i),:);
    train_relational(i,:) = mean([train_relational1;train_relational2]);
end
mid = (train_label==6);
x = find(mid==1);
for i = 1:size(x,2)/4
    x = find(mid==1);
    train_wm1 = train_data(x(4*i-3),:); 
    train_wm2 = train_data(x(4*i-2),:);
    train_wm3 = train_data(x(4*i-1),:);
    train_wm4 = train_data(x(4*i),:);
    train_wm(i,:) = mean([train_wm1;train_wm2;train_wm3;train_wm4]);
end

for k = 1:59412
    pre = [train_emotion(:,k),train_gambling(:,k),train_language(:,k),train_social(:,k),train_relational(:,k),train_wm(:,k)];
    n_task = 6;
    Ri = sum(pre)/size(train_data,1);
    R = sum(Ri)/6.*ones(1,6); 
    SSR = sum((R-Ri).^2);
    w_ori(k) = 12*SSR/(size(train_data,1)^2 * (n_task^3-n_task));
end
    w_ori = w_ori*size(train_data,1)*(n_task-1)/n_task*1e7;
    %w(k) = log(w_ori(k));

[~,b] = sort(w_ori,'descend');
str = ['hcp_features_kendall_percentage' num2str(percentage*2) '_internal' num2str(internal)];
str2 = ['hcp_6task_mean_' num2str(N_of_sub)];
cd(str2)
save(str, 'b','w_ori')
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
numClasses = 6;
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
 
