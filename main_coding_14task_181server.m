function main_coding_14task_181server
%% main function 

load num_contrast_1112
label = num_con;    
%% dividing subject every 100 subjects
for i = 1:10
    if i == 1
        rankran = randperm(size(label,2));
        label = label(rankran);
        save label_random_14task_987 label
    else
        load label_random_14task_987
    end
    if i==10
        [fortrain_test,label_for] = get_volume_14task_181server(label);
        label_f = 987;
    else
        [fortrain_test,label_for] = get_volume_14task_181server(label(1:100*i));
        label_f = 100*i;
    end

    %% train and test data
    for k = 1:10    %repetition
        fortrain = fortrain_test;
        forlabel = label_for;
        test_data = cell2mat(fortrain_test(k));
        test_label = cell2mat(label_for(k));
        fortrain(k) = [];
        forlabel(k) = [];
        train_data = [];
        train_label = [];
        for l = 1:9
            train_data = [train_data;cell2mat(fortrain(l))];
            train_label = [train_label,cell2mat(forlabel(l))];
        end
        train_contrast_14task_181server(train_data,train_label,test_data,test_label,k,label_f);
    end
end
        
            
        
        





