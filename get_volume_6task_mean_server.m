function [fortrain_test,label_for] = get_volume_6task_mean_server(label)
%% a cell structure inluding 10 portions (fortrain_test)
n_sub = size(label,2);
n_div = floor(n_sub/10);
%n_div = n_sub/10;
fortrain_test = {};
label_for = {};
%cd('C:\Users\xwy\Desktop\HCP')
load emotion_hcp
emotion_hcp = emotion_hcp(1:64984);
miss = ismissing(emotion_hcp);
order_omit = find(miss(1,:)==1);
clear emotion_hcp

for i = 1:10
    label_div = [];
    div_one = [];
%% emotion  
    if i==10
    for index=(i-1)*n_div+1:n_sub
                for m = 1:2
                    str=['emotion-new/sub' num2str(label(index)) '_cope' num2str(m) '.dtseries.nii'];
                if exist(str)~=0
                    data = ft_read_cifti(str);
                    data_using=data.dtseries(1:64984,:);
                    [data,vector] = rmmissing(data_using,'MinNumMissing',size(data_using,2));
                    div_one = [div_one;data_using'];
                    label_div = [label_div,1];
                else
                    break
                end
                end
                %gambling
                for m = 1:2
                    str=['gambling-new/sub' num2str(label(index)) '_cope' num2str(m) '.dtseries.nii'];
                if exist(str)~=0
                    data = ft_read_cifti(str);
                    data_using=data.dtseries(1:64984,:);
                    [data,vector] = rmmissing(data_using,'MinNumMissing',size(data_using,2));
                    div_one = [div_one;data_using'];
                    label_div = [label_div,2];
                else
                    break
                end
                end
                %language
                for m = 1:2
                    str=['language-new/sub' num2str(label(index)) '_cope' num2str(m) '.dtseries.nii'];
                if exist(str)~=0
                    data = ft_read_cifti(str);
                    data_using=data.dtseries(1:64984,:);
                    [data,vector] = rmmissing(data_using,'MinNumMissing',size(data_using,2));
                    div_one = [div_one;data_using'];
                    label_div = [label_div,3];
                else
                    break
                end
                end
                %social
                for m = 1:2
                    str=['social-new/sub' num2str(label(index)) '_cope' num2str(m) '.dtseries.nii'];
                if exist(str)~=0
                    data = ft_read_cifti(str);
                    data_using=data.dtseries(1:64984,:);
                    [data,vector] = rmmissing(data_using,'MinNumMissing',size(data_using,2));
                    div_one = [div_one;data_using'];
                    label_div = [label_div,4];
                else
                    break
                end
                end
                %relational
                for m = 1:2
                    str=['relational-new/sub' num2str(label(index)) '_cope' num2str(m) '.dtseries.nii'];
                if exist(str)~=0
                    data = ft_read_cifti(str);
                    data_using=data.dtseries(1:64984,:);
                    [data,vector] = rmmissing(data_using,'MinNumMissing',size(data_using,2));
                    div_one = [div_one;data_using'];
                    label_div = [label_div,5];
                else
                    break
                end
                end
                %wm
                for m = 1:4
                    str=['wm-new/sub' num2str(label(index)) '_cope' num2str(14+m) '.dtseries.nii'];
                if exist(str)~=0
                    data = ft_read_cifti(str);
                    data_using=data.dtseries(1:64984,:);
                    [data,vector] = rmmissing(data_using,'MinNumMissing',size(data_using,2));
                    div_one = [div_one;data_using'];
                    label_div = [label_div,6];
                else
                    break
                end     
                end    
    end
    else
    %emotion   
    for index=(i-1)*n_div+1:i*n_div
                for m = 1:2
                    str=['emotion-new/sub' num2str(label(index)) '_cope' num2str(m) '.dtseries.nii'];
                if exist(str)~=0
                    data = ft_read_cifti(str);
                    data_using=data.dtseries(1:64984,:);
                    [data,vector] = rmmissing(data_using,'MinNumMissing',size(data_using,2));
                    div_one = [div_one;data_using'];
                    label_div = [label_div,1];
                else
                    break
                end
                end
    %gambling
                for m = 1:2
                    str=['gambling-new/sub' num2str(label(index)) '_cope' num2str(m) '.dtseries.nii'];
                if exist(str)~=0
                    data = ft_read_cifti(str);
                    data_using=data.dtseries(1:64984,:);
                    [data,vector] = rmmissing(data_using,'MinNumMissing',size(data_using,2));
                    div_one = [div_one;data_using'];
                    label_div = [label_div,2];
                else
                    break
                end
                end
        %language
                for m = 1:2
                    str=['language-new/sub' num2str(label(index)) '_cope' num2str(m) '.dtseries.nii'];
                if exist(str)~=0
                    data = ft_read_cifti(str);
                    data_using=data.dtseries(1:64984,:);
                    [data,vector] = rmmissing(data_using,'MinNumMissing',size(data_using,2));
                    div_one = [div_one;data_using'];
                    label_div = [label_div,3];
                else
                    break
                end
                end
    %social
                for m = 1:2
                    str=['social-new/sub' num2str(label(index)) '_cope' num2str(m) '.dtseries.nii'];
                if exist(str)~=0
                    data = ft_read_cifti(str);
                    data_using=data.dtseries(1:64984,:);
                    [data,vector] = rmmissing(data_using,'MinNumMissing',size(data_using,2));
                    div_one = [div_one;data_using'];
                    label_div = [label_div,4];
                else
                    break
                end
                end
    %relational
                for m = 1:2
                    str=['relational-new/sub' num2str(label(index)) '_cope' num2str(m) '.dtseries.nii'];
                if exist(str)~=0
                    data = ft_read_cifti(str);
                    data_using=data.dtseries(1:64984,:);
                    [data,vector] = rmmissing(data_using,'MinNumMissing',size(data_using,2));
                    div_one = [div_one;data_using'];
                    label_div = [label_div,5];
                else
                    break
                end
                end
        %wm
                for m = 1:4
                    str=['wm-new/sub' num2str(label(index)) '_cope' num2str(14+m) '.dtseries.nii'];
                if exist(str)~=0
                    data = ft_read_cifti(str);
                    data_using=data.dtseries(1:64984,:);
                    [data,vector] = rmmissing(data_using,'MinNumMissing',size(data_using,2));
                    div_one = [div_one;data_using'];
                    label_div = [label_div,6];
                else
                    break
                end     
                end    
    end
    end
    div_one(:,order_omit) = [];
    for j = 1:n_div
        ansi = min(div_one(j,:));
        ansx = max(div_one(j,:));
        div_one(j,:) = (div_one(j,:)-ansi)./(ansx-ansi);
    end
    fortrain_test = [fortrain_test,mat2cell(div_one,size(div_one,1),size(div_one,2))];
    label_for = [label_for,mat2cell(label_div,size(label_div,1),size(label_div,2))];
end

        
        