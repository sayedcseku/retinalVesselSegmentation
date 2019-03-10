
%Run this file to train Random Forest Classier Model

%please add multi-scales's test.m file in the working directory

%%
%Enter 1 if you want to read features from excel
%Enter 2 to extract features from all training set
prompt = 'Enter Feature Selection mode: ';
x = input(prompt);

%Feature from stored excel files
if x ==1
    disp('Reading stored training samples...');
    CDB1_Vessel1 = xlsread('features\CHASEDB1_vessel1_H128.xlsx');
    CDB1_Vessel2 = xlsread('features\CHASEDB1_vessel2_H128.xlsx');
    CDB1_Non_Vessel1 = xlsread('features\CHASEDB1_non_vessel_H128.xlsx');
    
    STARE_Vessel1 = xlsread('features\STARE_vessel1_H128.xlsx');
    STARE_Vessel2 = xlsread('features\STARE_vessel2_H128.xlsx');
    STARE_Non_Vessel1 = xlsread('features\STARE_non_vessel_H128.xlsx');

    Drive_Vessel1 = xlsread('features\DRIVE_vessel1_H128.xlsx');
    Drive_Vessel2 = xlsread('features\DRIVE_vessel2_H128.xlsx');
    Drive_Non_Vessel1 = xlsread('features\DRIVE_non_vessel_H128.xlsx');
    
    
elseif x==2
      
        %first generate multi-scale segmented image for all training sets
        datasets = {'DRIVE','STARE','CHASEDB1'};
        extensions = {'\*.tif','\*.ppm','\*.jpg'};
        extensionsMask = {'\*.gif','\*.png','\*.jpg'};
        path = 'Images\RFC SET';
        for dset =1:length(datasets)
            
            fTableNon = [];
            fTableVes = [];
            
            %file information for fundus images, mask for multi-scale
            %and mask for proposed model specific mask
            files = dir (fullfile(path,datasets{dset},'train',extensions{dset}));
            files1 = dir (fullfile(path,datasets{dset},'multiscale_mask','\train',extensionsMask{dset}));
            files2 = dir (fullfile(path,datasets{dset},'rfc_mask','\*.png'));
            
            for i = 1:length(files)
                
                %%
                %multi-scale line operation
                img=imread(fullfile(files(i).folder,'\',files(i).name));
                mask = imread(fullfile(files1(i).folder,'\',files1(i).name));
                
                mSegmentedImg = multi_test(img,mask);
                fName = ['segmented_',strtok(files(i).name,'.'),'.png'];
                imwrite(mSegmentedImg,fullfile(path,datasets{dset},'train\multiscale_output',fName));
                %%
                
                %%
                %feature extraction
                rfc_mask = imread(fullfile(files2(i).folder,files2(i).name));
                
                %Features extraction from the segmented image
                %Ves tupples are true vessel pixel labeled features
                %nonVes tupples are false vessel pixel labeled features
                [Ves,nonVes] = extractFeatureH(img,mSegmentedImg,rfc_mask);
                fTableVes = [fTableVes; Ves];
                fTableNon = [fTableNon; nonVes];
                %%
            end
            
            %dividing the large table for storing purpose
            fTableVes1 = fTableVes(1:end/2,:);
            fTableVes2 = fTableVes(end/2+1:end,:);
            
            if dset==1
                
                Drive_Vessel1 = fTableVes1;
                Drive_Vessel2 = fTableVes2;
                Drive_Non_Vessel1 = fTableNon;
                
            elseif dset==2
                
                STARE_Vessel1 = fTableVes1;
                STARE_Vessel2 = fTableVes2;
                STARE_Non_Vessel1 = fTableNon;
            elseif dset==3
                
                CDB1_Vessel1 = fTableVes1;
                CDB1_Vessel2 = fTableVes2;
                CDB1_Non_Vessel1 = fTableNon;
            end
            
            %%
            %storing for future training purpose
            
            fName1 = [datasets{dset},'_vessel1_H128','.xlsx'];
            fName2 = [datasets{dset},'_vessel2_H128','.xlsx'];
            fName3 = [datasets{dset},'_non_vessel_H128','.xlsx'];
            xlswrite(fName1,fTableVes1);
            xlswrite(fName2,fTableVes2);
            xlswrite(fName3,fTableNon);
            %%
        end
        
        
end
%%

%%
%combining all features before training

%combining features vectors with vessel/'1' label
    fTableVes = [STARE_Vessel1;
        STARE_Vessel2;
        Drive_Vessel1;
        Drive_Vessel2;
        CDB1_Vessel1;
        CDB1_Vessel2];

    %keeping all the non-vessel/'0' feature vectors and suppose they are
    % 40% of whole training set
    
    %selecting other 60% of training set from vessel/'1' labeled feature
    %vectors randomly 
    nonVSize = length(STARE_Non_Vessel1)+length(Drive_Non_Vessel1)+length(CDB1_Non_Vessel1);
    VSize = (nonVSize * 0.6)/0.4;
    VSize = round(VSize);

    index = randsample(1:length(fTableVes), VSize);
    bar = fTableVes(index,:);

    %60% vessel pixels and 40% non-vessel pixels
    fTable = [bar;
        STARE_Non_Vessel1;
        Drive_Non_Vessel1;
        CDB1_Non_Vessel1];

    % suffle rows randomly before training
    %fTable_x = fTable(randperm(size(fTable, 1)), :);
%%

%%
    %training the Random Forest Classifier
    %first 64 elemnts of feature vector are feature against 65th
    %element/class label
    rng(1); % For reproducibility
    Mdl = TreeBagger(50,fTable(:,(1:128)),fTable(:,129),'OOBPrediction','On',...
        'Method','classification')

    view(Mdl.Trees{1},'Mode','graph')
    
    %assignin('base','trainedModel',Mdl)
    save('trainedModel_H_128')
%%
