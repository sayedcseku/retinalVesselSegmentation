clear all
clc
fTableNon = [];
fTableVes = [];

img = imread('Images\RFC SET\CHASEDB1\train\Image_03R.jpg');
mSegmentedImg=imread('Images\RFC SET\CHASEDB1\train\multiscale_output\Image_03R_segmented.png');
rfc_mask=imread('Images\RFC SET\CHASEDB1\rfc_mask\Image_03R_Mask.png');
new_mask = mSegmentedImg-rfc_mask;

% surf default parameters
%Options.upright=true;
%Options.tresh=0.000001;



%BW_img = im2bw(imgMsk);
%[row, col]=find(BW_img==1);
figure,imshow(img,[])
figure,imshow(imgMsk,[])

%imgGray=rgb2gray(img);
[Ves,nonVes] = extractFeatureH(img,mSegmentedImg,rfc_mask);
fTableVes = [fTableVes; Ves];
fTableNon = [fTableNon; nonVes];
%fTable = readtable('SURF_feature.xlsx');
%fTable = table2array(fTable);
%featuresH = [fTable;featuresH];

fTableVes1 = fTableVes(1:end/2,:);
fTableVes2 = fTableVes(end/2+1:end,:);
xlswrite('CDB1_SURF_feature_vessel1.xlsx',fTableVes1);
xlswrite('CDB1_SURF_feature_vessel2.xlsx',fTableVes2);
xlswrite('CDB1_SURF_feature_non_vessel.xlsx',fTableNon);

% i will need this code to integrate large fTableVes excel reads
%fTableVes = [fTableVes1;fTableVes2];

STARE_Vessel1 = xlsread('STARE_SURF_feature_vessel1.xlsx');
STARE_Vessel2 = xlsread('STARE_SURF_feature_vessel2.xlsx');
STARE_Non_Vessel1 = xlsread('STARE_SURF_feature_non_vessel.xlsx');

Drive_Vessel1 = xlsread('SURF_feature_vessel1.xlsx');
Drive_Vessel2 = xlsread('SURF_feature_vessel2.xlsx');
Drive_Non_Vessel1 = xlsread('SURF_feature1_non_vessel.xlsx');

fTableVes =     [STARE_Vessel1;
               STARE_Vessel2;
               Drive_Vessel1;
               Drive_Vessel2;
               fTableVes];

nonVSize = length(STARE_Non_Vessel1)+length(Drive_Non_Vessel1)+length(fTableNon);
VSize = (nonVSize * 0.6)/0.4;
VSize = round(VSize);

index = randsample(1:length(fTableVes), VSize);
bar = fTableVes(index,:);

%60% vessel pixels and 40% non-vessel pixels
fTable = [bar;
            STARE_Non_Vessel1;
            Drive_Non_Vessel1;
            fTableNon];

% suffle rows randomly before training 
%fTable_x = fTable(randperm(size(fTable, 1)), :);

rng(1); % For reproducibility
Mdl = TreeBagger(50,fTable(:,(1:64)),fTable(:,65),'OOBPrediction','On',...
    'Method','classification')

view(Mdl.Trees{1},'Mode','graph')

img = imread('Images\RFC SET\DRIVE\test\01_test.tif');
imgT = imread('Images\MultiScale\test\drive_01_test_segmented_pp.png');
[row col] =size(imgT);
new_img1=zeros(row,col);
feat = [];
%Ipts1=OpenSurf_Sheen(imgT,Options,imgT);
[interest_points, featuresH]=create_descriptor(img,imgT,32);

%for i=1:length(interest_points)
   % feat(i,:) = Ipts1(1,i).descriptor;
   % class=Mdl.predict(feat(i,:));
    %new_img1(Ipts1(1,i).y,Ipts1(1,i).x) = str2double(class);

%end

class=Mdl.predict(featuresH);

for i=1:length(interest_points)
    
    new_img1(interest_points(i,1),interest_points(i,2)) = str2double(class{i});
    
end
figure; imshow(new_img1);

imwrite(new_img1,'E:\Thesis\Code\Code\Images\RFC Output\DRIVE\13_test_segmented_RFC.png');

save('Mdl')

%test purpose, not important
newimg = zeros(row,col)
for i=1:row
    for j=1:col
        newimg(i,j)=new_img1(j,i);
    end
end
%test purpose, not important
for i=1:row
    for j=1:col
        if(imgT(i,j) == 255)
            loc=[double(i) double(j)];
            points=SURFPoints(loc);
            
            features=extractFeatures(imgT,points);
            class=Mdl.predict(features);
            if str2double(class)== 1.0
                new_img1(i,j)=1;
            end
        end
    end
end
%new_img1 = imrotate(new_img1,90);
figure; imshow(new_img1);