function [fTableVes, fTableNon] =  extractFeatureH(img,mSegmentedImg,imgMsk)

    fTableNon = [];
    fTableVes = [];


    %img=imread('Images\MultiScale\CHASEDB1\Image_03R_segmented.png');
    %imgMsk=imread('Images\masks\CHASEDB1\Image_03R_Mask.png');
    imgMsk = im2bw(imgMsk);
    new_mask = mSegmentedImg-imgMsk;


    %imgGray=rgb2gray(img);
    patchSize = 32;
    [interest_points, featuresH] = create_descriptor(img,imgMsk,patchSize);

    label_f=[];
    for i=1:length(interest_points)
        label_f(i,1)= 0;

    end
    featuresH = [featuresH label_f];

    fTableNon = [fTableNon;featuresH];

    featuresH = [];
    [interest_points1, featuresH] = create_descriptor(img,new_mask,patchSize);
    
    label_f=[];
    for i=1:length(interest_points1)
        label_f(i,1)= 1;
    end
    featuresH = [featuresH label_f];

    fTableVes = [fTableVes;featuresH];
end