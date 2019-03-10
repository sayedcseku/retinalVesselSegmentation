function [fTableVes, fTableNon] =  extractFeature(img,imgMsk)

    fTableNon = [];
    fTableVes = [];


    %img=imread('Images\MultiScale\CHASEDB1\Image_03R_segmented.png');
    %imgMsk=imread('Images\masks\CHASEDB1\Image_03R_Mask.png');
    imgMsk = im2bw(imgMsk);
    new_mask = img-imgMsk;

    Options.upright=true;
    Options.tresh=0.000001;



    %BW_img = im2bw(imgMsk);
    %[row, col]=find(BW_img==1);
    %figure,imshow(img,[])
    %figure,imshow(imgMsk,[])

    %imgGray=rgb2gray(img);
    Ipts1=OpenSurf_Sheen(img,Options,imgMsk);
    t=1;

    featuresH = [];
    label_f=[];
    for i=1:length(Ipts1)
        featuresH(i,:) = Ipts1(1,i).descriptor;
        label_f(i,1)= 0;

    end
    featuresH = [featuresH label_f];

    fTableNon = [fTableNon;featuresH];

    Ipts1=OpenSurf_Sheen(img,Options,new_mask);


    featuresH = [];
    label_f=[];
    for i=1:length(Ipts1)
        featuresH(i,:) = Ipts1(1,i).descriptor;
        label_f(i,1)= 1;

    end
    featuresH = [featuresH label_f];

    fTableVes = [fTableVes;featuresH];
end