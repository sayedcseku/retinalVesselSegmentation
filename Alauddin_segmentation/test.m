function mSegmentedImg =  test(img,mask)
%img = imread('test_images\drive_01_test.png');
%mask = imread('test_images\drive_01_test_mask.png');

%img = imread('E:\Thesis\Code\Code\Images\CHASEDB1\Image_14R.jpg');
%mask = imread('E:\Thesis\Code\Code\Images\CHASEDB1\CHASEDB1_mask\Image_14R.jpg');



W = 15;
segmentedimg = im_seg(img,mask,W);
%imwrite(segmentedimg,'test_images\drive_01_test_segmented.png');
%imwrite(segmentedimg,'E:\Thesis\Code\Code\Images\MultiScale\CHASEDB1\Image_14R_segmented_n.png');
noisesize = 100;
ppimg = noisefiltering(segmentedimg,noisesize);
%imwrite(ppimg,'test_images\drive_01_test_segmented_pp.png');
%imwrite(ppimg,'E:\Thesis\Code\Code\Images\MultiScale\CHASEDB1\Image_14R_segmented.png');
%imshow(ppimg)
mSegmentedImg = ppimg;
end