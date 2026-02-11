% Customized by Sajib (on 12/Nov/2015)
% to take image to segment and mask as inputs
function segimg=VesselSegment(img,mask)

W = 15;
segmentedimg = im_seg(img,mask,W);
%imwrite(segmentedimg,'test_images\drive_01_test_segmented.png');
noisesize = 100;
ppimg = noisefiltering(segmentedimg,noisesize);
segimg = ppimg;
%imwrite(ppimg,'test_images\drive_01_test_segmented_pp.png');
end