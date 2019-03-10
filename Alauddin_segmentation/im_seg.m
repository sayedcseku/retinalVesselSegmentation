function segmentedimg = im_seg(img,mask,W)
% img: original color image
% mask: mask of FOV
% W: window size which is chosen as around double of average vessel width
img = im2double(img);
mask = im2bw(mask);

img = 1-img(:,:,2);
%imshow(img);

img = fakepad(img,mask);

%imshow(img);

features = standardize(img,mask);
Ls = 1:2:W;
for j = 1:numel(Ls)
   L = Ls(j);  
   R = get_lineresponse(img,W,L); 
   R = standardize(R,mask);
   features = features+R;
   %disp(features);
   disp(['L = ',num2str(L),' finished!']);       
end     
segmentedimg = features/(1+numel(Ls));
segmentedimg = im2bw(segmentedimg,graythresh(segmentedimg));

end