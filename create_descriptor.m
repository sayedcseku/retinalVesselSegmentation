function [interest_points, descriptor] = create_descriptor( image,imgMsk,patch_size )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
image = rgb2gray(image); %if image is in RGB format
img=integralImage(image);
[M,N]=size(img);

BW_img = im2bw(imgMsk);
[row, col]=find(BW_img==1);

localidx = [row,col];
    
patch=patch_size;
imdim = M*N + 1;
C=patch_size/2;
i=0;
%interest_points = localidx;
for k=1:length(localidx)  
    if((localidx(k,1)-(C-1)>0)&(localidx(k,2)-(C-1)>0))& (((localidx(k,1)+C)<M)&((localidx(k,2)+C)<N))
        i=i+1;
        interest_points(i,1:2)= localidx(k,1:2);
        descriptor(i,:)=create_binary(localidx(k,1),localidx(k,2),img,patch,1)/511;      
    end  
end

end

