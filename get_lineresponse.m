function R = get_lineresponse(img,W,L)
% img: extended inverted gc
% W: window size, L: line length
% R: line detector response

avgmask = fspecial('average',W);
avgresponse = imfilter(img,avgmask,'replicate');

maxlinestrength = -Inf*ones(size(img));
for theta = 0:15:165
    linemask = get_linemask(theta,L);
    linemask = linemask/sum(linemask(:));
    imglinestrength = imfilter(img,linemask);    
    imglinestrength = imglinestrength - avgresponse;    
    maxlinestrength = max(maxlinestrength,imglinestrength);    
end
R = maxlinestrength;
end


