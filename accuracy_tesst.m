k=1;
s_accuracy=0.0;
s_sensitivity = 0.0;
s_specificity = 0.0;
s_AUC = 0.0;
    
%rfcImg=imread('E:\Thesis\Code\Code\Images\RFC Output\CHASEDB1\Image_11R_segmented_RFC.png');
rfcImg = imread('E:\Thesis\Code\Code\Images\MultiScale\test\drive_20_test_segmented_pp.png');

groundImg = imread('E:\Thesis\Code\Code\Images\DRIVE\DRIVE\test\2nd_manual\20_manual2.gif');
TP=0;
FN=0;
FP=0;
TN=0;

[row,col]=size(groundImg);

for i=1:row
    for j=1:col
        
        if groundImg(i,j) >0 
            if rfcImg(i,j)>0
                TP=TP+1;
            else
                FN=FN+1;
            end
        else
            if rfcImg(i,j)>0
                FP=FP+1;
            else
                TN=TN+1;
            end
        end
        
    end
end

accuracy = (TP+TN)/(TP+TN+FP+FN)
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
AUC = (sensitivity+specificity)/2

s_accuracy = s_accuracy+accuracy;
s_sensitivity = s_sensitivity+sensitivity;
s_specificity = s_specificity+specificity;
s_AUC = s_AUC+AUC;


str = string(k)+'_test';
final_res{k}.name = str;
final_res{k}.dataset = 'DRIVE';
final_res{k}.accuracy =accuracy;
final_res{k}.sensitivity =sensitivity;
final_res{k}.specificity =specificity;
final_res{k}.AUC =AUC;
final_res{k}.TP =TP;
final_res{k}.FN =FN;
final_res{k}.FP =FP;
final_res{k}.TN =TN;
final_res{k}.remark = 'FV';
k=k+1;


resT = cell2mat(final_res);

writetable(struct2table(resT), 'Result_MultiScale_081218.xlsx');

accuracyCDB = (final_res{1}.accuracy+final_res{2}.accuracy+final_res{3}.accuracy+final_res{4}.accuracy)/4
accuracyST = (final_res{5}.accuracy+final_res{6}.accuracy+final_res{7}.accuracy+final_res{8}.accuracy)/4
accuracyDR = (final_res{9}.accuracy+final_res{10}.accuracy+final_res{11}.accuracy+final_res{12}.accuracy)/4

sensitivityCDB = (final_res{1}.sensitivity+final_res{2}.sensitivity+final_res{3}.sensitivity+final_res{4}.sensitivity)/4
sensitivityST = (final_res{5}.sensitivity+final_res{6}.sensitivity+final_res{7}.sensitivity+final_res{8}.sensitivity)/4
sensitivityDR = (final_res{1}.sensitivity+final_res{2}.sensitivity+final_res{3}.sensitivity+final_res{4}.sensitivity)/4

specificityCDB = (final_res{1}.specificity+final_res{2}.specificity+final_res{3}.specificity+final_res{4}.specificity)/4
specificityST = (final_res{5}.specificity+final_res{6}.specificity+final_res{7}.specificity+final_res{8}.specificity)/4
specificityDR = (final_res{9}.specificity+final_res{10}.specificity+final_res{11}.specificity+final_res{12}.specificity)/4

AUCCDB = (final_res{1}.AUC+final_res{2}.AUC+final_res{3}.AUC+final_res{4}.AUC)/4
AUCST = (final_res{5}.AUC+final_res{6}.AUC+final_res{7}.AUC+final_res{8}.AUC)/4
AUCDR = (final_res{9}.AUC+final_res{10}.AUC+final_res{11}.AUC+final_res{12}.AUC)/4

ACC = s_accuracy/20
SE = s_sensitivity/20
SP = s_specificity/20
AC = s_AUC/20
