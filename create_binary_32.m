function fVector = create_binary( row,column,Intgral_Img,PatchSize,times )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%format short
times=times*2;
% temp_str='';
UL=[];
UR=[];
DL=[];
DR=[];
threshold=0;
% 
% UL_r=0;
% UL_c=0;
% 
% UR_r=0;
% UR_c=0;
% 
% DL_r=0;
% DL_c=0;
% 
% DR_r=0;
% DR_c=0;




    r=row;
    c=column;
    
    H_r=floor(PatchSize/2);
    H_c=floor(PatchSize/2);
    H_r_2=floor(H_r/2);
    H_c_2=floor(H_c/2);
    
    I=Intgral_Img;

    %feature 1=>left:right checking

    A1=I(r+H_r+1,c+1)+I(r-H_r+1,c-H_c+1)-I(r+H_r+1,c-H_c+1)-I(r-H_r+1,c+1);
    A1 = A1/(PatchSize*PatchSize*.5);
    A2=I(r+H_r+1,c+H_c+1)+I(r-H_r+1,c+1)-I(r+H_r+1,c+1)-I(r-H_r+1,c+H_c+1);
    A2 = A2/(PatchSize*PatchSize*.5);
    
    diff1 = A1-A2;

    

    %feature 2=> up:down checking

    A1=I(r+1,c+ H_c+1)+I(r-H_r+1,c-H_c+1)-I(r+1,c-H_c+1)-I(r-H_r+1,c+H_c+1);
    A1 = A1/(PatchSize*PatchSize*.5);
    A2=I(r+H_r+1,c+H_r+1)+I(r+1,c-H_c+1)-I(r+H_r+1,c-H_c+1)-I(r+1,c+ H_r+1);
    A2 = A2/(PatchSize*PatchSize*.5);
    
    diff2 = A1-A2;

    
    %feature 3=>alternate in the up-down direction

    A11=I(r+1,c+1)+I(r-H_r+1,c-H_c+1)-I(r+1,c-H_c+1)-I(r-H_r+1,c+1);
    A12=I(r+H_r+1,c+H_c+1)+I(r+1,c+1)-I(r+H_r+1,c+1)-I(r+1,c+H_c+1);
    A1=A11+A12;
    
    A1 = A1/(PatchSize*PatchSize*.5);

    A21=I(r+H_r+1,c+1)+I(r+1,c-H_c+1)-I(r+H_r+1,c-H_c+1)-I(r+1,c+1);
    A22=I(r+1,c+H_c+1)+I(r-H_r+1,c+1)-I(r+1,c+1)-I(r-H_r+1,c+H_c+1);
    A2=A21+A22;
    
    A2 = A2/(PatchSize*PatchSize*.5);
    
    diff3 = A1-A2;
    
     %feature 4=>alternate in the left-right direction

    temp_A11=I(r+H_r+1,c-H_c_2+1)-I(r-H_r+1,c-H_c_2+1)-I(r+H_r+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    temp_A12=I(r+H_r+1,c+H_c_2+1)-I(r-H_r+1,c+H_c_2+1)-I(r+H_r+1,c+1)+I(r-H_r_2+1,c+1);
    A1=temp_A11+temp_A12;
    A1 = A1/(PatchSize*PatchSize*.5);
    
    temp_A21=I(r+H_r+1,c+1)-I(r-H_r,c+1)-I(r+H_r+1,c-H_c_2+1)+I(r-H_r+1,c-H_c_2+1);
    temp_A22=I(r+H_r+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r+H_r+1,c+H_c_2+1)+I(r-H_r_2+1,c+H_c_2+1);
    A2=temp_A21+temp_A22;
    A2 = A2/(PatchSize*PatchSize*.5);
    
    diff4 = A1 - A2-threshold;

    %feature 5 => middle with left corner

    midA=I(r+H_r_2+1,c+H_c_2+1)-I(r-H_r_2+1,c+H_c_2+1)-I(r+H_r_2+1,c-H_c_2+1)+I(r-H_r_2+1,c-H_c_2+1);
    
    A1=I(r+1,c+1)-I(r-H_r+1,c+1)-I(r+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    A1 = A1/(PatchSize*PatchSize*.5);
    
    midA = midA/(PatchSize*PatchSize*.5);
    
    diff5 = A1 - midA-threshold;


    %feature 6 =>middle with right corner
    % previous midA can be used directly
    % midA=I(r+H_r_2+1,c+H_c_2+1)-I(r-H_r_2+1,c+H_c_2+1)-I(r+H_r_2+1,c-H_c_2+1)+I(r-H_r_2+1,c-H_c_2+1);
    
    A1=I(r+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r+1,c+1)+I(r-H_r+1,c+1);
    A1 = A1/(PatchSize*PatchSize*.5);
   
    diff6 = A1 - midA-threshold;
    
    diff7=[];diff8=[];diff9=[];diff10=[];diff11=[];diff12=[];diff13=[];diff14=[];diff15=[];diff16=[];diff17=[];diff18=[];
    diff19=[];diff20=[];diff21=[];diff22=[];diff23=[];diff24=[];diff25=[];diff26=[];diff27=[];diff28=[];diff29=[];diff30=[];
    diff31=[];diff32=[];
    
    if (times~=8)
     % feature 7 =>middle with right-down corner
    % previous midA can be used directly
    
    A1=I(r+H_r+1,c+H_c+1)-I(r+1,c+H_c+1)-I(r+H_r+1,c+1)+I(r+1,c+1);
    
     A1 = A1/(PatchSize*PatchSize*.5);
   
    diff7 = A1 - midA-threshold;

    
    % feature 8 =>middle with left corner
    % previous midA can be used directly
    A1=I(r+H_r+1,c+1)-I(r+1,c+1)-I(r+H_r+1,c-H_c+1)+I(r+1,c-H_c+1);

    A1 = A1/(PatchSize*PatchSize*.5);
   
    diff8 = A1 - midA-threshold;

    %feature 9

    temp_A11=I(r+H_r_2+1,c+1)-I(r-H_r+1,c+1)-I(r+H_r_2+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    temp_A12=I(r+H_r+1,c+H_c+1)-I(r+H_r_2+1,c+H_c+1)-I(r+H_r+1,c+1)+I(r+H_r_2+1,c+1);
    A1=temp_A11+temp_A12;
    
    temp_A21=I(r+H_r_2+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r+H_r_2+1,c+1)+I(r-H_r+1,c+1);
    temp_A22=I(r+H_r+1,c+1)-I(r+H_r_2+1,c+1)-I(r+H_r+1,c-H_c+1)+I(r+H_r_2+1,c-H_c+1);
    A2=temp_A21+temp_A22;
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff9 = A1 - A2-threshold;
    
    %feature 10

    temp_A11=I(r+H_r+1,c+1)-I(r-H_r_2+1,c+1)-I(r+H_r+1,c-H_c+1)+I(r-H_r_2+1,c-H_c+1);
    temp_A12=I(r-H_r_2+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r-H_r_2+1,c+1)+I(r-H_r+1,c+1);
    A1=temp_A11+temp_A12;
    
    temp_A21=I(r+H_r+1,c+H_c+1)-I(r-H_r_2+1,c+H_c+1)-I(r+H_r+1,c+1)+I(r-H_r_2+1,c+1);
    temp_A22=I(r-H_r_2+1,c+1)-I(r-H_r+1,c+1)-I(r-H_r_2+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    A2=temp_A21+temp_A22;
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff10 = A1 - A2-threshold;
    
    
    %feature 11

    temp_A11=I(r+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r+1,c-H_c_2+1)+I(r-H_r+1,c-H_c_2+1);
    temp_A12=I(r+H_r+1,c-H_c_2+1)-I(r+1,c-H_c_2+1)-I(r+H_r+1,c-H_c+1)+I(r+1,c-H_c+1);
    A1=temp_A11+temp_A12;
    
    temp_A21=I(r+H_r+1,c+H_c+1)-I(r+1,c+H_c+1)-I(r+H_r+1,c-H_c_2+1)+I(r+1,c-H_c_2+1);
    temp_A22=I(r+1,c-H_c_2+1)-I(r-H_r+1,c-H_c_2+1)-I(r+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    A2=temp_A21+temp_A22;
    
   A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff11 = A1 - A2-threshold;
    
    %feature 12

    temp_A11=I(r+1,c+H_c_2+1)-I(r-H_r+1,c+H_c_2+1)-I(r+1,c-H_c_2+1)+I(r-H_r+1,c-H_c_2+1);
    temp_A12=I(r+H_r+1,c+H_c+1)-I(r+1,c+H_c+1)-I(r+H_r+1,c+H_c_2+1)+I(r+1,c+H_c_2+1);
    temp_A13=I(r+H_r+1,c-H_c_2+1)-I(r+1,c-H_c_2+1)-I(r+H_r+1,c-H_c+1)+I(r+1,c-H_c+1);
    A1=temp_A11+temp_A12+temp_A13;
    
    temp_A21=I(r+H_r+1,c+H_c_2+1)-I(r+1,c+H_c_2+1)-I(r+H_r+1,c-H_c_2+1)+I(r+1,c-H_c_2+1);
    temp_A22=I(r+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r+1,c+H_c_2+1)+I(r-H_r+1,c+H_c_2+1);
    temp_A23=I(r+1,c-H_c_2+1)-I(r-H_r+1,c-H_c_2+1)-I(r+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    A2=temp_A21+temp_A22+temp_A23;
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff12 = A1 - A2-threshold;
    
    
       
    %if (times==2)
    
    %feature 13

    temp_A11=I(r+H_r_1+1,c+1)-I(r-H_r_2+1,c+1)-I(r+H_r_1+1,c-H_c+1)+I(r-H_r_2+1,c-H_c+1);
    temp_A12=I(r-H_r_2+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r-H_r_2+1,c+1)+I(r-H_r+1,c+1);
    temp_A13=I(r+H_r+1,c+H_c+1)-I(r+H_r_2+1,c+H_c+1)-I(r+H_r+1,c+1)+I(r+H_r_2+1,c+1);
    A1=temp_A11+temp_A12+temp_A13;
    
    temp_A21=I(r+H_r_2+1,c+H_c+1)-I(r-H_r_2+1,c+H_c+1)-I(r+H_r_2+1,c+1)+I(r-H_r_2+1,c+1);
    temp_A22=I(r-H_r_2+1,c+1)-I(r-H_r+1,c+1)-I(r-H_r_2+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    temp_A23=I(r+H_r+1,c+1)-I(r+H_r_2+1,c+1)-I(r+H_r+1,c-H_c+1)+I(r+H_r_2+1,c-H_c+1);
    A2=temp_A21+temp_A22+temp_A23;
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff13 = A1 - A2-threshold;


    %feature 14

    temp_A11=I(r-H_r_2+1,c+1)-I(r-H_r+1,c+1)-I(r-H_r_2+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    temp_A12=I(r+1,c+H_c+1)-I(r-H_r_2+1,c+H_c+1)-I(r+1,c+1)+I(r-H_r_2+1,c+1);
    temp_A13=I(r+H_r_2+1,c+1)-I(r+1,c+1)-I(r+H_r_2+1,c-H_c+1)+I(r+1,c-H_c+1);
    temp_A14=I(r+H_r+1,c+H_c+1)-I(r+H_r_2+1,c+H_c+1)-I(r+H_r+1,c+1)+I(r+H_r_2+1,c+1);
    A1=temp_A11+temp_A12+temp_A13+temp_A14;
    
    temp_A21=I(r-H_r_2+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r-H_r_2+1,c+1)+I(r-H_r+1,c+1);
    temp_A22=I(r+1,c+1)-I(r-H_r_2+1,c+1)-I(r+1,c-H_c+1)+I(r-H_r_2+1,c-H_c+1);
    temp_A23=I(r+H_r_2+1,c+H_c+1)-I(r+1,c+H_c+1)-I(r+H_r_2+1,c+1)+I(r+1,c+1);
    temp_A24=I(r+H_r+1,c+1)-I(r+H_r_2+1,c+1)-I(r+H_r+1,c-H_c+1)+I(r+H_r_2+1,c-H_c+1);
    A2=temp_A21+temp_A22+temp_A23+temp_A24;
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff14 = A1 - A2-threshold;
    
    %feature 15

    temp_A11=I(r+1,c-H_c_2+1)-I(r-H_r+1,c-H_c_2+1)-I(r+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    temp_A12=I(r+1,c+H_c_2+1)-I(r-H_r+1,c+H_c_2+1)-I(r+1,c+1)+I(r-H_r+1,c+1);
    temp_A13=I(r+H_r+1,c+1)-I(r+1,c+1)-I(r+H_r+1,c-H_c_2+1)+I(r+1,c-H_c_2+1);
    temp_A14=I(r+H_r+1,c+H_c+1)-I(r+1,c+H_c+1)-I(r+H_r+1,c+H_c_2+1)+I(r+1,c+H_c_2+1);
    A1=temp_A11+temp_A12+temp_A13+temp_A14;
    
    temp_A21=I(r+1,c+1)-I(r-H_r+1,c+1)-I(r+1,c-H_c_2+1)+I(r-H_r+1,c-H_c_2+1);
    temp_A22=I(r+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r+1,c+H_c_2+1)+I(r-H_r+1,c+H_c_2+1);
    temp_A23=I(r+H_r+1,c-H_c_2+1)-I(r+1,c-H_c_2+1)-I(r+H_r+1,c-H_c+1)+I(r+1,c-H_c+1);
    temp_A24=I(r+H_r+1,c+H_c_2+1)-I(r+1,c+H_c_2+1)-I(r+H_r+1,c+1)+I(r+1,c+1);
    A2=temp_A21+temp_A22+temp_A23+temp_A24;
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff15 = A1 - A2-threshold;
    
    %feature 16

    temp_A11=I(r+1,c+1)-I(r-H_r+1,c+1)-I(r+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    temp_A12=I(r+H_r+1,c+1)-I(r+1,c+1)-I(r+H_r+1,c-H_c_2+1)+I(r+1,c-H_c_2+1);
    temp_A13=I(r+H_r+1,c+H_c_2+1)-I(r+1,c+H_c_2+1)-I(r+H_r+1,c+1)+I(r+1,c+1);
    A1=temp_A11+temp_A12+temp_A13;
    
    temp_A21=I(r+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r+1,c+1)+I(r-H_r+1,c+1);
    temp_A22=I(r+H_r+1,c-H_c_2+1)-I(r+1,c-H_c_2+1)-I(r+H_r+1,c-H_c+1)+I(r+1,c-H_c+1);
    temp_A23=I(r+H_r+1,c+H_c_2+1)-I(r+1,c+H_c_2+1)-I(r+H_r+1,c+1)+I(r+1,c+1);
    A2=temp_A21+temp_A22+temp_A23;
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff16 = A1 - A2-threshold;
%feature 17

    A1=I(r+1,c+1)-I(r-H_r+1,c+1)-I(r+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    A2=I(r+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r+1,c+1)+I(r-H_r+1,c+1);
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff17 = A1 - A2-threshold;



% feature 18
    % previous A1 can be used directly
    
    A3=I(r+H_r+1,c+H_c+1)-I(r+1,c+H_c+1)-I(r+H_r+1,c+1)+I(r+1,c+1);
    
    A3 = A3/(PatchSize*PatchSize*.5);
   
    diff18 = A1 - A3-threshold;

    
    % feature 19
    % previous A1 can be used directly
    A4=I(r+H_r+1,c+1)-I(r+1,c+1)-I(r+H_r+1,c-H_c+1)+I(r+1,c-H_c+1);

    A4 = A4/(PatchSize*PatchSize*.5);
   
    diff19 = A1 - A4-threshold;

% feature 20
    % previous A2, A3 can be used directly
      
    diff20 = A2 - A3-threshold;

% feature 21
    % previous A2, A4 can be used directly
   
    diff21 = A2 - A4-threshold;

% feature 22
    % previous A3, A4 can be used directly

    diff22 = A3 - A4-threshold;

%feature 23

    temp_A11=I(r+1,c+H_c_2+1)-I(r-H_r+1,c+H_c_2+1)-I(r+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    temp_A12=I(r+H_r+1,c+H_c+1)-I(r+1,c+H_c+1)-I(r+H_r+1,c+H_c_2+1)+I(r+1,c+H_c_2+1);
    A1=temp_A11+temp_A12;
    
    temp_A21=I(r+H_r+1,c+H_c_2+1)-I(r+1,c+H_c_2+1)-I(r+H_r+1,c-H_c+1)+I(r+1,c-H_c+1);
    temp_A22=I(r+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r+1,c+H_c_2+1)+I(r-H_r+1,c+H_c_2+1);
    A2=temp_A21+temp_A22;
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff23 = A1 - A2-threshold;

    %feature 24

    temp_A11=I(r+H_r+1,c+1)-I(r-H_r_2+1,c+1)-I(r+H_r+1,c-H_c+1)+I(r-H_r_2+1,c-H_c+1);
    temp_A12=I(r-H_r_2+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r-H_r_2+1,c+1)+I(r-H_r+1,c+1);
    A1=temp_A11+temp_A12;
    
    temp_A21=I(r+H_r+1,c+H_c+1)-I(r-H_r_2+1,c+H_c+1)-I(r+H_r+1,c+1)+I(r-H_r_2+1,c+1);
    temp_A22=I(r-H_r_2+1,c+1)-I(r-H_r+1,c+1)-I(r-H_r_2+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    A2=temp_A21+temp_A22;
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff24 = A1 - A2-threshold;

    %feature 25

    temp_A11=I(r+H_r_2+1,c+1)-I(r-H_r+1,c+1)-I(r+H_r_2+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    temp_A12=I(r+H_r+1,c+H_c+1)-I(r+H_r_2+1,c+H_c+1)-I(r+H_r+1,c+1)+I(r+H_r_2+1,c+1);
    A1=temp_A11+temp_A12;
    
    temp_A21=I(r+H_r_2+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r+H_r_2+1,c+1)+I(r-H_r+1,c+1);
    temp_A22=I(r+H_r+1,c+1)-I(r+H_r_2+1,c+1)-I(r+H_r+1,c-H_c+1)+I(r+H_r_2+1,c-H_c+1);
    A2=temp_A21+temp_A22;
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff25 = A1 - A2-threshold;

    %feature 26

    temp_A11=I(r+1,c+H_c_2+1)-I(r-H_r_2+1,c+H_c_2+1)-I(r+1,c-H_c_2+1)+I(r-H_r_2+1,c-H_c_2+1);
    temp_A12=I(r+H_r+1,c+H_c_2+1)-I(r+H_r_2+1,c+H_c_2+1)-I(r+H_r+1,c-H_c_2+1)+I(r+H_r_2+1,c-H_c_2+1);
    temp_A13=I(r+H_r+1,c-H_c_2+1)-I(r-H_r+1,c-H_c_2+1)-I(r+H_r+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    A1=temp_A11+temp_A12+temp_A13;
    
    temp_A21=I(r-H_r_2+1,c+H_c_2+1)-I(r-H_r+1,c+H_c_2+1)-I(r-H_r_2+1,c-H_c_2+1)+I(r-H_r+1,c-H_c_2+1);
    temp_A22=I(r+H_r_2+1,c+H_c_2+1)-I(r+1,c+H_c_2+1)-I(r+H_r_2+1,c-H_c_2+1)+I(r+1,c-H_c_2+1);
    temp_A23=I(r+H_r+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r+H_r+1,c+H_c_2+1)+I(r-H_r+1,c+H_c_2+1);
    A2=temp_A21+temp_A22+temp_A23;
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff26 = A1 - A2-threshold;

%feature 27

    A1=temp_A11+temp_A12+temp_A23;
    
    A2=temp_A21+temp_A22+temp_A13;
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff27 = A1 - A2-threshold;

%feature 28

    A1=I(r+1,c+H_c_2+1)-I(r-H_r+1,c+H_c_2+1)-I(r+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    A2=I(r+H_r+1,c+H_c+1)-I(r+1,c+H_c+1)-I(r+H_r+1,c-H_c_2+1)+I(r+1,c-H_c_2+1);
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff28 = A1 - A2-threshold;


    %feature 29

    A1=I(r+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r+1,c-H_c_2+1)+I(r-H_r+1,c-H_c_2+1);
    A2=I(r+H_r+1,c+H_c_2+1)-I(r+1,c+H_c_2+1)-I(r+H_r+1,c-H_c+1)+I(r+1,c-H_c+1);
    
   A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff29 = A1 - A2-threshold;

    
    %feature 30

    A1=I(r+H_r_2+1,c+1)-I(r-H_r+1,c+1)-I(r+H_r_2+1,c-H_c+1)+I(r-H_r+1,c-H_c+1);
    A2=I(r+H_r+1,c+H_c+1)-I(r-H_r_2+1,c+H_c+1)-I(r+H_r+1,c+1)+I(r-H_r_2+1,c+1);
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff30 = A1 - A2-threshold;


    %feature 31

    A1=I(r+H_r+1,c+1)-I(r-H_r_2+1,c+1)-I(r+H_r+1,c-H_c+1)+I(r-H_r_2+1,c-H_c+1);
    A2=I(r+H_r_2+1,c+H_c+1)-I(r-H_r+1,c+H_c+1)-I(r+H_r_2+1,c+1)+I(r-H_r+1,c+1);
    
    A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff31 = A1 - A2-threshold;


    %feature 32

    A1=I(r+1,c+H_c_2+1)-I(r-H_r+1,c+H_c_2+1)-I(r+1,c-H_c_2+1)+I(r-H_r+1,c-H_c_2+1);
    A2=I(r+H_r+1,c+H_c_2+1)-I(r+1,c+H_c_2+1)-I(r+H_r+1,c-H_c_2+1)+I(r+1,c-H_c_2+1);
    
   A1 = A1/(PatchSize*PatchSize*.5);
    A2 = A2/(PatchSize*PatchSize*.5);
   
    diff32 = A1 - A2-threshold;
    
    end
    

     fVector = [diff1,diff2,diff3,diff4,diff5,diff6,diff7,diff8,diff9,diff10,diff11,diff12,diff13,diff14,...
         diff15,diff16,diff17,diff18,diff19,diff20,diff21,diff22,diff23,diff24,diff25,diff26,diff27,diff28,...
         diff29,diff30,diff31,diff32];
     PatchSize=PatchSize/2;
    
    row_fix=r-PatchSize;
    column_fix=c-PatchSize;
    
    new_r=r-(row_fix);
    new_c=c-(column_fix);
    
    new_r=floor(new_r/2);
    new_c=floor(new_c/2);

    UL_r=new_r+row_fix;
    UL_c=new_c+column_fix;

    UR_r=new_r+row_fix;
    UR_c=PatchSize+new_c+column_fix;

    DL_r=PatchSize+new_r+row_fix;
    DL_c=new_c+column_fix;

    DR_r=PatchSize+new_r+row_fix;
    DR_c=PatchSize+new_c+column_fix;
    
    if times<=4

        UL=create_binary(UL_r,UL_c,I,PatchSize,times);
        UR=create_binary(UR_r,UR_c,I,PatchSize,times);
        DL=create_binary(DL_r,DL_c,I,PatchSize,times);
        DR=create_binary(DR_r,DR_c,I,PatchSize,times);
        %fVector=[fVector UL UR DL DR];
    end
    
    
    fVector=[fVector UL UR DL DR];
   
    


end

