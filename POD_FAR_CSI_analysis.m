clear all
applyset = h5read('GLM_applyset_CAMS.h5','/applyset');
applyset = applyset';
% Adjustable thresolds tuned here
applyset(:,20) = applyset(:,19)>0.45;


POD_record = zeros(112,272);
FAR_record = zeros(112,272);
CSI_record = zeros(112,272);

for i = 1:112
      disp(i);
     for j = 1:272
       
         index = find(applyset(:,2) == i & applyset(:,3) == j);
         if length(index)==0
             continue
         end
         record_new = applyset(index,[18,20]);
         TP = record_new(record_new(:,1) == 1 & record_new(:,2) == 1,:);
         FN = record_new(record_new(:,1) == 1 & record_new(:,2) == 0,:);
         FP = record_new(record_new(:,1) == 0 & record_new(:,2) == 1,:);
         if (length(TP) + length(FN)) == 0 
             continue
         end
         if (length(TP) + length(FP)) == 0 
             continue
         end
         POD = length(TP)/(length(TP) + length(FN));
         FAR = 1 - length(TP)/(length(TP) + length(FP));
         CSI = length(TP)/(length(TP) + length(FP) + length(FN));
         POD_record(i,j) = POD;
         FAR_record(i,j) = FAR;
         CSI_record(i,j) = CSI;
     end
end

TIFdata = flip(POD_record);
fileName = ['POD_analysis.tif'];
TIFdata = flip(FAR_record);
fileName = ['FAR_analysis.tif'];
TIFdata = flip(CSI_record);
fileName = ['CSI_analysis.tif'];

DTM=TIFdata;                 
rasterSize=size(DTM);       
latlim= [23,51];
lonlim= [-132,-64];
R = georefcells(latlim,lonlim,rasterSize);   
geotiffwrite(fileName, DTM, R);    
