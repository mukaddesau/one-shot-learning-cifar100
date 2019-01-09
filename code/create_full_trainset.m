function[] = create_full_trainset(myFolder)
fid1 = fopen('C:\\Users\\Mukaddes\\Desktop\\calisan\\train_14_5.txt', 'w');

% fprintf(fid1,'@relation authorship\n@attribute text string\n@attribute class{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30}\n@data\n');
fclose('all');

filePattern = fullfile(myFolder, '*.txt'); 
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
%   fprintf(1, 'Now reading %s\n', fullFileName);
  % Now do whatever you want with this file name,
  % such as reading it in as an image array with imread()
  
%   find the label of the data
  lines = (read_file(fullFileName));
  lines = erasePunctuation(lines);
  lines = lower(lines);
  directories = textscan(fullFileName,'%s','Delimiter','\');
  tmp_str = char(directories{1}(6,1));
  tmp_lbl = textscan(tmp_str, '%s','Delimiter','.'); 
  label = char(tmp_lbl{1,1}(1,1));
  
  fid1 = fopen('C:\\Users\\Mukaddes\\Desktop\\calisan\\train_14_5.txt', 'a+');
  fprintf(fid1,label);
%   fprintf(fid1,'%s', 39);
  fprintf(fid1, ' %s',lines{:});
%   fprintf(fid1,'%s', 39);
%   fprintf(fid1,',');
 
  fprintf(fid1,'\n');
  fclose('all');   

end

end