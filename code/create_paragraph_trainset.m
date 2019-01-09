function[] = create_paragraph_trainset(myFolder)
fid1 = fopen('C:\\Users\\Mukaddes\\Desktop\\calisan\\train_14_5_p.txt', 'w');

% fprintf(fid1,'@relation authorship\n@attribute text string\n@attribute class{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30}\n@data\n');
fclose('all');

filePattern = fullfile(myFolder, '*.txt'); 
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  
  % preproces lines
  lines = (read_file(fullFileName));
  lines = erasePunctuation(lines);
  lines = lower(lines); 
  
  
  % find the label of the data
  directories = textscan(fullFileName,'%s','Delimiter','\');
  tmp_str = char(directories{1}(6,1));
  tmp_lbl = textscan(tmp_str, '%s','Delimiter','.'); 
  label = char(tmp_lbl{1,1}(1,1));
  
  fid1 = fopen('C:\\Users\\Mukaddes\\Desktop\\calisan\\train_14_5_p.txt', 'a+');
  
     
  
  num_of_lines = size(lines,1);
  for i=1:num_of_lines    
    line = string(lines(i)); 
    if(~strcmp(line,""))
    fprintf(fid1,label);
%     fprintf(fid1,'%s', 39);
    fprintf(fid1, ' %s',line);
%     fprintf(fid1,'%s', 39);
%     fprintf(fid1,',');
    
    fprintf(fid1,'\n');
    end
  end
  
  fclose('all');

end

end