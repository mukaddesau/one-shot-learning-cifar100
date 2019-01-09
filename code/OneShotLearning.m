% download image files
% download_files();

% Specify the folder where the files live.
myFolder = 'C:\\Users\Mukaddes\\Desktop\\columns_14_5';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isfolder(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end

create_full_trainset(myFolder);
create_paragraph_trainset (myFolder);


