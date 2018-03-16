% Author: Robert Neff, 2018
% Description: Converts database files of signature points into images
% through naive method of plotting points then saving the plot. Also places
% images in new easily accessible folders.
%
% Database: http://www.vision.caltech.edu/mariomu/research.html
% Download database and run this file in the top directory to get images.
%
% Reference:
% Based on code view_signatures.m and view_forgeries.m from:
% Mario E. Munich & Pietro Perona
% California Institute of Technology

% Loads all the images from starting format to image format withing their
% respective folders
function load_database()
    % Loop over datasets    
    for set=1:2
        % Set the directory name
        directory = ['set',num2str(set)];
        mkdir('images', num2str(set));

        % Set the list file name
        listfile = [directory,'/list',directory];
        
        % Load the list file
        eval(['load ',listfile]);
        eval(['list = list',directory,';']);
        n_subjects = size(list, 1);
        
        % Loop over subject directories
        for subject=1:n_subjects
            mkdir(['images', num2str(set), '/', 'subject_',sprintf('%.3d',subject)])
            get_subj_signatures(directory, subject, set);
            get_subj_forgeries(directory, subject, set);
        end  
    end
    return;
end

% Gets and saves the signatures for current subject
function get_subj_signatures(directory, subject, set)
    seq = ['s',sprintf('%.3d',subject)];
    mkdir(['images', num2str(set), '/', 'subject_', sprintf('%.3d',subject), '/'], 'Genuine');
    
    % Get data count
    countfile = [directory,'/data/',seq,'/',seq,'Count'];

    % Load the count file
    fid = fopen(countfile,'r');
    count = fscanf(fid,'%d\n',[1,1]);
    fclose(fid);

    % Loop over signatures to save
    for s=1:count-1
        path = [directory,'/data/'];
        save_path = ['images', num2str(set), '/', 'subject_', sprintf('%.3d',subject), '/Genuine'];
        save_signature(path, save_path, seq, s, false);
    end
end

% Gets and saves the forgeries for current subject
function get_subj_forgeries(directory, subject, set)
    seq = ['s',sprintf('%.3d',subject)];
    mkdir(['images', num2str(set), '/', 'subject_', sprintf('%.3d',subject), '/'], 'Forgeries');
    
    % Get data count
    countfile = [directory,'/forgeries/',seq,'/',seq,'fCount'];

    % Load the count file
    fid = fopen(countfile,'r');
    count = fscanf(fid,'%d\n',[1,1]);
    fclose(fid);

    % Loop over forgeries to save
    for s=1:count-1
        path = [directory,'/forgeries/'];
        save_path = ['images', num2str(set), '/', 'subject_', sprintf('%.3d',subject), '/Forgeries'];
        save_signature(path, save_path, seq, s, true);
    end
end

% Saves the plot of the data values
function save_signature(path, save_path, seq, number, forged)
    % Set the file
    file = 0;
    if (forged)
        file = [path,seq,'/',seq,'f',sprintf('%.3d',number)];
    else
        file = [path,seq,'/',seq,sprintf('%.3d',number)];
    end

    % Load the signature
    fid = fopen(file,'r');
    lgn = fscanf(fid,'%d\n',[1,1]);
    sig = fscanf(fid,'%f %f\n',[2,lgn]);
    fclose(fid);

    % Plot the signature
    x = sig(1,:); y = -sig(2,:)+max(sig(2,:));
    figure(1); 
    set(1,'Visible', 'off'); % close plot window
    clf;
    plot(x, y, '-k');
    axis([min(x)-5 max(x)+5 min(y)-5 max(y)+5])
    
    % Save to image with axis off
    ax = gca;
    ax.Visible = 'off';
    saveas(1, fullfile(save_path, [seq, sprintf('%.3d',number)]), 'jpeg');
    return;
end
