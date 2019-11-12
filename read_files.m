function [numerical_data,user_unique,user_unique_idx,song_unique,song_unique_idx] = read_files(data_size,user_file,song_file)
%% Read Files

% 1M User Data
user_fileID = fopen(user_file);
sizeUsers = [1000000 3];
users = textscan(user_fileID,'%s %s %s',sizeUsers,'headerlines', 0);

% 1M Songs Data 
song_fileID = fopen(song_file);
sizeSongs = [100000 5]; 
songs = textscan(song_fileID,'%s%s%s%s%s',sizeSongs,'delimiter',',','headerlines', 1);

% Close files
fclose(song_fileID);
fclose(user_fileID);

%% Tables for both Users and Songs

user_ID = users{1,1}; % User ID
user_songID = users{1,2}; % Songs ID
user_listens = users{1,3}; % Listen Times

% Create a table for users
user_table = table(user_ID,user_songID,user_listens,'VariableNames',{'userID','songID','listenTimes'});
user_table = user_table(randperm(size(user_table,1)),:);


song_songID = songs{1,1}; % Song ID
song_title = songs{1,2}; % Song Title 
song_release = songs{1,3}; % Release
song_artistName = songs{1,4}; % Artist Name 
song_year = songs{1,5}; % Year

% Create a table for songs
song_table = table(song_songID,song_title,song_release,song_artistName,song_year,'VariableNames',{'songID','songTitle','songRelease','songArtist','songYear'}); 
song_table = song_table(randperm(size(song_table,1)),:);

%% Join the tables

user_song_table = innerjoin(user_table,song_table);
data = table2array(user_song_table);
data = data(1:data_size,:);

%% Build numerical data matrix

[~,user_unique,user_unique_idx] = unique(data(:,1)); %user
[~,song_unique,song_unique_idx] = unique(data(:,2)); %songs
[~,~,release_unique_idx] = unique(data(:,4)); %song release
[~,~,artist_unique_idx] = unique(data(:,6)); %song artist

numerical_data = zeros(size(data,1),3);
numerical_data(:,1) = user_unique_idx(:); % first column -> user idx
numerical_data(:,2) = song_unique_idx(:); % second column -> song idx
numerical_data(:,3) = cellfun(@str2num,data(:,3)); % third column -> listen counts

% user_songArtist_listen = zeros(size(data,1),1);
% for ii=1:size(data,1)
%     user_songArtist_listen(ii) = sum( numerical_data(find(artist_unique_idx==artist_unique_idx(ii) & numerical_data(:,1)==numerical_data(ii,1)),3) );
% end;
% numerical_data(:,4) = user_songArtist_listen(:); % fourth column -> parent (artist) listen counts
% numerical_data(:,5) = artist_unique_idx(:); % fifth column -> artist idx
% 
% user_songRelease_listen = zeros(size(data,1),1);
% for ii=1:size(data,1)
%     user_songRelease_listen(ii) = sum( numerical_data(find(release_unique_idx==release_unique_idx(ii) & numerical_data(:,1)==numerical_data(ii,1)),3) );
% end;
% numerical_data(:,6) = user_songRelease_listen(:); % sixth column -> parent (release) listen counts
% numerical_data(:,7) = release_unique_idx(:); % seventh column -> release idx


end