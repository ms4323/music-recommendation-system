function user_song = userSongMatrix(user_unique,song_unique,numerical_data)

%% Build User-Song Matrix

user_song = zeros(size(user_unique,1),size(song_unique,1));
for ii =1:size(user_song,1)
    for jj=1:size(user_song,2)
        
        if ~isempty(numerical_data(find(numerical_data(:,1)==ii & numerical_data(:,2)==jj),3))
            user_song(ii,jj) = sum(numerical_data(find(numerical_data(:,1)==ii & numerical_data(:,2)==jj),3));
        end;
    end;    
end;

        


end