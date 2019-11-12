function numerical_data=popularity(unique_songs,numerical_data)

%% Song popularity (for all users of the training system a list of popular songs is recommended)

song_popularity = zeros(size(unique_songs,1),2);
song_popularity(:,1) = unique_songs;
for jj=1:size(unique_songs,1)
    
    song_popularity(jj,2) = sum(numerical_data(find(numerical_data(:,2)==jj),3));
end;    

song_popularity = sortrows(song_popularity,2); % sort the unique songs based on the popularity


for ii=1:size(numerical_data,1)

    for jj=1:size(unique_songs,1)
        if numerical_data(ii,2) == song_popularity(jj,1)
            numerical_data(ii,4) = song_popularity(jj,2);
        end;
    end;

end;

%% Artist popularity (for all users of the training system a list of popular artists is recommended)
% [~,unique_artists,~] = unique(data(:,6));
% 
% artist_popularity = zeros(size(unique_artists,1),2);
% artist_popularity(:,1) = unique_artists;
% for jj=1:size(unique_artists,1)
%     
%     artist_popularity(jj,2) = sum(numerical_data(find(numerical_data(:,6)==jj),3));
% end;    
% 
% [~,idx_artist_popularity]=sort(artist_popularity(:,2),'descend');
% idx_recommend_artist_popularity = artist_popularity(idx_artist_popularity,1);

end