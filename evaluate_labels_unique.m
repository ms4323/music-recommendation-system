function scr = evaluate_labels_unique(user_unique,user_unique_idx,Y,label_result)

% res = Y - label_result;
% score = 0;
% for i=1:size(res,1)
% 
%     if res(i) == 0
%         score = score + 1/(2^i);
%         %score = score + 1;
%     end;
% 
% end;
% 
% score = score/size(res,1);

for ii=1:size(user_unique,1)
    
user_idx{ii} = find(user_unique_idx == ii);
    
end; 

score = zeros(size(user_unique,1),1);
for ii=1:size(user_unique,1)
    for jj=1:size(user_idx{ii},1)
        if Y(user_idx{ii}(jj)) - label_result(user_idx{ii}(jj)) == 0
            score(ii) = score(ii) +  1/(2^jj);
        end;
    end;
    %score(ii) = score(ii) / size(user_idx{ii},1);
end; 

scr = mean(score);

end