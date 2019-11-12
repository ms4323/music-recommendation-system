function [AA,B] = correlation_calculation(A,sigma,gamma,type) 

AA = zeros(size(A,1),size(A,1));


if strcmp(type,'cosine')
    % Cosine Similarity:
    n_row = size(A,1);
    norm_r = sqrt(sum(abs(A).^2,2)); 
    AA = zeros(n_row,n_row);
    for i = 1:n_row
        for j = i:n_row
            AA(i,j) = dot(A(i,:), A(j,:)) / (norm_r(i) * norm_r(j));
            AA(j,i) = AA(i,j);
        end;
    end;
    

elseif strcmp(type,'pearson')
% Pearson correlation: 
% U = corr(A'); % basically U is the same as AA (the user-based similarity)
% S = corr(A); % S is item-based (song) similarity

AA = corr(A');

else
% Paper correlation:
for ii=1:size(A,1)
    
    c = sqrt(sum(A(ii,:)));
    a = 0;
    for jj=1:size(A,1)
        
        
        for kk=1:size(A,2)
            a = (A(ii,kk)-A(jj,kk))^2 + a;
        end;
        d = sqrt(sum(A(jj,:))); 
        b = c*d;
        
        AA(ii,jj) = (a/b)^(-2);
    end;
    
end;
end;

B = AA;
B(1:size(B,1),1:size(B,2)) = 0.5 * tanh(sigma*AA(1:size(B,1),1:size(B,2))+gamma) + 0.5; 

% for ii=1:size(B,1)
%     for kk=1:size(B,2)
%         if B(ii,kk) == 0
%             ssum = 0;
%             for jj=1:size(B,1)
%                 if B(jj,kk) ~= 0
%                     
%                     if kk <= size(AA,1) && jj <= size(AA,2)
%                         ssum = ssum + AA(kk,jj)*B(jj,kk);
%                     end;
%                 end;     
%             end;
%             B(ii,kk) = ssum;
%         end;    
%     end;
% end;    
%         
end