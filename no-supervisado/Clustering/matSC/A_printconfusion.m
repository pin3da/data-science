function A_printconfusion(CMm,CMs)
nC = size(CMm,1);

fprintf('              |')
for i = 1 : nC
    fprintf('     C%2d      |',i)
end
fprintf('\n')
for j = 1 : nC+1
    fprintf('---------------')
end
fprintf('\n')
for i = 1 : nC
    fprintf('     C%2d      |',i)
    for j = 1 : nC
        
        fprintf('%6.2f+-%5.2f |',CMm(i,j),CMs(i,j))
    end
    fprintf('\n')
    for j = 1 : nC+1
        fprintf('---------------')
    end
    fprintf('\n')
end