function label = loadSPimage(sp)

label = zeros(size(sp,1),size(sp,2));
sp = double(rgb2gray(sp));

for id = 1: 5000

[row,col] = find(label==0, 1);
if(length(row)==0)
break;
end

residual = ones(size(sp))*sp(row,col) ;

res = ((sp - residual) ==0);

lab = bwlabel(res);
idx = find(lab == lab(row,col));

label(idx) = id;

end



end