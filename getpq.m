function [bestp,bestq,bestAIC] = getpq(n)
LOGL = zeros(5,5); %Initialize
PQ   = zeros(5,5);
for p = 1:5
    for q = 1:5
        mod = arima(p,0,q);
        [fit,~,logL] = estimate(mod,n,'display','Off');
        LOGL(p,q) = logL;
        PQ(p,q) = p+q;
     end
end
LOGL = reshape(LOGL,25,1);
PQ = reshape(PQ,25,1);
% using AIC
[aic,~] = aicbic(LOGL,PQ+1,length(n));
bestpq=(reshape(aic,5,5));
[m,i] = min(bestpq);      % i(I)=8 indicates that p=4
[M,I] = min(m)   ;        % I=7  indicates that q=5
bestq = I;
bestp = i(I);
bestAIC = bestpq(bestp,bestq);

end


