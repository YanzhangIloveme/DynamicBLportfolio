function [rmse1,mad1,map1]=getfa(y,yh)

rmse1=sqrt(mean((y-yh).^2));
mad1=mean(abs(y-yh));
if any(y==0)
  'y=0, no mape'
  map1=NaN;
else
   map1=mean(abs(y-yh)./y);
end   