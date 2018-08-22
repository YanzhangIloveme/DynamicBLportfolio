dbstop if error
%%
%==================
% +++ Part One : EDA +++
%==================

% Load in the datase
load GrpAsgmData.mat

% Create the price matrix and return matrix
p_mat = [BX CTB PSX JNJ LBTYA];
clear BX CTB PSX JNJ LBTYA

% To gain a basic insight into the original dataset
shape_original = size(p_mat)

% Sepatate the dataset into the in-sample set (~80%) and forecast set (~20%)
s = round(0.8*shape_original(1));
is_p = p_mat(1:s, :);
f_p = p_mat(s+1:end, :);
shape_is = size(is_p)
shape_f = size(f_p)

% Plot the in-sample price series
figure; plot(Date(1:s), is_p); legend('BX', 'CTB', 'PSX', 'JNJ', 'LBTYA', 'location', 'northwest'); grid on;
datetick('x', 'mmm/yy'); title('In-sample Price Series for Five Stocks'); xlim([min(Date(2:s)) max(Date(2:s))])

% Create the return series for both in-sample and forecast price sets
full_ret = price2ret(p_mat);
is_ret = full_ret(1:round(0.8*length(full_ret)), :); %price2ret(is_p);
f_ret = full_ret(round(0.8*length(full_ret))+1:end, :); %price2ret(f_p);
shape_is = size(is_ret)
shape_f = size(f_ret)

% Plot the in-sample return series 
figure; plot(Date(2:s), is_ret); legend('BX', 'CTB', 'PSX', 'JNJ', 'LBTYA', 'location','southwest'); grid on;
title('In-sample Return Series for Five Stocks'); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))])

% Exploratory Data Analysis for in-sample returns
% (1) Descriptive Statistics
stocks = {'BX', 'CTB', 'PSX', 'JNJ', 'LBTYA'};
Mean = [mean(is_ret(:,1)); mean(is_ret(:,2)); mean(is_ret(:,3)); mean(is_ret(:,4)); mean(is_ret(:,5))];
Median = [median(is_ret(:,1)); median(is_ret(:,2)); median(is_ret(:,3)); median(is_ret(:,4)); median(is_ret(:,5))];
STD = [std(is_ret(:,1)); std(is_ret(:,2)); std(is_ret(:,3)); std(is_ret(:,4)); std(is_ret(:,5))];
MIN = [min(is_ret(:,1)); min(is_ret(:,2)); min(is_ret(:,3)); min(is_ret(:,4)); min(is_ret(:,5))];
MAX = [max(is_ret(:,1)); max(is_ret(:,2)); max(is_ret(:,3)); max(is_ret(:,4)); max(is_ret(:,5))];
SKEWNESS = [skewness(is_ret(:,1)); skewness(is_ret(:,2)); skewness(is_ret(:,3)); skewness(is_ret(:,4)); skewness(is_ret(:,5))];
KURTOSIS = [kurtosis(is_ret(:,1)); kurtosis(is_ret(:,2)); kurtosis(is_ret(:,3)); kurtosis(is_ret(:,4)); kurtosis(is_ret(:,5))];
stats_table = table(Mean, Median, STD, MIN, MAX, SKEWNESS, KURTOSIS, 'RowNames',stocks)

% Histograms of return series
figure; subplot(1,5,1); hist(is_ret(:,1), 50); title('Histogram of returns for BX'); 
subplot(1,5,2); hist(is_ret(:,2), 50); title('Histogram of returns for CTB'); 
subplot(1,5,3); hist(is_ret(:,3), 50); title('Histogram of returns for PSX'); 
subplot(1,5,4); hist(is_ret(:,4), 50); title('Histogram of returns for JNJ');
subplot(1,5,5); hist(is_ret(:,5), 50); title('Histogram of returns for LBTYA'); 
% 
% for i = 1:5
%     figure; hist(is_ret(:,i), 50); title(strcat('Histogram of returns for, ', stocks(i))); 
% end

%%
%==================
% +++ Part Two Factor Model +++
%==================

% Conduct factor analysis with m=2 factors
[lam2_rets, psi2_rets, T2_ret, stats2_ret, F2_ret] = factoran(is_ret, 2, 'maxit', 1000, 'rotate', 'none'); % with 1000 iterations

% Display standardised factor loadings and specifc error variances
std_fl = table([lam2_rets(:,1)], [lam2_rets(:,2)], [psi2_rets], 'RowNames', stocks,...
    'VariableNames',{'F1_Std_Loadings','F2_Std_Loadings','Error_Variance'})

% Display the two columns of actual factor loadings, then specific error
% variances, SER and adjusted R-squared for each industry series from the
% factor model
lam2_ret = lam2_rets;
lam2_ret(:,1) = lam2_rets(:,1).*(std(is_ret))';
lam2_ret(:,2) = lam2_rets(:,2).*(std(is_ret))';
psi2_ret = psi2_rets.*(var(is_ret))';
act_fl = table([lam2_ret(:,1)], [lam2_ret(:,2)], [psi2_ret], [sqrt(psi2_ret)], [(1-psi2_ret'./var(is_ret))'], ...
    'VariableNames',{'F1_Loadings','F2_Loadings','Spc_Err_Var', 'SER', 'Adj_R2'}, 'RowNames', stocks)

% Overall amount of variance explained by the factor model
var_explained = (trace(cov(is_ret)) - sum(psi2_ret))/trace(cov(is_ret))

% Stats from the factor analysis
stats2_ret

% Estimated error variances, and sample variances, for each asset 
err_var = table([psi2_ret], [var(is_ret)'], 'VariableNames',{'Est_Var','Sample_Var'}, 'RowNames', stocks)

% Combine those results
fa_table = table([lam2_ret(:,1)], [lam2_ret(:,2)],[psi2_ret],[var(is_ret)'],[sqrt(psi2_ret)],[(1-psi2_ret'./var(is_ret))'], ...
    'VariableNames',{'F1_Loadings','F2_Loadings','Spc_Err_Var', 'Sam_Var', 'SER', 'Adj_R2'}, 'RowNames', stocks)

% Sample means and variance of 2 factors
mean_2f = mean(F2_ret)
var_2f = var(F2_ret)

% SER, R-squared and STD for each asset
est_table = table([sqrt(psi2_ret)], [(1-psi2_ret'./var(is_ret))'], [std(is_ret)'], 'RowNames', stocks, ...
    'VariableNames',{'SER','R_squared', 'STD'})

% Plots
figure; subplot(3,1,1); plot(Date(2:s), is_ret);
title('In-sample Return Series for Five Stocks'); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
subplot(3,1,2); plot(Date(2:s), F2_ret(:,1)); datetick('x', 'mmm/yy'); 
xlim([min(Date(2:s)) max(Date(2:s))]); title('In-sample Series for 1st Factor');
subplot(3,1,3); plot(Date(2:s), F2_ret(:,2)); datetick('x', 'mmm/yy'); 
xlim([min(Date(2:s)) max(Date(2:s))]); title('In-sample Series for 2nd Factor');

% Correlations between asset return and factors
corr_vbls = {'BX', 'CTB', 'PSX', 'JNJ', 'LBTYA', 'F1', 'F2'};
array2table(corr([is_ret F2_ret]), 'VariableNames', corr_vbls, 'RowNames', corr_vbls)

% Create biplot of two factors
figure; biplot(lam2_ret, 'varlabels', stocks, 'LineWidth', 2, 'Markersize', 20); title('Biplot for Unrotated Factors from MLE');

% Carry out factor rotation
rot_mle = rotatefactors(lam2_ret); 
figure; biplot(rot_mle, 'varlabels', stocks, 'LineWidth', 2, 'Markersize', 20); title('Biplot for Rotated Factors from MLE');


% To further carry out Principal Component Analysis on the original return
% series
[pc_ret, score_ret, latent_ret] = pca(is_ret);

% Percentage variance and cumulative variance explained per component
pct_var = latent_ret./sum(latent_ret);
cum_var = cumsum(latent_ret)./sum(latent_ret);
pca_var = table([pct_var], [cum_var], 'RowNames', {'PC1', 'PC2', 'PC3', 'PC4', 'PC5'},...
    'VariableNames', {'Pct_Var_Explained','Cum_Var_Explained'})

% Create biplot for PC1 & 2 as well as PC1 to 3
figure; biplot(pc_ret(:,1:2), 'varlabels', stocks, 'LineWidth', 1, 'Markersize', 20); title('2-D Biplot for Unrotated Factors from PCA');
figure; biplot(pc_ret(:,1:3), 'varlabels', stocks, 'LineWidth', 1, 'Markersize', 20); title('3-D Biplot for Unrotated Factors from PCA');

% Carry out factor rotation on the first three PCs
rot_pca = rotatefactors(pc_ret(:,1:3), 'Method','orthomax'); 
figure; biplot(rot_pca(:,1:2), 'varlabels', stocks, 'LineWidth', 1, 'Markersize', 20); title('2-D Biplot for Rotated Factors from PCA');
figure; biplot(rot_pca(:,1:3), 'varlabels', stocks, 'LineWidth', 1, 'Markersize', 20); title('3-D Biplot for Rotated Factors from PCA');

% Plot of 5 return series together along with first 3 components
figure; subplot(4,1,1); plot(Date(2:s), is_ret); 
title('In-sample Return Series for Five Stocks'); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
subplot(4,1,2); plot(Date(2:s), score_ret(:,1)); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
title('In-sample Series for 1st Principal Component');
subplot(4,1,3); plot(Date(2:s), score_ret(:,2)); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
title('In-sample Series for 2nd Principal Component');
subplot(4,1,4); plot(Date(2:s), score_ret(:,3)); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
title('In-sample Series for 3rd Principal Component');

% Correlation table of all 5 components
corr_pca = {'PC1', 'PC2', 'PC3', 'PC4', 'PC5'};
array2table(corr(score_ret), 'VariableNames', corr_pca, 'RowNames', corr_pca)

% Since the first three factors capture most variance, we attempt to choose
% these three factors to construct a 3-factor model
xmat = [ones(length(is_ret), 1) score_ret(:,1:3)]; % Create X matrix for regression wifactorth the first three PCs

% Iterate over each returns series to fit the factor model
for i=1:5    
    [betas, betaCI, resid, residCI, stat] = regress(is_ret(:,i), xmat); % run OLS regression
    reg_table(i,:) = table(betas(1),betaCI(1,:),betas(2),betaCI(2,:),betas(3),betaCI(3,:),...
        betas(4),betaCI(4,:),sqrt(stat(4)),(1-stat(4)/var(is_ret(:,i))));    
end
reg_table.Properties.VariableNames = {'Alpha','Alpha_95CI','B1','B1_95CI','B2','B2_95CI','B3','B3_95CI','SER','AdjR2'};
reg_table.Properties.RowNames = stocks

%%
%=========================================================
% +++ Part 3&4 build model and forecast(i) ad-hoc: +++
%=========================================================

ad_hoc_returns = zeros(length(f_ret),5);


for t=1:length(f_ret)
    ret_is = full_ret(t:length(f_ret)+t-1,:);
    ad_hoc_returns(t,:) = mean(ret_is(end-5:end,:));
end


rmse_adhoc =zeros(5,1);
mad_adhoc =zeros(5,1);
for i=1:5
    
    [rmse_adhoc(i,1),mad_adhoc(i,1)]=getfa(ad_hoc_returns(:,i),f_ret(:,i))
end

%%
%===================================================
% +++ Part 3 build model and forecast(ii) ARMA: +++
%===================================================
lbqtestReturn_Pvalues_5     = zeros(5,1);
lbqtestReturn_Pvalues_10    = zeros(5,1);
lbqtestResiduals_Pvalues_15 = zeros(5,1);
lbqtestResiduals_Pvalues_20 = zeros(5,1);
SkewnessResiduals           = zeros(5,1);
KurtosisResiduals           = zeros(5,1);
jbtestResiduals_Pvalues     = zeros(5,1);
bestp                       = zeros(5,1);
bestq                       = zeros(5,1);
minaic                      = zeros(5,1);
arma_returns                = zeros(length(f_ret),5);

for i=1:5
    series = is_ret(:,i);
    %1. lbtest for all return series;
    [H5, pval5]                   = lbqtest(series, 5, 0.05);
    [H10, pval10]                 = lbqtest(series, 10, 0.05);
    lbqtestReturn_Pvalues_5(i,1)  = pval5;
    lbqtestReturn_Pvalues_10(i,1) = pval10;
    %2. find best p and q
    [p,q,best]=getpq(series);
    bestp(i,1)                    = p;
    bestq(i,1)                    = q;
    minaic(i,1)                   = best;
    %3. estimate models
    Mdl=arima(p,0,q);
    [EstMdl,EstParamCov,logL,info] = estimate(Mdl,series); % estimates the model
    %4. get residuals % df
    [E0,V0] = infer(EstMdl,series);
    %5. JB test
    SkewnessResiduals(i,1)        = skewness(E0);
    KurtosisResiduals(i,1)        = kurtosis(E0);
    [h,pvalues] = jbtest(E0);
    jbtestResiduals_Pvalues(i,1)  = pvalues;
    %6. lbqtestResidua 15 lags
    [H15, pval15]                 = lbqtest(E0, 15, 0.05,15-p-q);
    [H20, pval20]                 = lbqtest(E0, 20, 0.05,20-p-q);
    lbqtestResiduals_Pvalues_15(i,1)   = pval15;
    lbqtestResiduals_Pvalues_20(i,1)   = pval20;
    %7. Moving horizon forecasts (update weekly); 
    [arma_returns(1,i)] = forecast(EstMdl,1,'Y0',series); % 1-period forecasts
    for t=2:length(f_ret)
        ret_is = full_ret(t:length(series)+t-1,i);
        if mod(t,5)==0  
            Mdl=arima(p,0,q);[EstMdl,EstParamCov,logL,info] = estimate(Mdl,ret_is,'display','Off');
        end
        [arma_returns(t,i)] = forecast(EstMdl,1,'Y0',ret_is);
    end
end

% Create Table to record the test results
final_results = array2table([bestp,bestq,minaic,lbqtestReturn_Pvalues_5, ...,
lbqtestReturn_Pvalues_10, lbqtestResiduals_Pvalues_15,lbqtestResiduals_Pvalues_20, ...,
jbtestResiduals_Pvalues,SkewnessResiduals,KurtosisResiduals]);


final_results.Properties.VariableNames = {'p' 'q' 'minimumaic' 'Pv_lbq_returns_5' 'Pv_lbq_returns_10' ...,
'Pv_lbq_residuals_15' 'Pv_lbq_residuals_20' 'Pv_JBtest_Err' 'Skewness_Err' 'Kurtosis_Err'};
final_results.Properties.RowNames = {'BX' 'CTB' 'JNJ' 'LBTYA' 'PSX'} % not reject H0

% Calculate RMSE and MAE
rmse_arma = zeros(5,1);
mad_arma  = zeros(5,1);
for i=1:5
    [rmse_arma(i,1),mad_arma(i,1)]=getfa(arma_returns(:,i),f_ret(:,i));
end


%% plot ARMA predicions:
for i=1:5
    figure;plot(Date(s+1:end),arma_returns(:,i));hold on;
    plot(Date(s+1:end),f_ret(:,i));
    legend('ARMA','observations')
end

%%
%===================================================
% +++ Part 3 build model and forecast(iii) ARCH: +++
%===================================================

%ACF; LB; Engle's ARCH test(use PSX as example)
figure;subplot(3,1,1);plot(Date(2:s),is_ret(:,3));ylabel('log return')
title('In sample PSX returns'); % add title
subplot(3,1,2);autocorr(is_ret(:,3), 30);
title('In sample PSX returns ACF');
subplot(3,1,3);autocorr(power(is_ret(:,3),2), 30);
title('In sample PSX returns squares ACF');

%LBQtest on squared returns
[H5, pValue5, Qstat5, CriticalValue5] = lbqtest(is_ret(:,3).^2, 5, 0.05);  % 5 lags
[H10, pValue10, Qstat10, CriticalValue10] = lbqtest(is_ret(:,3).^2, 10, 0.05); % 10 lags
results = array2table([[H5,pValue5,Qstat5,CriticalValue5];
[H10,pValue10,Qstat10,CriticalValue10]]);
results.Properties.VariableNames = {'Reject' 'Pvalue' 'statistics' 'CriticalValues'};
results.Properties.RowNames = {'5lags' '10lags'}
% all of test reject H0: ARCH effect exist

% ARCH test on mean-corrected returns
a=is_ret(:,3)-mean(is_ret(:,3));
[H5,pValue5,ARCHstat5,CriticalValue5] = archtest(a,5)
[H10,pValue10,ARCHstat10,CriticalValue10] = archtest(a,10)
% Reject H0!
results = array2table([[H5,pValue5,ARCHstat5,CriticalValue5]
[H10,pValue10,ARCHstat10,CriticalValue10]])
results.Properties.VariableNames = {'Reject' 'Pvalue' 'statistics' 'CriticalValues'};
results.Properties.RowNames = {'5lags' '10lags'}

% check normal-dist ARMA 
LLF=0;aic1=0;sic1=0;
for p=1:20
  Mdl = garch(0,p);Mdl.Offset=NaN;
  [EstMdl,EstParamCov,logL,info] = estimate(Mdl,is_ret(:,3),'display','off');
  aic1(p)=-2*logL+2*p;sic1(p)=-2*logL+log(length(is_ret(:,3)))*p;
end
figure;plot(aic1,'b+-');hold on;plot(sic1,'r+-');
title('AIC & SIC');legend('AIC','SIC');
[m,i]=min(aic1)
[m,i]=min(sic1)
%aic1 : p=10
%sic1 : p=10
Mdl = garch(0,10);Mdl.Offset=NaN;
EstMdl = estimate(Mdl,is_ret(:,3));
v10=infer(EstMdl,is_ret(:,3));s10=sqrt(v10); %infer the conditional variance and calculate standard deviations
a10 = is_ret(:,3)-EstMdl.Offset; 
a1A=EstMdl.ARCH; 
% assess the fit
e10=a10./s10;  %standarised error
figure;subplot(2,1,1);plot(Date(2:s),e10);
title('ARCH(10) Standardised Residuals');
subplot(2,1,2);hist(e10,25)
title('Histogram of ARCH(10) Standardised Residuals');
% residuals and squared residuals acf
subplot(2,1,1);autocorr(e10);
title('ACF of ARCH(10) Standardised Residuals');
subplot(2,1,2);autocorr(e10.^2);
title('ACF of  ARCH(10) Squared Standardised Residuals');
%qq plot
figure;
qqplot(e10);
title('QQ plot ARCH(10) Standardised Residuals');
[H, pValue, Qstat, CriticalValue] = lbqtest(e10, [15 20], 0.05, [5 10]);
% need AR term
% df: lags+17
%LB test on squared standardised residuals
[H, pValue, Qstat, CriticalValue] = lbqtest(e10.^2, [15 20], 0.05, [5 10]);
% dont need ARCH term
% JB test
[skewness(e10) kurtosis(e10)];
[h,p] = jbtest(e10);

% the results showing t-dist is better

%% ARCH with t-distribution (use PSX as example)
% 
logL=0;aic1=0;sic1=0;
for p=1:20
  Mdl = garch(0,p);Mdl.Offset=NaN;Mdl.Distribution='t';
 [EstMdl,EstParamCov,logL,info] = estimate(Mdl,is_ret(:,3),'display','off');
 aic1(p)=-2*logL+2*p;sic1(p)=-2*logL+log(length(is_ret(:,3)))*p;
end
figure;plot(aic1,'b+-');title('AIC & SIC for t-distribution ARCH models')
hold on;plot(sic1,'r+-');legend('AIC','SIC');

% sic =2;   aic =10
[m,i]=min(sic1);
[m,i]=min(aic1);
% use aic
Mdl = garch(0,10);Mdl.Offset=NaN;Mdl.Distribution='t';
EstMdl = estimate(Mdl,is_ret(:,3));
v10t=infer(EstMdl,is_ret(:,3));s10t=sqrt(v10t); %infer the conditional variance and calculate standard deviations
a10t = is_ret(:,3)-EstMdl.Offset; %calculate innovations
a1At=EstMdl.ARCH; %store ARCH coefficients for later use
%plot conditional statistics
figure;subplot(3,1,1);plot(Date(2:s),a10t);
title('ARCH(10) t-dist Innovations');
subplot(3,1,2);plot(Date(2:s),s10t);
title('ARCH(10) t-dist Conditional Standard Deviations');
subplot(3,1,3);plot(Date(2:s),is_ret(:,3));
title('Log returns')
% plot the results
figure;plot(Date(2:s),is_ret(:,3),'c+-.');hold on;plot(Date(2:s),s10t);
xlim([Date(2) Date(s)]);      % set range of x-axis
title('ARCH(1) Conditional Standard Deviations and BHP Returns');
legend('BX returns', 'conditional standard deviations','location',...
    'South','Orientation','horizontal' );

% check the fitness:
e10t=a10t./s10t;

figure;subplot(2,1,1);plot(Date(2:s),e10t)
xlim([Date(2) Date(s)]);             % set range of x-axis
title('ARCH(10)-t Standardised Residuals');
subplot(2,1,2);autocorr(e10t)
title('ACF of ARCH(10)-t Standardised Residuals');

%qq
figure;subplot(2,1,1);hist(e10t,25)
title('Histogram of ARCH(10)-t Standardised Residuals');
subplot(2,1,2);qqplot(e10t);
title('QQ plot ARCH(10)-t Standardised Residuals');

% transform t-errors to normal errors
df=EstMdl.Distribution.DoF; % Get estimated degree of freedom parameter
ge=norminv(tcdf(sqrt(df)/sqrt(df-2)*e10t,df)); % transform t-errors to normal errors

% Plot Transformed Standardised Residuals and ACF
figure;subplot(2,1,1);plot(Date(2:s),ge)
xlim([Date(2) Date(s)]);              % set range of x-axis
title('ARCH(10)-t Transformed Standardised Residuals');
subplot(2,1,2);autocorr(ge)
title('ACF of ARCH(10)-t Transformed Standardised Residuals');

% Plot Histogram of Transformed Standardised Residuals and QQ plot
figure;subplot(2,1,1);hist(ge,25)
title('Histogram of ARCH(10)-t Transformed Standardised Residuals');
subplot(2,1,2);qqplot(ge);
title('QQ plot ARCH(10)-t Transformed Standardised Residuals');

% Plot ACF of Squared Transformed Standardised Residuals
figure;autocorr(ge.^2)
title('ACF of ARCH(10)-t Squared Transformed Standardised Residuals');
[H, pValue, Qstat, CriticalValue] = lbqtest(ge, [15 20], 0.05, [5 10])
% LB test on squared transformed standardised residuals
[H, pValue, Qstat, CriticalValue] = lbqtest(ge.^2, [15 20], 0.05, [5 10])

[skewness(ge) kurtosis(ge)]
[h,p] = jbtest(ge)
figure;plot(Date(2:s),s10,'b');
hold on;plot(Date(2:s),s10t,'r');             % set range of x-axis
legend('ARCH(10)','ARCH(10)-t');
title('Conditional Standard Deviations for Gaussian and t ARCH models');

%% ARCH predictions
lbqtestSquare_Return_Pvalues_5     = zeros(5,1);
lbqtestSquare_Return_Pvalues_10    = zeros(5,1);
ARCHtest_5                         = zeros(5,1);
ARCHtest_10                        = zeros(5,1);
bestp_t_dist                       = zeros(5,1);
minaic_t_dist                      = zeros(5,1);
minsic_t_dist                      = zeros(5,1);
lbqtestResiduals_t_dist_Pvalues_15 = zeros(5,1);
lbqtestResiduals_t_dist_Pvalues_20 = zeros(5,1);
lbqtestSquareRes_t_dist_Pvalues_15 = zeros(5,1);
lbqtestSquareRes_t_dist_Pvalues_20 = zeros(5,1);
SkewnessResiduals_t_dist           = zeros(5,1);
KurtosisResiduals_t_dist           = zeros(5,1);
jbtestResiduals_Pvalues_t_dist     = zeros(5,1);
sigma                              = zeros(length(f_ret),5);
ARCH_VaR_f                         = zeros(length(f_ret),5);
SFAOgt                             = zeros(length(f_ret),5);

for i=1:5
    series = is_ret(:,i);
    %1. lbtest for all series;
    [H5, pval5]                   = lbqtest(series.^2, 5, 0.05);
    [H10, pval10]                 = lbqtest(series.^2, 10, 0.05);
    lbqtestSquare_Return_Pvalues_5(i,1)  = pval5;
    lbqtestSquare_Return_Pvalues_10(i,1) = pval10;
    % arch test
    a=series-mean(series);
    [H5,pValue_arch_5,ARCHstat5,CriticalValue5] = archtest(a,5);
    [H10,pValue_arch_10,ARCHstat10,CriticalValue10] = archtest(a,10);
    ARCHtest_5(i,1)  = pValue_arch_5;
    ARCHtest_10(i,1) = pValue_arch_10;
    %2. find best p
    logL=0;aic1=0;sic1=0;
    for p=1:20
        Mdl = garch(0,p);Mdl.Offset=NaN;Mdl.Distribution='t';
        [EstMdl,EstParamCov,logL,info] = estimate(Mdl,series,'display','off');
        aic1(p)=-2*logL+2*p;sic1(p)=-2*logL+log(length(series))*p;
    end
    [m_sic,i_sic]=min(sic1);
    [m_aic,i_aic]=min(aic1);
    
    bestp_t_dist(i,1)                      = i_aic;
    minaic_t_dist(i,1)                     = m_aic;
    minsic_t_dist(i,1)                     = m_sic;
    %3. fit the arch(p) using AIC
    Mdl = garch(0,i_aic);Mdl.Offset=NaN;Mdl.Distribution='t';
    EstMdl = estimate(Mdl,series);
    vpt=infer(EstMdl,series);spt=sqrt(vpt); %infer the conditional variance and calculate standard deviations
    apt = series-EstMdl.Offset;
    ept=apt./spt;
    %4. transfrom the residuals to normal-dist
    df=EstMdl.Distribution.DoF; % Get estimated degree of freedom parameter
    ge=norminv(tcdf(sqrt(df)/sqrt(df-2)*ept,df)); 
    
    [H, pValue_r, Qstat, CriticalValue] = lbqtest(ge, [15 20], 0.05, [5 10]);
    lbqtestResiduals_t_dist_Pvalues_15(i,1) = pValue_r(1);
    lbqtestResiduals_t_dist_Pvalues_20(i,1) = pValue_r(2);
    
    [H, pValue_r2, Qstat, CriticalValue] = lbqtest(ge.^2, [15 20], 0.05, [5 10]);
    lbqtestSquareRes_t_dist_Pvalues_15(i,1) = pValue_r2(1);
    lbqtestSquareRes_t_dist_Pvalues_20(i,1) = pValue_r2(2);
    %jb test
    [h,p_jb] = jbtest(ge);
    SkewnessResiduals_t_dist(i,1)           =skewness(ge);
    KurtosisResiduals_t_dist(i,1)           =kurtosis(ge); 
    jbtestResiduals_Pvalues_t_dist(i,1)     =p_jb;
    
    %5. predictios VaR and sigma
    for t=1:length(f_ret)
        ret_is = full_ret(t:length(is_ret)+t-1,i);
        if mod(t,5)==0|t==1
            Mdl = garch(0,i_aic);Mdl.Offset=NaN;Mdl.Distribution='t'; %ARCH(5)
            [EstMdl,EstParamCov,LLF,info]=estimate(Mdl,ret_is,'display','off');
            %[E0,V0] = infer(EstMdl,series);
        end
        sigma(t,i)=forecast(EstMdl,1,'Y0',ret_is);
        p0Gt=EstMdl.Offset;
        dfGt   = EstMdl.Distribution.DoF;
        SFAOgt(t,i) = sqrt(sigma(t,i));
        ARCH_VaR_f(t,i) = p0Gt+tinv(0.05,dfGt)*SFAOgt(t,i)*sqrt((dfGt-2)/dfGt);
        
    end
end

% create table of the results

after = array2table([bestp_t_dist, minaic_t_dist,minsic_t_dist,lbqtestResiduals_t_dist_Pvalues_15 ...,
    lbqtestResiduals_t_dist_Pvalues_20,lbqtestSquareRes_t_dist_Pvalues_15 ...,
    lbqtestSquareRes_t_dist_Pvalues_20,SkewnessResiduals_t_dist ...,
    KurtosisResiduals_t_dist,jbtestResiduals_Pvalues_t_dist]);
after.Properties.VariableNames = {'p' 'minimumAIC' 'minimumSIC' 'Pv_lb_ERR_15' 'Pv_lb_ERR_20' ...,
   'Pv_lb_squ_ERR_15' 'Pv_lb_squ_ERR_20' 'kewness_Err' 'Kurtosis_Err' 'Pv_jb_ERR'};
after.Properties.RowNames = {'BX' 'CTB' 'PSX' 'JNJ' 'LBTYA' }


%%
%===================================================
% +++ Part 3 build model and forecast(iv) GARCH: +++
%===================================================

% Use AIC and BIC to choose a suitable GARCH(p,q) model with either Gaussian or Student-t errors.
aic0 = 0; bic0 = 0; aict0 = 0; bict0 = 0; LLF = 0;

for i=1:5
    for p=1:5
        for q=1:5
            % GARCH(p,q) with Gaussian errors
            mdlGG = garch(p,q); mdlGG.Offset = NaN; mdlGG.Distribution = 'Gaussian';
            [EstMdl, EstParamCov, LLF, info] = estimate(mdlGG, is_ret(:,i), 'display', 'off');
            aic0(p,q) = -2*LLF+2*(p+q);
            bic0(p,q) = -2*LLF+log(length(is_ret))*(p+q);
            [p_aic0opt, q_aic0opt] = find(aic0 == min(min(aic0)));
            [p_bic0opt, q_bic0opt] = find(bic0 == min(min(bic0)));
            
            % GARCH(p,q) with Student-t errors
            mdlGT = garch(p,q); mdlGT.Offset = NaN; mdlGT.Distribution = 't';
            [EstMdl, EstParamCov, LLF, info] = estimate(mdlGT, is_ret(:,i), 'display', 'off');
            aict0(p,q) = -2*LLF+2*(p+q+1);
            bict0(p,q) = -2*LLF+log(length(is_ret))*(p+q+1);
            [p_aict0opt, q_aict0opt] = find(aict0 == min(min(aict0)));
            [p_bict0opt, q_bict0opt] = find(bict0 == min(min(bict0)));
            
            garch_optpq(i,:) = table([p_aic0opt, q_aic0opt, min(min(aic0))], ...
                [p_bic0opt, q_bic0opt, min(min(bic0))], ...  
                [p_aict0opt, q_aict0opt, min(min(aict0))], ...
                [p_bict0opt, q_bict0opt, min(min(bict0))]);
        end
    end
end

garch_optpq.Properties.VariableNames = {'GARCH_AIC','GARCH_BIC','GARCH_t_AIC','GARCH_t_BIC'};
garch_optpq.Properties.RowNames = stocks 

%===================================================
% +++ Part 3 build model and forecast(v) AR-GARCH: +++
%===================================================

% Use AIC and BIC to choose a suitable AR(1)-GARCH(p,q) or AR(2)-GARCH(p,q) model with either Gaussian or Student-t errors.
aic1 = 0; bic1 = 0; aict1 = 0; bict1 = 0; aic2 = 0; bic2 = 0; aict2 = 0; bict2 = 0;LLF = 0;

for i=1:5
    for p=1:5
        for q=1:5
            % AR(1)-GARCH(p,q) with Gaussian errors
            mdlAGG = arima('ARLags', 1, 'Variance', garch(p,q)); mdlAGG.Distribution = 'Gaussian';
            [EstMdl, EstParamCov, LLF, info] = estimate(mdlAGG, is_ret(:,i), 'display', 'off');
            aic1(p,q) = -2*LLF+2*(p+q+1);
            bic1(p,q) = -2*LLF+log(length(is_ret))*(p+q+1);
            [p_aic1opt, q_aic1opt] = find(aic1 == min(min(aic1)));
            [p_bic1opt, q_bic1opt] = find(bic1 == min(min(bic1)));

            % AR(1)-GARCH(p,q) with Studnet-t errors
            mdlAGT = arima('ARLags', 1, 'Variance', garch(p,q)); mdlAGG.Distribution = 't';
            [EstMdl, EstParamCov, LLF, info] = estimate(mdlAGT, is_ret(:,i), 'display', 'off');
            aict1(p,q) = -2*LLF+2*(p+q+2);
            bict1(p,q) = -2*LLF+log(length(is_ret))*(p+q+2);
            [p_aict1opt, q_aict1opt] = find(aict1 == min(min(aict1)));
            [p_bict1opt, q_bict1opt] = find(bict1 == min(min(bict1)));
            
            % AR(2)-GARCH(p,q) with Gaussian errors
            mdlAGG = arima('ARLags', 2, 'Variance', garch(p,q)); mdlAGG.Distribution = 'Gaussian';
            [EstMdl, EstParamCov, LLF, info] = estimate(mdlAGG, is_ret(:,i), 'display', 'off');
            aic2(p,q) = -2*LLF+2*(p+q+2);
            bic2(p,q) = -2*LLF+log(length(is_ret))*(p+q+2);
            [p_aic2opt, q_aic2opt] = find(aic2 == min(min(aic2)));
            [p_bic2opt, q_bic2opt] = find(bic2 == min(min(bic2)));

            % AR(2)-GARCH(p,q) with Studnet-t errors
            mdlAGT = arima('ARLags', 2, 'Variance', garch(p,q)); mdlAGG.Distribution = 't';
            [EstMdl, EstParamCov, LLF, info] = estimate(mdlAGT, is_ret(:,i), 'display', 'off');
            aict2(p,q) = -2*LLF+2*(p+q+3);
            bict2(p,q) = -2*LLF+log(length(is_ret))*(p+q+3);
            [p_aict2opt, q_aict2opt] = find(aict2 == min(min(aict2)));
            [p_bict2opt, q_bict2opt] = find(bict2 == min(min(bict2)));
            
            argarch_optpq(i,:) = table([p_aic1opt, q_aic1opt, min(min(aic1))], ...
                [p_bic1opt, q_bic1opt, min(min(bic1))], ...
                [p_aict1opt, q_aict1opt, min(min(aict1))], ...
                [p_bict1opt, q_bict1opt, min(min(bict1))], ...
                [p_aic2opt, q_aic2opt, min(min(aic2))], ...
                [p_bic2opt, q_bic2opt, min(min(bic2))], ...
                [p_aict2opt, q_aict2opt, min(min(aict2))], ...
                [p_bict2opt, q_bict2opt, min(min(bict2))]);
        end
    end
end

argarch_optpq.Properties.VariableNames = {'AR1_GARCH_AIC','AR1_GARCH_BIC',...
    'AR1_GARCH_t_AIC','AR1_GARCH_t_BIC', 'AR2_GARCH_AIC','AR2_GARCH_BIC', ...
    'AR2_GARCH_t_AIC','AR2_GARCH_t_BIC'};
argarch_optpq.Properties.RowNames = stocks 

%===================================================
% +++ Part 3 build model and forecast(vi) GJR-GARCH: +++
%===================================================

% Use AIC and BIC to choose a suitable GJR-GARCH(p,q) model with either Gaussian or Student-t errors.
aic3 = 0; bic3 = 0; aict3 = 0; bict3 = 0; LLF = 0;

for i=1:5
    for p=1:5
        for q=1:5  
            
            % GJR-GARCH(p,q) with Gaussian errors
            mdlGJR = gjr(p,q); mdlGJR.Offset = NaN; mdlGJR.Distribution = 'Gaussian';
            [EstMdl, EstParamCov, LLF, info] = estimate(mdlGJR, is_ret(:,i), 'display', 'off');
            aic3(p,q) = -2*LLF+2*(p+q);
            bic3(p,q) = -2*LLF+log(length(is_ret))*(p+q);
            [p_aic3opt, q_aic3opt] = find(aic3 == min(min(aic3)));
            [p_bic3opt, q_bic3opt] = find(bic3 == min(min(bic3)));
            
            % GJR-GARCH(p,q) with Student-t errors
            mdlGJRt = gjr(p,q); mdlGJRt.Distribution='t'; mdlGJRt.Offset=NaN; % specify model 
            [EstMdl,EstParamCov,LLF,info] = estimate(mdlGJRt, is_ret(:,i), 'display', 'off');
            aict3(p,q) = -2*LLF+2*(p+q+1);
            bict3(p,q) = -2*LLF+log(length(is_ret))*(p+q+1);
            [p_aict3opt, q_aict3opt] = find(aict3 == min(min(aict3)));
            [p_bict3opt, q_bict3opt] = find(bict3 == min(min(bict3)));
            
            gjrgarch_optpq(i,:) = table([p_aic3opt, q_aic3opt, min(min(aic3))], ...
                [p_bic3opt, q_bic3opt, min(min(bic3))], ... 
                [p_aict3opt, q_aict3opt, min(min(aict3))], ...
                [p_bict3opt, q_bict3opt, min(min(bict3))]);
        end
    end
end

gjrgarch_optpq.Properties.VariableNames = {'GJR_GARCH_AIC','GJR_GARCH_BIC','GJR_GARCH_t_AIC','GJR_GARCH_t_BIC'};
gjrgarch_optpq.Properties.RowNames = stocks 

%%
% Then, we choose to fit GARCH(1,1)-t to all series, as chosen by BIC
% AND, we choose to fit AR-GARCH(p,q)-t to different series by AIC
% AND, we choose to fit GJR-GARCH(1,1)-t to all series, as chosen by BIC

% Create variables to store conditional variance for each model
sigGt = zeros(length(is_ret), 1); 
sigAGt = zeros(length(is_ret), 1); 
sigGJRt = zeros(length(is_ret), 1);

for i=1:5

    % Fit the GARCH(1,1)-t model to PSX series
    mdlGT = garch(1,1); mdlGT.Offset = NaN; mdlGT.Distribution = 't';
    [EstMdl_gt, EstParamCov, LLF, info] = estimate(mdlGT, is_ret(:,i));

    % Infer the conditional variance and calculate standard deviations 
    v = infer(EstMdl_gt, is_ret(:,i)); sd = sqrt(v);
    invt = is_ret(:,i) - EstMdl_gt.Offset; % Innovation calculation
    s_rsd = invt./sd; % Standardised residuals 
    
    sigGt = [sigGt sd]; 
    
    % Plot the standardised residuals, conditional standard deviations and log returns
    figure; subplot(3,1,1); plot(Date(2:s), s_rsd); 
    title(strcat('GARCH(1,1)-t Standardised Residuals,', stocks(i))); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    subplot(3,1,2); plot(Date(2:s), sd); 
    title(strcat('GARCH(1,1)-t Conditional Standard Deviations,', stocks(i))); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    subplot(3,1,3); plot(Date(2:s), is_ret(:,i)); 
    title(strcat('Log Returns,', stocks(i))); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);

    figure; plot(Date(2:s), is_ret(:,i), 'c'); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    hold on; plot(Date(2:s), sd, 'r'); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    title(strcat('Log Returns and GARCH(1,1)-t Conditional Standard Deviations,', stocks(i)));
    legend('Log returns', 'Conditional standard deviations', 'location','south','orientation','horizontal');

    df = EstMdl_gt.Distribution.DoF;

    % Assess fit to data
    e1 = sqrt(df)/sqrt(df-2)*s_rsd; % This should have a Student-t with df degrees of freedom
    eg = norminv(tcdf(e1, df)); % Transform to normal errors for diagnostic analysis

    figure; subplot(2,1,1); plot(Date(2:s), eg); 
    title(strcat('GARCH(1,1)-t Standardised Residuals,', stocks(i))); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    title(strcat('GARCH(1,1)-t Transformed Standardised Residuals,', stocks(i))); 
    subplot(2,1,2); autocorr(eg);

    figure; subplot(2,1,1); hist(eg, 30);
    title(strcat('Histogram of GARCH(1,1)-t Transformed Standardised Residuals,', stocks(i))); 
    subplot(2,1,2); qqplot(eg);
    title(strcat('Q-Q Plot of GARCH(1,1)-t Transformed Standardised Residuals, ', stocks(i))); 
    figure; autocorr(eg.^2);
    title(strcat('ACF of GARCH(1,1)-t Squared Transformed Standardised Residuals, ', stocks(i))); 

    % LB test on standardised residuals
    [H_gt1,pValue_gt1,Qstat_gt1,CriVal_gt1] = lbqtest(eg, [7, 12], 0.05, [5 10]);
    % LB test on squared standardised residuals
    [H_gt2,pValue_gt2,Qstat_gt2,CriVal_gt2] = lbqtest(eg.^2, [7, 12], 0.05, [5 10]);

    % JB test 
    skewness_gt = skewness(eg); kurtosis_gt = kurtosis(eg);
    [h_gt, p_gt] = jbtest(eg);
    
    garch_fit(i,:) = table(pValue_gt1(1),pValue_gt1(2),pValue_gt2(1),pValue_gt2(2),p_gt,skewness_gt,kurtosis_gt);
    
        
    if i==1
        % Fit the AR(1)-GARCH(1,1)-t model to BX series
        mdlAGT_bx = arima('ARLags', 1, 'Variance', garch(1,1), 'Distribution', 'T');
        [EstMdl_agt_bx, EstParamCov, LLF, info] = estimate(mdlAGT_bx, is_ret(:,i));
        % Infer the conditional variance and calculate standard deviations 
        [e_agt_bx, v_agt_bx, logL] = infer(EstMdl_agt_bx, is_ret(:,i));
        sd = sqrt(v_agt_bx);
        s_rsd = e_agt_bx./sd; % Standardised residuals 
        df = EstMdl_agt_bx.Distribution.DoF;

    elseif i==2 
        % Fit the AR(1)-GARCH(5,1)-t model to CTB series
        mdlAGT_ctb = arima('ARLags', 1, 'Variance', garch(5,1), 'Distribution', 'T');
        [EstMdl_agt_ctb, EstParamCov, LLF, info] = estimate(mdlAGT_ctb, is_ret(:,i));
        % Infer the conditional variance and calculate standard deviations 
        [e_agt_ctb, v_agt_ctb, logL] = infer(EstMdl_agt_ctb, is_ret(:,i));
        sd = sqrt(v_agt_ctb);
        s_rsd = e_agt_ctb./sd; % Standardised residuals 
        df = EstMdl_agt_ctb.Distribution.DoF;
        
    elseif i==3
        % Fit the AR(1)-GARCH(4,2)-t model to PSX series
        mdlAGT_psx = arima('ARLags', 1, 'Variance', garch(4,2), 'Distribution', 'T');
        [EstMdl_agt_psx, EstParamCov, LLF, info] = estimate(mdlAGT_psx, is_ret(:,i));
        % Infer the conditional variance and calculate standard deviations 
        [e_agt_psx, v_agt_psx, logL] = infer(EstMdl_agt_psx, is_ret(:,i));
        sd = sqrt(v_agt_psx);
        s_rsd = e_agt_psx./sd; % Standardised residuals
        df = EstMdl_agt_psx.Distribution.DoF;
        
    elseif i==4
        % Fit the AR(1)-GARCH(1,1)-t model to JNJ series
        mdlAGT_jnj = arima('ARLags', 1, 'Variance', garch(1,1), 'Distribution', 'T');
        [EstMdl_agt_jnj, EstParamCov, LLF, info] = estimate(mdlAGT_jnj, is_ret(:,i));
        % Infer the conditional variance and calculate standard deviations 
        [e_agt_jnj, v_agt_jnj, logL] = infer(EstMdl_agt_jnj, is_ret(:,i));
        sd = sqrt(v_agt_jnj);
        s_rsd = e_agt_jnj./sd; % Standardised residuals
        df = EstMdl_agt_jnj.Distribution.DoF;
        
    else
        % Fit the AR(1)-GARCH(1,3)-t model to LBYTA series
        mdlAGT_lbyta = arima('ARLags', 1, 'Variance', garch(1,3), 'Distribution', 'T');
        [EstMdl_agt_lbyta, EstParamCov, LLF, info] = estimate(mdlAGT_lbyta, is_ret(:,i));
        % Infer the conditional variance and calculate standard deviations 
        [e_agt_lbyta, v_agt_lbyta, logL] = infer(EstMdl_agt_lbyta, is_ret(:,i));
        sd = sqrt(v_agt_lbyta);
        s_rsd = e_agt_lbyta./sd; % Standardised residuals  
        df = EstMdl_agt_lbyta.Distribution.DoF;
    end 
    
    sigAGt = [sigAGt sd]; 
    
    % Plot the standardised residuals, conditional standard deviations and log returns
    figure; subplot(3,1,1); plot(Date(2:s), s_rsd); 
    title(strcat('AR(1)-GARCH-t Standardised Residuals,', stocks(i))); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    subplot(3,1,2); plot(Date(2:s), sd); 
    title(strcat('AR(1)-GARCH-t Conditional Standard Deviations,', stocks(i))); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    subplot(3,1,3); plot(Date(2:s), is_ret(:,i)); 
    title(strcat('Log Returns,', stocks(i))); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);

    figure; plot(Date(2:s), is_ret(:,i), 'c'); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    hold on; plot(Date(2:s), sd, 'r'); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    title(strcat('Log Returns and AR(1)-GARCH-t Conditional Standard Deviations,', stocks(i)));
    legend('Log returns', 'Conditional standard deviations', 'location','south','orientation','horizontal');

    % Assess fit to data
    e1 = sqrt(df)/sqrt(df-2)*s_rsd; % This should have a Student-t with df degrees of freedom
    eg = norminv(tcdf(e1, df)); % Transform to normal errors for diagnostic analysis

    figure; subplot(2,1,1); plot(Date(2:s), eg); 
    title(strcat('AR(1)-GARCH-t Standardised Residuals,', stocks(i))); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    title(strcat('AR(1)-GARCH-t Transformed Standardised Residuals,', stocks(i))); 
    subplot(2,1,2); autocorr(eg);

    figure; subplot(2,1,1); hist(eg, 30);
    title(strcat('Histogram of AR(1)-GARCH-t Transformed Standardised Residuals,', stocks(i))); 
    subplot(2,1,2); qqplot(eg);
    title(strcat('Q-Q Plot of AR(1)-GARCH-t Transformed Standardised Residuals, ', stocks(i))); 
    figure; autocorr(eg.^2);
    title(strcat('ACF of AR(1)-GARCH-t Squared Transformed Standardised Residuals, ', stocks(i))); 
    
    if i==1 || i==4
        % LB test on standardised residuals
        [H_agt1,pValue_agt1,Qstat_agt1,CriVal_agt1] = lbqtest(eg, [8, 13], 0.05, [5 10]);
        % LB test on squared standardised residuals
        [H_agt2,pValue_agt2,Qstat_agt2,CriVal_agt2] = lbqtest(eg.^2, [8, 13], 0.05, [5 10]);
        
    elseif i==2 || i==3
        % LB test on standardised residuals
        [H_agt1,pValue_agt1,Qstat_agt1,CriVal_agt1] = lbqtest(eg, [12, 17], 0.05, [5 10]);
        % LB test on squared standardised residuals
        [H_agt2,pValue_agt2,Qstat_agt2,CriVal_agt2] = lbqtest(eg.^2, [12, 17], 0.05, [5 10]);
                
    else
        % LB test on standardised residuals
        [H_agt1,pValue_agt1,Qstat_agt1,CriVal_agt1] = lbqtest(eg, [10, 15], 0.05, [5 10]);
        % LB test on squared standardised residuals
        [H_agt2,pValue_agt2,Qstat_agt2,CriVal_agt2] = lbqtest(eg.^2, [10, 15], 0.05, [5 10]);    
    end
            
    % JB test 
    skewness_agt = skewness(eg); kurtosis_agt = kurtosis(eg);
    [h_agt, p_agt] = jbtest(eg);
    
    argarch_fit(i,:) = table(pValue_agt1(1),pValue_agt1(2),pValue_agt2(1),pValue_agt2(2),p_agt,skewness_agt,kurtosis_agt);
        

    % Fit the GJR-GARCH(1,1)-t model to PSX series
    mdlGJRt = gjr(1,1); mdlGJRt.Offset = NaN; mdlGJRt.Distribution = 't';
    [EstMdl_gjrt, EstParamCov, LLF, info] = estimate(mdlGJRt, is_ret(:,i));

    % Infer the conditional variance and calculate standard deviations 
    v = infer(EstMdl_gjrt, is_ret(:,i)); sd = sqrt(v);
    invt = is_ret(:,i) - EstMdl_gjrt.Offset; % Innovation calculation
    s_rsd = invt./sd; % Standardised residuals 

    sigGJRt = [sigGJRt sd]; aGJt = invt; sigGJt = sd;
    
    % Plot the standardised residuals, conditional standard deviations and log returns
    figure; subplot(3,1,1); plot(Date(2:s), s_rsd); 
    title(strcat('GJR-GARCH(1,1)-t Standardised Residuals,', stocks(i))); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    subplot(3,1,2); plot(Date(2:s), sd); 
    title(strcat('GJR-GARCH(1,1)-t Conditional Standard Deviations,', stocks(i))); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    subplot(3,1,3); plot(Date(2:s), is_ret(:,i)); 
    title(strcat('Log Returns,', stocks(i))); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);

    figure; plot(Date(2:s), is_ret(:,i), 'c'); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    hold on; plot(Date(2:s), sd, 'r'); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    title(strcat('Log Returns and GJR-GARCH(1,1)-t Conditional Standard Deviations,', stocks(i)));
    legend('Log returns', 'Conditional standard deviations', 'location','south','orientation','horizontal');

    df = EstMdl_gjrt.Distribution.DoF;

    % Assess fit to data
    e1 = sqrt(df)/sqrt(df-2)*s_rsd; % This should have a Student-t with df degrees of freedom
    eg = norminv(tcdf(e1, df)); % Transform to normal errors for diagnostic analysis

    figure; subplot(2,1,1); plot(Date(2:s), eg); 
    title(strcat('GJR-GARCH(1,1)-t Standardised Residuals,', stocks(i))); datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]);
    title(strcat('GJR-GARCH(1,1)-t Transformed Standardised Residuals,', stocks(i))); 
    subplot(2,1,2); autocorr(eg);

    figure; subplot(2,1,1); hist(eg, 30);
    title(strcat('Histogram of GJR-GARCH(1,1)-t Transformed Standardised Residuals,', stocks(i))); 
    subplot(2,1,2); qqplot(eg);
    title(strcat('Q-Q Plot of GJR-GARCH(1,1)-t Transformed Standardised Residuals, ', stocks(i))); 
    figure; autocorr(eg.^2);
    title(strcat('ACF of GJR-GARCH(1,1)-t Squared Transformed Standardised Residuals, ', stocks(i)));  
    
    % LB test on standardised residuals
    [H_gt1,pValue_gt1,Qstat_gt1,CriVal_gt1] = lbqtest(eg, [9, 14], 0.05, [5 10]);
    % LB test on squared standardised residuals
    [H_gt2,pValue_gt2,Qstat_gt2,CriVal_gt2] = lbqtest(eg.^2, [9, 14], 0.05, [5 10]);

    % JB test 
    skewness_gt = skewness(eg); kurtosis_gt = kurtosis(eg);
    [h_gt, p_gt] = jbtest(eg);
    
    gjrgarch_fit(i,:) = table(pValue_gt1(1),pValue_gt1(2),pValue_gt2(1),pValue_gt2(2),p_gt,skewness_gt,kurtosis_gt);

    % To further create NIC plots for GJR-GARCH(1,1)-t
    if i==2 || i==3
        
        % Get required estimated GJR model coefficients 
        a1GJt=cell2mat(EstMdl_gjrt.ARCH); b1GJt=cell2mat(EstMdl_gjrt.GARCH); 
        g1GJt=cell2mat(EstMdl_gjrt.Leverage); 
        dfGJt=EstMdl_gjrt.Distribution.DoF; 
        a0GJt=EstMdl_gjrt.Constant;

        % Caculate values for NIC plot 
        a=min(aGJt):0.01:max(aGJt); % range of plot
        sg=var(aGJt); % sample variance of innovations
        sigt = a0GJt+a1GJt*a.^2+b1GJt*sg+(g1GJt.*(a<0)).*(a.^2); % asymmetric curve values
        sigt2=a0GJt+(a1GJt+g1GJt/2)*a.^2+b1GJt*sg; % symmetric curve values

        % NIC plot for a(t-1) 
        figure; plot(a,sigt); axis([min(aGJt) max(aGJt) 0 max(sigt)]); hold on;
        plot(a,sigt2,'r--'); legend('Asymmetric Curve','Symmetric Curve'); title(strcat('GJR-GARCH-t NIC curve for a(t-1),', stocks(i)));

        % Compare precited conditional standard deviation values following positive and negative shocks of size 2 
        a=-2; sigtm2=a0GJt+a1GJt*a^2+b1GJt*sg+(g1GJt*(a<0))*(a^2);
        a=2; sigta2=a1GJt+a1GJt*a^2+b1GJt*sg+(g1GJt*(a<0))*(a^2); 
        [sigtm2 sigta2 sigtm2/sigta2];

        % calculate NIC curve - now plotted against standardised shocks 
        eps=aGJt./sigGJt; 
        e=min(eps):0.01:max(eps); 
        a=sqrt(sg)*e; 
        sigt=a0GJt+a1GJt*a.^2+b1GJt*sg+(g1GJt.*(a<0)).*(a.^2); 
        sigt2=a0GJt+(a1GJt+g1GJt/2)*a.^2+b1GJt*sg;

        % Create NIC plot for e(t-1) 
        figure; plot(e,sigt); axis([min(e) max(e) 0 max(sigt)]); hold on;
        plot(e,sigt2,'r--'); legend('Asymmetric Curve','Symmetric Curve'); title(strcat('GJR-t NIC curve for e(t-1),', stocks(i)));
    end

    % Create Plots for conditional standard deviations for different models
    % in different series.
    figure; plot(Date(2:s), sigGt(:, i+1), 'y', 'LineWidth', 2); hold on;
    plot(Date(2:s), sigAGt(:, i+1), 'b', 'LineWidth', 2); hold on;
    plot(Date(2:s), sigGJRt(:,i+1), 'r', 'LineWidth', 2);
    title(strcat('Conditional Standard Deviations for GARCH-t, AR-GARCH-t and GJR-t,', stocks(i))); 
    datetick('x', 'mmm/yy'); xlim([min(Date(2:s)) max(Date(2:s))]); 
    legend('GARCH-t', 'AR-GARCH-t', 'GJR-GARCH-t', 'location', 'northwest');
end

% Decorate the table summarizing fit details
garch_fit.Properties.VariableNames = {'pVal_LBtest_Err_5','pVal_LBtest_Err_10','pVal_LBtest_SqarErr_5','pVal_LBtest_SqarErr_10',...
    'pVal_JBtest_Err','Skewness_Err','Kurtosis_Err'};
garch_fit.Properties.RowNames = stocks  

argarch_fit.Properties.VariableNames = {'pVal_LBtest_Err_5','pVal_LBtest_Err_10','pVal_LBtest_SqarErr_5','pVal_LBtest_SqarErr_10',...
    'pVal_JBtest_Err','Skewness_Err','Kurtosis_Err'};
argarch_fit.Properties.RowNames = stocks

gjrgarch_fit.Properties.VariableNames = {'pVal_LBtest_Err_5','pVal_LBtest_Err_10','pVal_LBtest_SqarErr_5','pVal_LBtest_SqarErr_10',...
    'pVal_JBtest_Err','Skewness_Err','Kurtosis_Err'};
gjrgarch_fit.Properties.RowNames = stocks
    
% Delete the first column in each conditional variance for later plotting 
sigGt(:,1) = []; sigAGt(:,1) = []; sigGJRt(:,1) = []; 


%%
%====================================================
% +++ Part Four (i) voltility forecast measurement +++
%====================================================

% Generate forecasts for returns and volatility
% Initialise vectors for keeping return and volatility forecasts
sig_bx = 0; sig_ctb = 0; sig_psx = 0; sig_jnj = 0; sig_lbyta = 0; 
ret_bx = 0; ret_ctb = 0; ret_psx = 0; ret_jnj = 0; ret_lbyta = 0;
upper_bx = 0; upper_ctb = 0; upper_psx = 0; upper_jnj = 0; upper_lbyta = 0;
lower_bx = 0; lower_ctb = 0; lower_psx = 0; lower_jnj = 0; lower_lbyta = 0;

% Initialise vectors for storing VaR forecasts
VaR1_bx = 0; VaR1_ctb = 0; VaR1_psx = 0; VaR1_jnj = 0; VaR1_lbyta = 0;
VaR5_bx = 0; VaR5_ctb = 0; VaR5_psx = 0; VaR5_jnj = 0; VaR5_lbyta = 0;

% Specify models for each series 
mdlGT = garch(1,1); mdlGT.Distribution = 't'; mdlGT.Offset = NaN;
mdlAGT_bx = arima('ARLags', 1, 'Variance', garch(1,1), 'Distribution', 'T');
mdlAGT_ctb = arima('ARLags', 1, 'Variance', garch(5,1), 'Distribution', 'T');
mdlAGT_psx = arima('ARLags', 1, 'Variance', garch(4,2), 'Distribution', 'T');
mdlAGT_jnj = arima('ARLags', 1, 'Variance', garch(1,1), 'Distribution', 'T');
mdlAGT_lbyta = arima('ARLags', 1, 'Variance', garch(1,3), 'Distribution', 'T');
mdlGJRt = gjr(1,1);  mdlGJRt.Distribution = 't'; mdlGJRt.Offset = NaN;

for t=1:length(f_ret)   
    % Create the training set to fit models
    series_fit = full_ret(t:t+length(is_ret)-1, :);
       
    % Fit or Re-fit the model at specified period
    if mod(t, 5) == 0 || t == 1
        % 1. BX series
        % GARCH(1,1) model
        [EstMdl_gt_bx, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,1), 'display', 'off');
        v_gt_bx = infer(EstMdl_gt_bx, series_fit(:,1)); 
        sd_gt_bx = sqrt(v_gt_bx); dfGt_bx = EstMdl_gt_bx.Distribution.DoF;
        % AR(1)-GARCH(1,1) model
        [EstMdl_agt_bx, EstParamCov, LLF, info] = estimate(mdlAGT_bx, series_fit(:,1), 'display', 'off');
        [e_agt_bx, v_agt_bx, logL] = infer(EstMdl_agt_bx, series_fit(:,1));
        sd_agt_bx = sqrt(v_agt_bx); dfAGt_bx = EstMdl_agt_bx.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_bx, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,1), 'display', 'off');
        v_gjrt_bx = infer(EstMdl_gjrt_bx, series_fit(:,1)); 
        sd_gjrt_bx = sqrt(v_gjrt_bx); dfGJRt_bx = EstMdl_gjrt_bx.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_bx = EstMdl_gt_bx.Offset; p0GJRt_bx = EstMdl_gjrt_bx.Offset;
        
        % 2. CTB series
        % GARCH(1,1) model
        [EstMdl_gt_ctb, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,2), 'display', 'off');
        v_gt_ctb = infer(EstMdl_gt_ctb, series_fit(:,2)); 
        sd_gt_ctb = sqrt(v_gt_ctb); dfGt_ctb = EstMdl_gt_ctb.Distribution.DoF;
        % AR(1)-GARCH(5,1) model
        [EstMdl_agt_ctb, EstParamCov, LLF, info] = estimate(mdlAGT_ctb, series_fit(:,2), 'display', 'off');
        [e_agt_ctb, v_agt_ctb, logL] = infer(EstMdl_agt_ctb, series_fit(:,2));
        sd_agt_ctb = sqrt(v_agt_ctb); dfAGt_ctb = EstMdl_agt_ctb.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_ctb, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,2), 'display', 'off');
        v_gjrt_ctb = infer(EstMdl_gjrt_ctb, series_fit(:,2)); 
        sd_gjrt_ctb = sqrt(v_gjrt_ctb); dfGJRt_ctb = EstMdl_gjrt_ctb.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_ctb = EstMdl_gt_ctb.Offset; p0GJRt_ctb = EstMdl_gjrt_ctb.Offset;
               
        % 3. PSX series
        % GARCH(1,1) model
        [EstMdl_gt_psx, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,3), 'display', 'off');
        v_gt_psx = infer(EstMdl_gt_psx, series_fit(:,3));
        sd_gt_psx = sqrt(v_gt_psx); dfGt_psx = EstMdl_gt_psx.Distribution.DoF;
        % AR(1)-GARCH(4,2) model
        [EstMdl_agt_psx, EstParamCov, LLF, info] = estimate(mdlAGT_psx, series_fit(:,3), 'display', 'off');
        [e_agt_psx, v_agt_psx, logL] = infer(EstMdl_agt_psx, series_fit(:,3));
        sd_agt_psx = sqrt(v_agt_psx);  dfAGt_psx = EstMdl_agt_psx.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_psx, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,3), 'display', 'off');
        v_gjrt_psx = infer(EstMdl_gjrt_psx, series_fit(:,3));
        sd_gjrt_psx = sqrt(v_gjrt_psx); dfGJRt_psx = EstMdl_gjrt_psx.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_psx = EstMdl_gt_psx.Offset; p0GJRt_psx = EstMdl_gjrt_psx.Offset;

        % 4. JNJ series
        % GARCH(1,1) model
        [EstMdl_gt_jnj, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,4), 'display', 'off');
        v_gt_jnj = infer(EstMdl_gt_jnj, series_fit(:,4));
        sd_gt_jnj = sqrt(v_gt_jnj); dfGt_jnj = EstMdl_gt_jnj.Distribution.DoF;
        % AR(1)-GARCH(1,1) model
        [EstMdl_agt_jnj, EstParamCov, LLF, info] = estimate(mdlAGT_jnj, series_fit(:,4), 'display', 'off');
        [e_agt_jnj, v_agt_jnj, logL] = infer(EstMdl_agt_jnj, series_fit(:,4));
        sd_agt_jnj = sqrt(v_agt_jnj);  dfAGt_jnj = EstMdl_agt_jnj.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_jnj, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,4), 'display', 'off');
        v_gjrt_jnj = infer(EstMdl_gjrt_jnj, series_fit(:,4)); 
        sd_gjrt_jnj = sqrt(v_gjrt_jnj); dfGJRt_jnj = EstMdl_gjrt_jnj.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_jnj = EstMdl_gt_jnj.Offset; p0GJRt_jnj = EstMdl_gjrt_jnj.Offset;
        
        % 5. LBYTA series
        % GARCH(1,1) model
        [EstMdl_gt_lbyta, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,5), 'display', 'off');
        v_gt_lbyta = infer(EstMdl_gt_lbyta, series_fit(:,5));
        sd_gt_lbyta = sqrt(v_gt_lbyta); dfGt_lbyta = EstMdl_gt_lbyta.Distribution.DoF;
        % AR(1)-GARCH(1,1) model
        [EstMdl_agt_lbyta, EstParamCov, LLF, info] = estimate(mdlAGT_lbyta, series_fit(:,5), 'display', 'off');
        [e_agt_lbyta, v_agt_lbyta, logL] = infer(EstMdl_agt_lbyta, series_fit(:,5));
        sd_agt_lbyta = sqrt(v_agt_lbyta); dfAGt_lbyta = EstMdl_agt_lbyta.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_lbyta, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,5), 'display', 'off');
        v_gjrt_lbyta = infer(EstMdl_gjrt_lbyta, series_fit(:,5)); 
        sd_gjrt_lbyta = sqrt(v_gjrt_lbyta); dfGJRt_lbyta = EstMdl_gjrt_lbyta.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_lbyta = EstMdl_gt_lbyta.Offset; p0GJRt_lbyta = EstMdl_gjrt_lbyta.Offset;
    end  
        
    % Generate volatility and return forecasts for BX
    sig_bx(t, 1) = sqrt(forecast(EstMdl_gt_bx, 1, 'Y0', series_fit(:,1)));
    [ret_bx(t, 1), YMSE_bx , v_bx] = forecast(EstMdl_agt_bx, 1, 'Y0', series_fit(:,1));
    upper_bx(t, 1) = ret_bx(t, 1) + tinv(0.95, dfAGt_bx)*sqrt(YMSE_bx); 
    lower_bx(t, 1) = ret_bx(t, 1) - tinv(0.95, dfAGt_bx)*sqrt(YMSE_bx);
    sig_bx(t, 2) = sqrt(v_bx); 
    sig_bx(t, 3) = sqrt(forecast(EstMdl_gjrt_bx, 1, 'Y0', series_fit(:,1)));
    % Generate VaR forecasts for BX
    VaR1_bx(t, 1) = p0Gt_bx+tinv(0.01, dfGt_bx)*sig_bx(t, 1)*sqrt((dfGt_bx-2)/dfGt_bx);
    VaR1_bx(t, 2) = ret_bx(t, 1)+tinv(0.01, dfAGt_bx)*sig_bx(t, 2)*sqrt((dfAGt_bx-2)/dfAGt_bx);
    VaR1_bx(t, 3) = p0GJRt_bx+tinv(0.01, dfGJRt_bx)*sig_bx(t, 3)*sqrt((dfGJRt_bx-2)/dfGJRt_bx);
    VaR5_bx(t, 1) = p0Gt_bx+tinv(0.05, dfGt_bx)*sig_bx(t, 1)*sqrt((dfGt_bx-2)/dfGt_bx);
    VaR5_bx(t, 2) = ret_bx(t, 1)+tinv(0.05, dfAGt_bx)*sig_bx(t, 2)*sqrt((dfAGt_bx-2)/dfAGt_bx);
    VaR5_bx(t, 3) = p0GJRt_bx+tinv(0.05, dfGJRt_bx)*sig_bx(t, 3)*sqrt((dfGJRt_bx-2)/dfGJRt_bx);
   
    % Generate volatility and return forecasts for CTB
    sig_ctb(t, 1) = sqrt(forecast(EstMdl_gt_ctb, 1, 'Y0', series_fit(:,2)));
    [ret_ctb(t, 1), YMSE_ctb, v_ctb] = forecast(EstMdl_agt_ctb, 1, 'Y0', series_fit(:,2));
    upper_ctb(t, 1) = ret_ctb(t, 1) + tinv(0.95, dfAGt_ctb)*sqrt(YMSE_ctb); 
    lower_ctb(t, 1) = ret_ctb(t, 1) - tinv(0.95, dfAGt_ctb)*sqrt(YMSE_ctb);
    sig_ctb(t, 2) = sqrt(v_ctb); 
    sig_ctb(t, 3) = sqrt(forecast(EstMdl_gjrt_ctb, 1, 'Y0', series_fit(:,2)));
    % Generate VaR forecasts for CTB
    VaR1_ctb(t, 1) = p0Gt_ctb+tinv(0.01, dfGt_ctb)*sig_ctb(t, 1)*sqrt((dfGt_ctb-2)/dfGt_ctb);
    VaR1_ctb(t, 2) = ret_ctb(t, 1)+tinv(0.01, dfAGt_ctb)*sig_ctb(t, 2)*sqrt((dfAGt_ctb-2)/dfAGt_ctb);
    VaR1_ctb(t, 3) = p0GJRt_ctb+tinv(0.01, dfGJRt_ctb)*sig_ctb(t, 3)*sqrt((dfGJRt_ctb-2)/dfGJRt_ctb);
    VaR5_ctb(t, 1) = p0Gt_ctb+tinv(0.05, dfGt_ctb)*sig_ctb(t, 1)*sqrt((dfGt_ctb-2)/dfGt_ctb);
    VaR5_ctb(t, 2) = ret_ctb(t, 1)+tinv(0.05, dfAGt_ctb)*sig_ctb(t, 2)*sqrt((dfAGt_ctb-2)/dfAGt_ctb);
    VaR5_ctb(t, 3) = p0GJRt_ctb+tinv(0.05, dfGJRt_ctb)*sig_ctb(t, 3)*sqrt((dfGJRt_ctb-2)/dfGJRt_ctb);
    
    % Generate volatility and return forecasts for PSX
    sig_psx(t, 1) = sqrt(forecast(EstMdl_gt_psx, 1, 'Y0', series_fit(:,3)));
    [ret_psx(t, 1), YMSE_psx, v_psx] = forecast(EstMdl_agt_psx, 1, 'Y0', series_fit(:,3));
    upper_psx(t, 1) = ret_psx(t, 1) + tinv(0.95, dfAGt_psx)*sqrt(YMSE_psx);
    lower_psx(t, 1) = ret_psx(t, 1) - tinv(0.95, dfAGt_psx)*sqrt(YMSE_psx);
    sig_psx(t, 2) = sqrt(v_psx); 
    sig_psx(t, 3) = sqrt(forecast(EstMdl_gjrt_psx, 1, 'Y0', series_fit(:,3)));
    % Generate VaR forecasts for PSX
    VaR1_psx(t, 1) = p0Gt_psx+tinv(0.01, dfGt_psx)*sig_psx(t, 1)*sqrt((dfGt_psx-2)/dfGt_psx);
    VaR1_psx(t, 2) = ret_psx(t, 1)+tinv(0.01, dfAGt_psx)*sig_psx(t, 2)*sqrt((dfAGt_psx-2)/dfAGt_psx);
    VaR1_psx(t, 3) = p0GJRt_psx+tinv(0.01, dfGJRt_psx)*sig_psx(t, 3)*sqrt((dfGJRt_psx-2)/dfGJRt_psx);
    VaR5_psx(t, 1) = p0Gt_psx+tinv(0.05, dfGt_psx)*sig_psx(t, 1)*sqrt((dfGt_psx-2)/dfGt_psx);
    VaR5_psx(t, 2) = ret_psx(t, 1)+tinv(0.05, dfAGt_psx)*sig_psx(t, 2)*sqrt((dfAGt_psx-2)/dfAGt_psx);
    VaR5_psx(t, 3) = p0GJRt_psx+tinv(0.05, dfGJRt_psx)*sig_psx(t, 3)*sqrt((dfGJRt_psx-2)/dfGJRt_psx);
    
    % Generate volatility and return forecasts for JNJ
    sig_jnj(t, 1) = sqrt(forecast(EstMdl_gt_jnj, 1, 'Y0', series_fit(:,4)));
    [ret_jnj(t, 1), YMSE_jnj, v_jnj] = forecast(EstMdl_agt_jnj, 1, 'Y0', series_fit(:,4));
    upper_jnj(t, 1) = ret_jnj(t, 1) + tinv(0.95, dfAGt_jnj)*sqrt(YMSE_jnj); 
    lower_jnj(t, 1) = ret_jnj(t, 1) - tinv(0.95, dfAGt_jnj)*sqrt(YMSE_jnj);
    sig_jnj(t, 2) = sqrt(v_jnj); 
    sig_jnj(t, 3) = sqrt(forecast(EstMdl_gjrt_jnj, 1, 'Y0', series_fit(:,4)));
    % Generate VaR forecasts for JNJ
    VaR1_jnj(t, 1) = p0Gt_jnj+tinv(0.01, dfGt_jnj)*sig_jnj(t, 1)*sqrt((dfGt_jnj-2)/dfGt_jnj);
    VaR1_jnj(t, 2) = ret_jnj(t, 1)+tinv(0.01, dfAGt_jnj)*sig_jnj(t, 2)*sqrt((dfAGt_jnj-2)/dfAGt_jnj);
    VaR1_jnj(t, 3) = p0GJRt_jnj+tinv(0.01, dfGJRt_jnj)*sig_jnj(t, 3)*sqrt((dfGJRt_jnj-2)/dfGJRt_jnj);
    VaR5_jnj(t, 1) = p0Gt_jnj+tinv(0.05, dfGt_jnj)*sig_jnj(t, 1)*sqrt((dfGt_jnj-2)/dfGt_jnj);
    VaR5_jnj(t, 2) = ret_jnj(t, 1)+tinv(0.05, dfAGt_jnj)*sig_jnj(t, 2)*sqrt((dfAGt_jnj-2)/dfAGt_jnj);
    VaR5_jnj(t, 3) = p0GJRt_jnj+tinv(0.05, dfGJRt_jnj)*sig_jnj(t, 3)*sqrt((dfGJRt_jnj-2)/dfGJRt_jnj);
    
    % Generate volatility and return forecasts for LBYTA
    sig_lbyta(t, 1) = sqrt(forecast(EstMdl_gt_lbyta, 1, 'Y0', series_fit(:,5)));
    [ret_lbyta(t, 1), YMSE_lbyta, v_lbyta] = forecast(EstMdl_agt_lbyta, 1, 'Y0', series_fit(:,5));
    upper_lbyta(t, 1) = ret_lbyta(t, 1) + tinv(0.95, dfAGt_lbyta)*sqrt(YMSE_lbyta); 
    lower_lbyta(t, 1) = ret_lbyta(t, 1) - tinv(0.95, dfAGt_lbyta)*sqrt(YMSE_lbyta);
    sig_lbyta(t, 2) = sqrt(v_lbyta); 
    sig_lbyta(t, 3) = sqrt(forecast(EstMdl_gjrt_lbyta, 1, 'Y0', series_fit(:,5)));  
    % Generate VaR forecasts for LBYTA
    VaR1_lbyta(t, 1) = p0Gt_lbyta+tinv(0.01, dfGt_lbyta)*sig_lbyta(t, 1)*sqrt((dfGt_lbyta-2)/dfGt_lbyta);
    VaR1_lbyta(t, 2) = ret_lbyta(t, 1)+tinv(0.01, dfAGt_lbyta)*sig_lbyta(t, 2)*sqrt((dfAGt_lbyta-2)/dfAGt_lbyta);
    VaR1_lbyta(t, 3) = p0GJRt_lbyta+tinv(0.01, dfGJRt_lbyta)*sig_lbyta(t, 3)*sqrt((dfGJRt_lbyta-2)/dfGJRt_lbyta);
    VaR5_lbyta(t, 1) = p0Gt_lbyta+tinv(0.05, dfGt_lbyta)*sig_lbyta(t, 1)*sqrt((dfGt_lbyta-2)/dfGt_lbyta);
    VaR5_lbyta(t, 2) = ret_lbyta(t, 1)+tinv(0.05, dfAGt_lbyta)*sig_lbyta(t, 2)*sqrt((dfAGt_lbyta-2)/dfAGt_lbyta);
    VaR5_lbyta(t, 3) = p0GJRt_lbyta+tinv(0.05, dfGJRt_lbyta)*sig_lbyta(t, 3)*sqrt((dfGJRt_lbyta-2)/dfGJRt_lbyta);
end

% To plot the forecasting conditional variance with is-sample conditional
% variance for each asset
ret_Df_AGt = [ret_bx ret_ctb ret_psx ret_jnj ret_lbyta];
upper_f_AGt = [upper_bx upper_ctb upper_psx upper_jnj upper_lbyta];
lower_f_AGt = [lower_bx lower_ctb lower_psx lower_jnj lower_lbyta];
colors = {'y', 'r', 'b'};
for i=1:5
    
    figure; subplot(2,1,1);
    plot(Date(2:s), sigGt(:, i), 'y'); hold on
    plot(Date(2:s), sigAGt(:, i), 'r'); hold on
    plot(Date(2:s), sigGJRt(:, i), 'b'); hold on
    if i==1
        for c = 1:3
        plot(Date(s+1:end), sig_bx(:, c), 'color', colors{c}); hold on
        end
    elseif i == 2
        for c = 1:3
        plot(Date(s+1:end), sig_ctb(:, c), 'color', colors{c}); hold on
        end
    elseif i == 3
        for c = 1:3
        plot(Date(s+1:end), sig_psx(:, c), 'color', colors{c}); hold on
        end
    elseif i == 4
        for c = 1:3
        plot(Date(s+1:end), sig_jnj(:, c), 'color', colors{c}); hold on
        end
    else
        for c = 1:3
        plot(Date(s+1:end), sig_lbyta(:, c), 'color', colors{c}); hold on
        end
    end
    plot([Date(s+1), Date(s+1)], [0, max(sigGJRt(:, i))*(1+0.5)], 'k--', 'LineWidth',1); % split the plot into in-sample part and forecasting part
    hold off
    title(strcat('Forecasting Conditional Standard Deviations for GARCH-t AR-GARCH-t and GJR-t,', stocks(i))); 
    datetick('x', 'mmm/yy'); xlim([min(Date(1:end)) max(Date(1:end))]); 
    legend('GARCH-t', 'AR-GARCH-t', 'GJR-GARCH-t', 'location', 'northwest', 'orientation', 'horizontal');
    
    subplot(2,1,2); 
    plot(Date(2:s), is_ret(:,i), 'color', [.75,.75,.75]); hold on
    plot(Date(s+1:end), ret_Df_AGt(:, i), 'r'); hold on
    plot(Date(s+1:end), [upper_f_AGt(:, i), lower_f_AGt(:, i)],'k','LineWidth', 0.5);
    hold off
    title(strcat('Forecasting Returns under the AR-GARCH model,', stocks(i))); 
    datetick('x', 'mmm/yy'); xlim([min(Date(1:end)) max(Date(1:end))]); 
    legend('In-sample Return', 'Forecasting Returns', '95% CI', 'location', 'south', 'orientation', 'horizontal');
    
    % To further plot the forecasting returns with forecasting observations
    figure; plot(Date(s+1:end), f_ret(:, i), 'color', [.75,.75,.75], 'LineWidth', 2); hold on
    plot(Date(s+1:end), ret_Df_AGt(:, 2), 'r', 'LineWidth', 2); hold on
    plot(Date(s+1:end), [upper_f_AGt(:, i), lower_f_AGt(:, i)],'k', 'LineWidth', 2.5); hold on
    datetick('x', 'mmm/yy'); xlim([min(Date(s+1:end)) max(Date(s+1:end))]); 
    title(strcat('Forecasting Returns under the AR-GARCH model,', stocks(i))); 
    legend('Observed Returns', 'Predictive Returns', '95% CI', 'location', 'south', 'orientation', 'horizontal');
end    

% Assess each model's volatility forecast accuracy using proxies 1 to 4
% Combine all volatility forecasts under the same model
sig_Df_Gt = [sig_bx(:, 1) sig_ctb(:, 1) sig_psx(:, 1) sig_jnj(:, 1) sig_lbyta(:, 1)];
sig_Df_AGt = [sig_bx(:, 2) sig_ctb(:, 2) sig_psx(:, 2) sig_jnj(:, 2) sig_lbyta(:, 2)];
sig_Df_GJRt = [sig_bx(:, 3) sig_ctb(:, 3) sig_psx(:, 3) sig_jnj(:, 3) sig_lbyta(:, 3)];

sig_f_ACt = sqrt(sigma); clear sigma;
load OpenHighLowPrice.mat; 

for i=1:5
    % Construct the proxy 1,2,3 & 4 for each series 
    prox1C = abs(full_ret(:,i)-mean(full_ret(:,i)));
    prox1Ci = prox1C(1:length(is_ret)); prox1Cf = prox1C(end-length(f_ret)+1:end);
    rangP=log(HighP(:,i)./LowP(:,i)); 
    rangP(rangP <= 0) = mean([0 min(rangP(rangP>0))]);
    prox2C = sqrt(0.3607*(rangP.^2)); prox2C = prox2C(2:end);
    prox2Cf = prox2C(end-length(f_ret)+1:end);
    prox3C = 1.107*prox2C.^2 + 0.68*((log(OpenP(2:end, i)./ p_mat(1:end-1, i))).^2);
    prox3C = sqrt(prox3C); prox3Cf = prox3C(end-length(f_ret)+1:end);
    prox4C = exp(2*log(rangP)-0.86+2*0.29^2);
    prox4C=sqrt(prox4C(2:end)); prox4Cf = prox4C(end-length(f_ret)+1:end);

    % Plot all proxies against each other
    proxCf = [prox1Cf prox2Cf prox3Cf prox4Cf];
    figure; colors = {'bd', 'g+', 'mo', 'y^'};
    for p = 1:4
        plot(Date(s+1:end), proxCf(:,p), colors{p}); hold on;
    end
    title(strcat('All Four Proxies,', stocks(i)));
    legend('Proxy 1','Proxy 2','Proxy 3','Proxy 4', 'location', 'north', 'orientation', 'horizontal');
    datetick('x', 'mmm/yy'); xlim([min(Date(s+1:end)) max(Date(s+1:end))]); 
    
    % Calculate RMSEs for proxy 1,2,3,4 over forecast-sample for all model forecasts
    for p = 1:4
        rmse_ACt_v(:,p) = sqrt(mean((sig_f_ACt(:,i) - proxCf(:,p)).^2));
        rmse_Gt_v(:,p) = sqrt(mean((sig_Df_Gt(:,i) - proxCf(:,p)).^2));
        rmse_AGt_v(:,p) = sqrt(mean((sig_Df_AGt(:,i) - proxCf(:,p)).^2));
        rmse_GJRt_v(:,p) = sqrt(mean((sig_Df_GJRt(:,i) - proxCf(:,p)).^2));
    end
    % Calculate RMSEs for forecasting returns under the AR-GARCH model
    rmse_AGt_r = sqrt(mean((ret_Df_AGt(:,i) - f_ret(:,i)).^2));

    % Calculate MADs for proxy 1 over forecast-sample for all models
    for p = 1:4
        mad_AC_v(:,p) = mean(abs(sqrt(sig_f_ACt(:,i))-proxCf(:,p)));
        mad_Gt_v(:,p) = mean(abs(sqrt(sig_Df_Gt(:,i))-proxCf(:,p)));
        mad_AGt_v(:,p) = mean(abs(sqrt(sig_Df_AGt(:,i))-proxCf(:,p)));
        mad_GJRt_v(:,p) = mean(abs(sqrt(sig_Df_GJRt(:,i))-proxCf(:,p)));
    end
    % Calculate MADs for forecasting returns under the AR-GARCH model
    mad_AGt_r = mean(abs(sqrt(ret_Df_AGt(:,i))-f_ret(:,i)));
    
    % Create summary table
    % Proxy 1
    prox1_sum(i, :) = table([rmse_ACt_v(1) mad_AC_v(1)], [rmse_Gt_v(1) mad_Gt_v(1)], ...
        [rmse_AGt_v(1) mad_AGt_v(1)], [rmse_GJRt_v(1) mad_GJRt_v(1)]);
    % Proxy 2
    prox2_sum(i, :) = table([rmse_ACt_v(2) mad_AC_v(2)], [rmse_Gt_v(2) mad_Gt_v(2)], ...
        [rmse_AGt_v(2) mad_AGt_v(2)], [rmse_GJRt_v(2) mad_GJRt_v(2)]);
    % Proxy 3
    prox3_sum(i, :) = table([rmse_ACt_v(3) mad_AC_v(3)], [rmse_Gt_v(3) mad_Gt_v(3)], ...
        [rmse_AGt_v(3) mad_AGt_v(3)], [rmse_GJRt_v(3) mad_GJRt_v(3)]);
    % Proxy 4
    prox4_sum(i, :) = table([rmse_ACt_v(4) mad_AC_v(4)], [rmse_Gt_v(4) mad_Gt_v(4)], ...
        [rmse_AGt_v(4) mad_AGt_v(4)], [rmse_GJRt_v(4) mad_GJRt_v(4)]);
    ret_sum(i, :) = table(rmse_AGt_r, mad_AGt_r);
    
    % Plot the volatility forecasts against the proxy 1,2,3,4
    figure; 
    for p = 1:4
        plot(Date(s+1:end), proxCf(:,p), colors{p}); hold on;
    end
    plot(Date(s+1:end), sig_f_ACt(:, i), 'y', 'LineWidth', 1.5); hold on;
    plot(Date(s+1:end), sig_Df_Gt(:, i), 'k', 'LineWidth', 1.5); hold on;
    plot(Date(s+1:end), sig_Df_AGt(:, i), 'r', 'LineWidth', 1.5); hold on;
    plot(Date(s+1:end), sig_Df_GJRt(:, i), 'b', 'LineWidth', 1.5); hold on;
    title(strcat('Proxy 1,2,3,4 with Model Forecasts,', stocks(i)));
    legend('Proxy 1','Proxy 2','Proxy 3','Proxy 4','ARCH-t','GARCH-t','AR-GARCH-t','GJR-GARCH-t', ...
        'location', 'north', 'orientation', 'horizontal');
    datetick('x', 'mmm/yy'); xlim([min(Date(s+1:end)) max(Date(s+1:end))]); 
end

prox1_sum.Properties.VariableNames = {'Proxy1_ARCH_t_RMSE_MAD', 'Proxy1_GARCH_t_RMSE_MAD','Proxy1_AR_GARCH_t_RMSE_MAD','Proxy1_GJR_GARCH_t_RMSE_MAD'};
prox1_sum.Properties.RowNames = stocks; 
prox2_sum.Properties.VariableNames = {'Proxy2_ARCH_t_RMSE_MAD', 'Proxy2_GARCH_t_RMSE_MAD','Proxy2_AR_GARCH_t_RMSE_MAD','Proxy2_GJR_GARCH_t_RMSE_MAD'};
prox2_sum.Properties.RowNames = stocks 
prox3_sum.Properties.VariableNames = {'Proxy3_ARCH_t_RMSE_MAD', 'Proxy3_GARCH_t_RMSE_MAD','Proxy3_AR_GARCH_t_RMSE_MAD','Proxy3_GJR_GARCH_t_RMSE_MAD'};
prox3_sum.Properties.RowNames = stocks 
prox4_sum.Properties.VariableNames = {'Proxy4_ARCH_t_RMSE_MAD', 'Proxy4_GARCH_t_RMSE_MAD','Proxy4_AR_GARCH_t_RMSE_MAD','Proxy4_GJR_GARCH_t_RMSE_MAD'};
prox4_sum.Properties.RowNames = stocks 
ret_sum.Properties.VariableNames = {'AR_GARCH_t_RMSE','AR_GARCH_t_MAD'};
ret_sum.Properties.RowNames = stocks 

% Combine all daily VaR forecasts under the same model for later portfolio
% construction
VaR1_Df_Gt = [VaR1_bx(:,1) VaR1_ctb(:,1) VaR1_psx(:,1) VaR1_jnj(:,1) VaR1_lbyta(:,1)];
VaR1_Df_AGt = [VaR1_bx(:,2) VaR1_ctb(:,2) VaR1_psx(:,2) VaR1_jnj(:,2) VaR1_lbyta(:,2)];
VaR1_Df_GJRt = [VaR1_bx(:,3) VaR1_ctb(:,3) VaR1_psx(:,3) VaR1_jnj(:,3) VaR1_lbyta(:,3)];
VaR5_Df_Gt = [VaR5_bx(:,1) VaR5_ctb(:,1) VaR5_psx(:,1) VaR5_jnj(:,1) VaR5_lbyta(:,1)];
VaR5_Df_AGt = [VaR5_bx(:,2) VaR5_ctb(:,2) VaR5_psx(:,2) VaR5_jnj(:,2) VaR5_lbyta(:,2)];
VaR5_Df_GJRt = [VaR5_bx(:,3) VaR5_ctb(:,3) VaR5_psx(:,3) VaR5_jnj(:,3) VaR5_lbyta(:,3)];



%%
%====================================================
% +++ Part Four (ii) returns forecast measurement +++
%====================================================
% we have already record ad-hoc, ARMA returns
% Now record ar_garch returns
rmse_ar_garch = zeros(5,1);
mad_ar_garch  = zeros(5,1);
for i=1:5
    [rmse_ar_garch(i,1),mad_ar_garch(i,1)]= getfa(ret_Df_AGt(:,i),f_ret(:,i));
end
% combined model for returns
combine_retusns  = (ad_hoc_returns+arma_returns+ret_Df_AGt)/3;
%plot 
rmse_comb = zeros(5,1);
mad_comb  = zeros(5,1);
for i=1:5
    [rmse_comb(i,1),mad_comb(i,1)]=getfa(combine_retusns(:,i),f_ret(:,i));
end

%create the table
return_rmse = array2table([rmse_adhoc,rmse_arma ...,
    ,rmse_ar_garch, rmse_comb]);
return_rmse.Properties.VariableNames = {'AdhocRMSE'  'ARMARMSE' ...,
     'ARGARCHRMSE' 'ModelCombinationRMSE'};
return_rmse.Properties.RowNames = {'BX' 'CTB' 'PSX' 'JNJ' 'LBTYA' }
%
return_mae = array2table([mad_adhoc, mad_arma, mad_ar_garch, mad_comb]);
return_mae.Properties.VariableNames = {'AdhocMAE' ...,
    'ARMAMAE' 'ARGARCHMAE' 'ModelCombinationMAE'};
return_mae.Properties.RowNames = {'BX' 'CTB' 'PSX' 'JNJ' 'LBTYA' }


% PLOT the results 
 ...,
figure;plot(Date(s+1:end),ad_hoc_returns(:,1),'g');hold on;
plot(Date(s+1:end),arma_returns(:,1),'r');
plot(Date(s+1:end),ret_Df_AGt(:,1),'b');
plot(Date(s+1:end),combine_retusns(:,1),'y');
plot(Date(s+1:end),f_ret(:,1),'Color',[.75,.75,.75]);
title('BX stock returns predictions');
legend('Adhoc','ARMA','ARGARCH','ModelComb','Obsearved');
%======================
figure;plot(Date(s+1:end),ad_hoc_returns(:,2),'g');hold on;
plot(Date(s+1:end),arma_returns(:,2),'r');
plot(Date(s+1:end),ret_Df_AGt(:,2),'b');
plot(Date(s+1:end),combine_retusns(:,2),'y');
plot(Date(s+1:end),f_ret(:,2),'Color',[.75,.75,.75]);
title('CTB stock returns predictions');
legend('Adhoc','ARMA','ARGARCH','ModelComb','Obsearved');
%=====================
figure;plot(Date(s+1:end),ad_hoc_returns(:,3),'g');hold on;
plot(Date(s+1:end),arma_returns(:,3),'r');
plot(Date(s+1:end),ret_Df_AGt(:,3),'b');
plot(Date(s+1:end),combine_retusns(:,3),'y');
plot(Date(s+1:end),f_ret(:,3),'Color',[.75,.75,.75]);
title( 'PSX stock returns predictions');
legend('Adhoc','ARMA','ARGARCH','ModelComb','Obsearved');

%=====================
figure;plot(Date(s+1:end),ad_hoc_returns(:,4),'g');hold on;
plot(Date(s+1:end),arma_returns(:,4),'r');
plot(Date(s+1:end),ret_Df_AGt(:,4),'b');
plot(Date(s+1:end),combine_retusns(:,4),'y');
plot(Date(s+1:end),f_ret(:,4),'Color',[.75,.75,.75]);
title( 'JNJ stock returns predictions');
legend('Adhoc','ARMA','ARGARCH','ModelComb','Obsearved');
%=======================
figure;plot(Date(s+1:end),ad_hoc_returns(:,5),'g');hold on;
plot(Date(s+1:end),arma_returns(:,5),'r');
plot(Date(s+1:end),ret_Df_AGt(:,5),'b');
plot(Date(s+1:end),combine_retusns(:,5),'y');
plot(Date(s+1:end),f_ret(:,5),'Color',[.75,.75,.75]);
title( 'LBTYA stock returns predictions');
legend('Adhoc','ARMA','ARGARCH','ModelComb','Obsearved');

%%
%===========================================
% +++ Part Five: portfolio construction+++
%===========================================

%1. equal weights portfolios

returns_benchmark = mean(f_ret,2);

acc_benchmark     = exp(cumsum(returns_benchmark));

figure;plot(Date(s+1:end),acc_benchmark);title('Accumulate returns of Equal weights portfolio');

mean(returns_benchmark) 
std(returns_benchmark) 
acc_benchmark(1:end)


%% 2.1 maximum returns portfolios (daily)
% use AD-hoc returns (daily)
w_r = zeros(252,5);
for t=1:length(f_ret)
    sum_v = sum(exp(100*ad_hoc_returns(t,:)));
    for i=1:5
        w_r(t,i) = exp(100*ad_hoc_returns(t,i))/sum_v;
    end
end

%cauculate returns
acc_r_adhoc = exp(cumsum(sum(w_r.*f_ret,2)))
figure;plot(Date(s+1:end),acc_benchmark,'Color',[.75,.75,.75]);hold on
plot(Date(s+1:end),acc_r_adhoc);;title('Accumulate returns of maximum Adhoc returns portfolio');
legend('benchmark','maximum Adhoc returns portfolio')

[mean(sum(w_r.*f_ret,2)),std(sum(w_r.*f_ret,2)),acc_r_adhoc(end,:)]

%% use ARMA returns (daily)
w_r = zeros(252,5);
for t=1:length(f_ret)
    sum_v = sum(exp(100*arma_returns(t,:)));
    for i=1:5
        w_r(t,i) = exp(100*arma_returns(t,i))/sum_v;
    end
end

%cauculate returns
acc_r_arma = exp(cumsum(sum(w_r.*f_ret,2)))
figure;plot(Date(s+1:end),acc_benchmark,'Color',[.75,.75,.75]);hold on
plot(Date(s+1:end),acc_r_arma);;title('Accumulate returns of maximum ARMA returns portfolio');
legend('benchmark','maximum ARMA returns portfolio')

[mean(sum(w_r.*f_ret,2)),std(sum(w_r.*f_ret,2)),acc_r_arma(end,:)]
%% use AR-GARCH returns(daily)
w_r = zeros(252,5);
for t=1:length(f_ret)
    sum_v = sum(exp(100*ret_Df_AGt(t,:)));
    for i=1:5
        w_r(t,i) = exp(100*ret_Df_AGt(t,i))/sum_v;
    end
end

ARGARCH_Daily_optimal = w_r;
%cauculate returns
acc_r = exp(cumsum(sum(w_r.*f_ret,2)))

figure;plot(Date(s+1:end),acc_benchmark,'Color',[.75,.75,.75]);hold on
plot(Date(s+1:end),acc_r);;title('Accumulate returns of maximum AR-GARCH returns portfolio');
legend('benchmark','maximum AR-GARCH returns portfolio')


[mean(sum(w_r.*f_ret,2)),std(sum(w_r.*f_ret,2)),acc_r(end,:)]


%% use combinations returns(daily)
w_r = zeros(252,5);
for t=1:length(f_ret)
    sum_v = sum(abs(100*combine_retusns(t,:)));
    for i=1:5
        w_r(t,i) = abs(100*combine_retusns(t,i))/sum_v;
    end
end

%cauculate returns
acc_r_com = exp(cumsum(sum(w_r.*f_ret,2)))

figure;plot(Date(s+1:end),acc_benchmark,'Color',[.75,.75,.75]);hold on

plot(Date(s+1:end),acc_r_adhoc);
plot(Date(s+1:end),acc_r_arma);
plot(Date(s+1:end),acc_r);
plot(Date(s+1:end),acc_r_com);;title('Accumulate returns of maximum returns portfolio');
legend('benchmark','adhoc','arma', 'ar-garch', 'modelcombination')

[mean(sum(w_r.*f_ret,2)),std(sum(w_r.*f_ret,2)),acc_r_com(end,:)]


%% 2.2 maximum returns portfolios (Montly)
% Stragety 2 using montly updates adhoc
ad_hoc_returns_mf = zeros(length(f_ret),5);
for t=1:length(f_ret)
    ret_is = full_ret(t:length(f_ret)+t-1,:);
    if mod(t,21)==0 | t==1
        ad_hoc_returns_mf(t,:) = mean(ret_is(end-21:end,:));
    else
        ad_hoc_returns_mf(t,:) = ad_hoc_returns_mf(t-1,:);
    end 
end


w_r_m     = zeros(252,5);
for t=1:length(f_ret)
    sum_v = sum(abs(100*ad_hoc_returns_mf(t,:)));
    for i=1:5
        w_r_m(t,i) = abs(100*ad_hoc_returns_mf(t,i))/sum_v;
    end
end

acc_r_adhoc_m = exp(cumsum(sum(w_r_m.*f_ret,2)))
figure;plot(Date(s+1:end),acc_benchmark,'Color',[.75,.75,.75]);hold on
plot(Date(s+1:end),acc_r_adhoc_m);;title('Accumulate returns of maximum adhoc returns portfolio');
legend('benchmark','maximum adhoc returns portfolio')

ADHOC_Monthly_optimal = w_r_m;
[mean(sum(w_r_m.*f_ret,2)),std(sum(w_r_m.*f_ret,2)),acc_r_adhoc_m(end,:)]

%% Stragety 2 using montly updates ARMA returns
%get the montly ARMA return
ret_mf                      = zeros(length(f_ret),5);
for i=1:5
    series = is_ret(:,i);
    [p,q,best]=getpq(series);
    %3. estimate models
    Mdl=arima(p,0,q);
    [ret_mf(1,i)] = forecast(EstMdl,1,'Y0',series); % 1-period forecasts
    for t=2:length(f_ret)
        ret_is = full_ret(t:length(series)+t-1,i);
        if mod(t,21)==0  
            Mdl=arima(p,0,q);[EstMdl,EstParamCov,logL,info] = estimate(Mdl,ret_is,'display','Off');
        end
        [ret_mf(t:t+20,i)] = forecast(EstMdl,21,'Y0',ret_is);
    end
end
ret_mf = ret_mf(1:end-20,:);
% fit in the portfolio 
w_r = zeros(252,5);
for t=1:length(f_ret)
    sum_v = sum(exp(100*ret_mf(t,:)));
    for i=1:5
        w_r(t,i) = exp(100*ret_mf(t,i))/sum_v;
    end
end

%plot the results
acc_r_arma_m = exp(cumsum(sum(w_r.*f_ret,2)))
figure;plot(Date(s+1:end),acc_benchmark,'Color',[.75,.75,.75]);hold on
plot(Date(s+1:end),acc_r_arma_m);;title('Accumulate returns of maximum ARMA returns portfolio');
legend('benchmark','maximum ARMA returns portfolio')
[mean(sum(w_r.*f_ret,2)),std(sum(w_r.*f_ret,2)),acc_r_arma_m(end,:)]

%% Stragety 2 using montly AR_GARCH
ret_Mf_AGt = [ret_bx ret_ctb ret_psx ret_jnj ret_lbyta];
w_r = zeros(252,5);
for t=1:length(f_ret)
    sum_v = sum(exp(100*ret_Mf_AGt(t,:)));
    for i=1:5
        w_r(t,i) = exp(100*ret_Mf_AGt(t,i))/sum_v;
    end
end

acc_r_ARGARCH_m = exp(cumsum(sum(w_r.*f_ret,2)))
figure;plot(Date(s+1:end),acc_benchmark,'Color',[.75,.75,.75]);hold on
plot(Date(s+1:end),acc_r_ARGARCH_m);;title('Accumulate returns of maximum ARGARCH returns portfolio');
legend('benchmark','maximum ARGARCH returns portfolio')


[mean(sum(w_r.*f_ret,2)),std(sum(w_r.*f_ret,2)),acc_r_ARGARCH_m(end,:)]

%% Stragety 2 using Montly conbined returns

combine_retusns_mf  = (ad_hoc_returns_mf+ret_mf+ret_Mf_AGt)/3;

w_r = zeros(252,5);
for t=1:length(f_ret)
    sum_v = sum(abs(100*combine_retusns_mf(t,:)));
    for i=1:5
        w_r(t,i) = abs(100*combine_retusns_mf(t,i))/sum_v;
    end
end

%cauculate returns
acc_r_com_mf = exp(cumsum(sum(w_r.*f_ret,2)))


% plot all the results
figure;plot(Date(s+1:end),acc_benchmark,'Color',[.75,.75,.75]);hold on

plot(Date(s+1:end),acc_r_adhoc_m);
plot(Date(s+1:end),acc_r_arma_m);
plot(Date(s+1:end),acc_r_ARGARCH_m);
plot(Date(s+1:end),acc_r_com_mf);;title('Accumulate returns of maximum montly exp returns portfolio');
legend('benchmark','adhoc','arma', 'ar-garch', 'modelcombination')

[mean(sum(w_r.*f_ret,2)),std(sum(w_r.*f_ret,2)),acc_r_com(end,:)]

%% 3 Min VaR portfolios (daily)
colors = {'r', 'b' , 'y', 'g', 'k', 'c'};
figure; plot(Date(s+1:end), f_ret(:, 1), 'color', [.75 .75 .75]);hold on;
for c = 1:3    
    plot(Date(s+1:end), VaR1_bx(:, c), 'color', colors{c}); hold on;   
    plot(Date(s+1:end), VaR5_bx(:, c), 'color', colors{c+3}); hold on;   
end
legend('Log Returns', 'VaR1GARCH-t', 'VaR1AR-GARCH-t' , 'VaR1GJR-GARCH-t', 'VaR5GARCH-t', 'VaR5AR-GARCH-t' , 'VaR5GJR-GARCH-t')


% Generate monthly forecasts for returns and volatility
% Initialise vectors for keeping return and volatility forecasts
sig_bx = 0; sig_ctb = 0; sig_psx = 0; sig_jnj = 0; sig_lbyta = 0; 
ret_bx = 0; ret_ctb = 0; ret_psx = 0; ret_jnj = 0; ret_lbyta = 0;
upper_bx = 0; upper_ctb = 0; upper_psx = 0; upper_jnj = 0; upper_lbyta = 0;
lower_bx = 0; lower_ctb = 0; lower_psx = 0; lower_jnj = 0; lower_lbyta = 0;

% Initialise vectors for storing VaR forecasts
VaR1_bx = 0; VaR1_ctb = 0; VaR1_psx = 0; VaR1_jnj = 0; VaR1_lbyta = 0;
VaR5_bx = 0; VaR5_ctb = 0; VaR5_psx = 0; VaR5_jnj = 0; VaR5_lbyta = 0;

% Specify models for each series 
mdlGT = garch(1,1); mdlGT.Distribution = 't'; mdlGT.Offset = NaN;
mdlAGT_bx = arima('ARLags', 1, 'Variance', garch(1,1), 'Distribution', 'T');
mdlAGT_ctb = arima('ARLags', 1, 'Variance', garch(5,1), 'Distribution', 'T');
mdlAGT_psx = arima('ARLags', 1, 'Variance', garch(4,2), 'Distribution', 'T');
mdlAGT_jnj = arima('ARLags', 1, 'Variance', garch(1,1), 'Distribution', 'T');
mdlAGT_lbyta = arima('ARLags', 1, 'Variance', garch(1,3), 'Distribution', 'T');
mdlGJRt = gjr(1,1);  mdlGJRt.Distribution = 't'; mdlGJRt.Offset = NaN;

for t=1:length(f_ret)-21
    % Create the training set to fit models
    series_fit = full_ret(t:t+length(is_ret)-1, :);
    
    % Fit the model at time t = 1
    if t == 1
        % 1. BX series
        % GARCH(1,1) model
        [EstMdl_gt_bx, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,1), 'display', 'off');
        v_gt_bx = infer(EstMdl_gt_bx, series_fit(:,1)); 
        sd_gt_bx = sqrt(v_gt_bx); dfGt_bx = EstMdl_gt_bx.Distribution.DoF;
        % AR(1)-GARCH(1,1) model
        [EstMdl_agt_bx, EstParamCov, LLF, info] = estimate(mdlAGT_bx, series_fit(:,1), 'display', 'off');
        [e_agt_bx, v_agt_bx, logL] = infer(EstMdl_agt_bx, series_fit(:,1));
        sd_agt_bx = sqrt(v_agt_bx); dfAGt_bx = EstMdl_agt_bx.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_bx, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,1), 'display', 'off');
        v_gjrt_bx = infer(EstMdl_gjrt_bx, series_fit(:,1)); 
        sd_gjrt_bx = sqrt(v_gjrt_bx); dfGJRt_bx = EstMdl_gjrt_bx.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_bx = EstMdl_gt_bx.Offset; p0GJRt_bx = EstMdl_gjrt_bx.Offset;
        
        % 2. CTB series
        % GARCH(1,1) model
        [EstMdl_gt_ctb, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,2), 'display', 'off');
        v_gt_ctb = infer(EstMdl_gt_ctb, series_fit(:,2)); 
        sd_gt_ctb = sqrt(v_gt_ctb); dfGt_ctb = EstMdl_gt_ctb.Distribution.DoF;
        % AR(1)-GARCH(5,1) model
        [EstMdl_agt_ctb, EstParamCov, LLF, info] = estimate(mdlAGT_ctb, series_fit(:,2), 'display', 'off');
        [e_agt_ctb, v_agt_ctb, logL] = infer(EstMdl_agt_ctb, series_fit(:,2));
        sd_agt_ctb = sqrt(v_agt_ctb); dfAGt_ctb = EstMdl_agt_ctb.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_ctb, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,2), 'display', 'off');
        v_gjrt_ctb = infer(EstMdl_gjrt_ctb, series_fit(:,2)); 
        sd_gjrt_ctb = sqrt(v_gjrt_ctb); dfGJRt_ctb = EstMdl_gjrt_ctb.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_ctb = EstMdl_gt_ctb.Offset; p0GJRt_ctb = EstMdl_gjrt_ctb.Offset;
               
        % 3. PSX series
        % GARCH(1,1) model
        [EstMdl_gt_psx, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,3), 'display', 'off');
        v_gt_psx = infer(EstMdl_gt_psx, series_fit(:,3));
        sd_gt_psx = sqrt(v_gt_psx); dfGt_psx = EstMdl_gt_psx.Distribution.DoF;
        % AR(1)-GARCH(4,2) model
        [EstMdl_agt_psx, EstParamCov, LLF, info] = estimate(mdlAGT_psx, series_fit(:,3), 'display', 'off');
        [e_agt_psx, v_agt_psx, logL] = infer(EstMdl_agt_psx, series_fit(:,3));
        sd_agt_psx = sqrt(v_agt_psx);  dfAGt_psx = EstMdl_agt_psx.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_psx, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,3), 'display', 'off');
        v_gjrt_psx = infer(EstMdl_gjrt_psx, series_fit(:,3));
        sd_gjrt_psx = sqrt(v_gjrt_psx); dfGJRt_psx = EstMdl_gjrt_psx.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_psx = EstMdl_gt_psx.Offset; p0GJRt_psx = EstMdl_gjrt_psx.Offset;

        % 4. JNJ series
        % GARCH(1,1) model
        [EstMdl_gt_jnj, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,4), 'display', 'off');
        v_gt_jnj = infer(EstMdl_gt_jnj, series_fit(:,4));
        sd_gt_jnj = sqrt(v_gt_jnj); dfGt_jnj = EstMdl_gt_jnj.Distribution.DoF;
        % AR(1)-GARCH(1,1) model
        [EstMdl_agt_jnj, EstParamCov, LLF, info] = estimate(mdlAGT_jnj, series_fit(:,4), 'display', 'off');
        [e_agt_jnj, v_agt_jnj, logL] = infer(EstMdl_agt_jnj, series_fit(:,4));
        sd_agt_jnj = sqrt(v_agt_jnj);  dfAGt_jnj = EstMdl_agt_jnj.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_jnj, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,4), 'display', 'off');
        v_gjrt_jnj = infer(EstMdl_gjrt_jnj, series_fit(:,4)); 
        sd_gjrt_jnj = sqrt(v_gjrt_jnj); dfGJRt_jnj = EstMdl_gjrt_jnj.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_jnj = EstMdl_gt_jnj.Offset; p0GJRt_jnj = EstMdl_gjrt_jnj.Offset;
        
        % 5. LBYTA series
        % GARCH(1,1) model
        [EstMdl_gt_lbyta, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,5), 'display', 'off');
        v_gt_lbyta = infer(EstMdl_gt_lbyta, series_fit(:,5));
        sd_gt_lbyta = sqrt(v_gt_lbyta); dfGt_lbyta = EstMdl_gt_lbyta.Distribution.DoF;
        % AR(1)-GARCH(1,1) model
        [EstMdl_agt_lbyta, EstParamCov, LLF, info] = estimate(mdlAGT_lbyta, series_fit(:,5), 'display', 'off');
        [e_agt_lbyta, v_agt_lbyta, logL] = infer(EstMdl_agt_lbyta, series_fit(:,5));
        sd_agt_lbyta = sqrt(v_agt_lbyta); dfAGt_lbyta = EstMdl_agt_lbyta.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_lbyta, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,5), 'display', 'off');
        v_gjrt_lbyta = infer(EstMdl_gjrt_lbyta, series_fit(:,5)); 
        sd_gjrt_lbyta = sqrt(v_gjrt_lbyta); dfGJRt_lbyta = EstMdl_gjrt_lbyta.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_lbyta = EstMdl_gt_lbyta.Offset; p0GJRt_lbyta = EstMdl_gjrt_lbyta.Offset;

        % 1. Generate VaR volatility and return forecasts for BX        
        [ret_bx(t:t+20, 1), YMSE_bx , v_bx] = forecast(EstMdl_agt_bx, 21, 'Y0', series_fit(:,1));
        upper_bx(t:t+20, 1) = ret_bx(t:t+20, 1) + tinv(0.95, dfAGt_bx)*sqrt(YMSE_bx); 
        lower_bx(t:t+20, 1) = ret_bx(t:t+20, 1) - tinv(0.95, dfAGt_bx)*sqrt(YMSE_bx);
        sig_bx(t:t+20, 2) = sqrt(v_bx); 
        VaR1_bx(t:t+20, 2) = ret_bx(t:t+20, 1)+tinv(0.01, dfAGt_bx)*sig_bx(t:t+20, 2)*sqrt((dfAGt_bx-2)/dfAGt_bx);
        VaR5_bx(t:t+20, 2) = ret_bx(t:t+20, 1)+tinv(0.05, dfAGt_bx)*sig_bx(t:t+20, 2)*sqrt((dfAGt_bx-2)/dfAGt_bx);
        sig_bx(t:t+20, 3) = sqrt(forecast(EstMdl_gjrt_bx, 21, 'Y0', series_fit(:,1)));      
        VaR1_bx(t:t+20, 3) = p0GJRt_bx+tinv(0.01, dfGJRt_bx)*sig_bx(t:t+20, 3)*sqrt((dfGJRt_bx-2)/dfGJRt_bx);
        VaR5_bx(t:t+20, 3) = p0GJRt_bx+tinv(0.05, dfGJRt_bx)*sig_bx(t:t+20, 3)*sqrt((dfGJRt_bx-2)/dfGJRt_bx);
        sig_bx(t:t+20, 1) = sqrt(forecast(EstMdl_gt_bx, 21, 'Y0', series_fit(:,1)));
        VaR1_bx(t:t+20, 1) = p0Gt_bx+tinv(0.01, dfGt_bx)*sig_bx(t:t+20, 1)*sqrt((dfGt_bx-2)/dfGt_bx);     
        VaR5_bx(t:t+20, 1) = p0Gt_bx+tinv(0.05, dfGt_bx)*sig_bx(t:t+20, 1)*sqrt((dfGt_bx-2)/dfGt_bx);    
        
        % 2. Generate VaR volatility and return forecasts for CTB
        [ret_ctb(t:t+20, 1), YMSE_ctb, v_ctb] = forecast(EstMdl_agt_ctb, 21, 'Y0', series_fit(:,2));
        upper_ctb(t:t+20, 1) = ret_ctb(t:t+20, 1) + tinv(0.95, dfAGt_ctb)*sqrt(YMSE_ctb); 
        lower_ctb(t:t+20, 1) = ret_ctb(t:t+20, 1) - tinv(0.95, dfAGt_ctb)*sqrt(YMSE_ctb);
        sig_ctb(t:t+20, 2) = sqrt(v_ctb); 
        VaR1_ctb(t:t+20, 2) = ret_ctb(t:t+20, 1)+tinv(0.01, dfAGt_ctb)*sig_ctb(t:t+20, 2)*sqrt((dfAGt_ctb-2)/dfAGt_ctb);
        VaR5_ctb(t:t+20, 2) = ret_ctb(t:t+20, 1)+tinv(0.05, dfAGt_ctb)*sig_ctb(t:t+20, 2)*sqrt((dfAGt_ctb-2)/dfAGt_ctb);
        sig_ctb(t:t+20, 3) = sqrt(forecast(EstMdl_gjrt_ctb, 21, 'Y0', series_fit(:,2)));
        VaR1_ctb(t:t+20, 3) = p0GJRt_ctb+tinv(0.01, dfGJRt_ctb)*sig_ctb(t:t+20, 3)*sqrt((dfGJRt_ctb-2)/dfGJRt_ctb);
        VaR5_ctb(t:t+20, 3) = p0GJRt_ctb+tinv(0.05, dfGJRt_ctb)*sig_ctb(t:t+20, 3)*sqrt((dfGJRt_ctb-2)/dfGJRt_ctb);
        sig_ctb(t:t+20, 1) = sqrt(forecast(EstMdl_gt_ctb, 21, 'Y0', series_fit(:,2)));
        VaR1_ctb(t:t+20, 1) = p0Gt_ctb+tinv(0.01, dfGt_ctb)*sig_ctb(t:t+20, 1)*sqrt((dfGt_ctb-2)/dfGt_ctb);     
        VaR5_ctb(t:t+20, 1) = p0Gt_ctb+tinv(0.05, dfGt_ctb)*sig_ctb(t:t+20, 1)*sqrt((dfGt_ctb-2)/dfGt_ctb);   

        % Generate VaR, volatility and return forecasts for PSX
        [ret_psx(t:t+20, 1), YMSE_psx, v_psx] = forecast(EstMdl_agt_psx, 21, 'Y0', series_fit(:,3));
        upper_psx(t:t+20, 1) = ret_psx(t:t+20, 1) + tinv(0.95, dfAGt_psx)*sqrt(YMSE_psx);
        lower_psx(t:t+20, 1) = ret_psx(t:t+20, 1) - tinv(0.95, dfAGt_psx)*sqrt(YMSE_psx);
        sig_psx(t:t+20, 2) = sqrt(v_psx); 
        VaR1_psx(t:t+20, 2) = ret_psx(t:t+20, 1)+tinv(0.01, dfAGt_psx)*sig_psx(t:t+20, 2)*sqrt((dfAGt_psx-2)/dfAGt_psx);
        VaR5_psx(t:t+20, 2) = ret_psx(t:t+20, 1)+tinv(0.05, dfAGt_psx)*sig_psx(t:t+20, 2)*sqrt((dfAGt_psx-2)/dfAGt_psx);
        sig_psx(t:t+20, 3) = sqrt(forecast(EstMdl_gjrt_psx, 21, 'Y0', series_fit(:,3)));
        VaR1_psx(t:t+20, 3) = p0GJRt_psx+tinv(0.01, dfGJRt_psx)*sig_psx(t:t+20, 3)*sqrt((dfGJRt_psx-2)/dfGJRt_psx);
        VaR5_psx(t:t+20, 3) = p0GJRt_psx+tinv(0.05, dfGJRt_psx)*sig_psx(t:t+20, 3)*sqrt((dfGJRt_psx-2)/dfGJRt_psx);
        sig_psx(t:t+20, 1) = sqrt(forecast(EstMdl_gt_psx, 21, 'Y0', series_fit(:,3)));
        VaR1_psx(t:t+20, 1) = p0Gt_psx+tinv(0.01, dfGt_psx)*sig_psx(t:t+20, 1)*sqrt((dfGt_psx-2)/dfGt_psx);     
        VaR5_psx(t:t+20, 1) = p0Gt_psx+tinv(0.05, dfGt_psx)*sig_psx(t:t+20, 1)*sqrt((dfGt_psx-2)/dfGt_psx);  

        % Generate VaR,volatility and return forecasts for JNJ
        [ret_jnj(t:t+20, 1), YMSE_jnj, v_jnj] = forecast(EstMdl_agt_jnj, 21, 'Y0', series_fit(:,4));
        upper_jnj(t:t+20, 1) = ret_jnj(t:t+20, 1) + tinv(0.95, dfAGt_jnj)*sqrt(YMSE_jnj); 
        lower_jnj(t:t+20, 1) = ret_jnj(t:t+20, 1) - tinv(0.95, dfAGt_jnj)*sqrt(YMSE_jnj);
        sig_jnj(t:t+20, 2) = sqrt(v_jnj); 
        VaR1_jnj(t:t+20, 2) = ret_jnj(t:t+20, 1)+tinv(0.01, dfAGt_jnj)*sig_jnj(t:t+20, 2)*sqrt((dfAGt_jnj-2)/dfAGt_jnj);
        VaR5_jnj(t:t+20, 2) = ret_jnj(t:t+20, 1)+tinv(0.05, dfAGt_jnj)*sig_jnj(t:t+20, 2)*sqrt((dfAGt_jnj-2)/dfAGt_jnj);     
        sig_jnj(t:t+20, 3) = sqrt(forecast(EstMdl_gjrt_jnj, 21, 'Y0', series_fit(:,4)));
        VaR1_jnj(t:t+20, 3) = p0GJRt_jnj+tinv(0.01, dfGJRt_jnj)*sig_jnj(t:t+20, 3)*sqrt((dfGJRt_jnj-2)/dfGJRt_jnj);
        VaR5_jnj(t:t+20, 3) = p0GJRt_jnj+tinv(0.05, dfGJRt_jnj)*sig_jnj(t:t+20, 3)*sqrt((dfGJRt_jnj-2)/dfGJRt_jnj);
        sig_jnj(t:t+20, 1) = sqrt(forecast(EstMdl_gt_jnj, 21, 'Y0', series_fit(:,4)));
        VaR1_jnj(t:t+20, 1) = p0Gt_jnj+tinv(0.01, dfGt_jnj)*sig_jnj(t:t+20, 1)*sqrt((dfGt_jnj-2)/dfGt_jnj);     
        VaR5_jnj(t:t+20, 1) = p0Gt_jnj+tinv(0.05, dfGt_jnj)*sig_jnj(t:t+20, 1)*sqrt((dfGt_jnj-2)/dfGt_jnj);  

        % Generate VaR, volatility and return forecasts for LBYTA
        [ret_lbyta(t:t+20, 1), YMSE_lbyta, v_lbyta] = forecast(EstMdl_agt_lbyta, 21, 'Y0', series_fit(:,5));
        upper_lbyta(t:t+20, 1) = ret_lbyta(t:t+20, 1) + tinv(0.95, dfAGt_lbyta)*sqrt(YMSE_lbyta); 
        lower_lbyta(t:t+20, 1) = ret_lbyta(t:t+20, 1) - tinv(0.95, dfAGt_lbyta)*sqrt(YMSE_lbyta);
        sig_lbyta(t:t+20, 2) = sqrt(v_lbyta); 
        VaR1_lbyta(t:t+20, 2) = ret_lbyta(t:t+20, 1)+tinv(0.01, dfAGt_lbyta)*sig_lbyta(t:t+20, 2)*sqrt((dfAGt_lbyta-2)/dfAGt_lbyta);
        VaR5_lbyta(t:t+20, 2) = ret_lbyta(t:t+20, 1)+tinv(0.05, dfAGt_lbyta)*sig_lbyta(t:t+20, 2)*sqrt((dfAGt_lbyta-2)/dfAGt_lbyta);
        sig_lbyta(t:t+20, 3) = sqrt(forecast(EstMdl_gjrt_lbyta, 21, 'Y0', series_fit(:,5)));  
        VaR1_lbyta(t:t+20, 3) = p0GJRt_lbyta+tinv(0.01, dfGJRt_lbyta)*sig_lbyta(t:t+20, 3)*sqrt((dfGJRt_lbyta-2)/dfGJRt_lbyta);
        VaR5_lbyta(t:t+20, 3) = p0GJRt_lbyta+tinv(0.05, dfGJRt_lbyta)*sig_lbyta(t:t+20, 3)*sqrt((dfGJRt_lbyta-2)/dfGJRt_lbyta);
        sig_lbyta(t:t+20, 1) = sqrt(forecast(EstMdl_gt_lbyta, 21, 'Y0', series_fit(:,5)));
        VaR1_lbyta(t:t+20, 1) = p0Gt_lbyta+tinv(0.01, dfGt_lbyta)*sig_lbyta(t:t+20, 1)*sqrt((dfGt_lbyta-2)/dfGt_lbyta);     
        VaR5_lbyta(t:t+20, 1) = p0Gt_lbyta+tinv(0.05, dfGt_lbyta)*sig_lbyta(t:t+20, 1)*sqrt((dfGt_lbyta-2)/dfGt_lbyta);  
    end
       
    % Re-fit the model at specified period
    if mod(t, 21) == 0
        % 1. BX series
        % GARCH(1,1) model
        [EstMdl_gt_bx, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,1), 'display', 'off');
        v_gt_bx = infer(EstMdl_gt_bx, series_fit(:,1)); 
        sd_gt_bx = sqrt(v_gt_bx); dfGt_bx = EstMdl_gt_bx.Distribution.DoF;
        % AR(1)-GARCH(1,1) model
        [EstMdl_agt_bx, EstParamCov, LLF, info] = estimate(mdlAGT_bx, series_fit(:,1), 'display', 'off');
        [e_agt_bx, v_agt_bx, logL] = infer(EstMdl_agt_bx, series_fit(:,1));
        sd_agt_bx = sqrt(v_agt_bx); dfAGt_bx = EstMdl_agt_bx.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_bx, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,1), 'display', 'off');
        v_gjrt_bx = infer(EstMdl_gjrt_bx, series_fit(:,1)); 
        sd_gjrt_bx = sqrt(v_gjrt_bx); dfGJRt_bx = EstMdl_gjrt_bx.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_bx = EstMdl_gt_bx.Offset; p0GJRt_bx = EstMdl_gjrt_bx.Offset;
        
        % 2. CTB series
        % GARCH(1,1) model
        [EstMdl_gt_ctb, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,2), 'display', 'off');
        v_gt_ctb = infer(EstMdl_gt_ctb, series_fit(:,2)); 
        sd_gt_ctb = sqrt(v_gt_ctb); dfGt_ctb = EstMdl_gt_ctb.Distribution.DoF;
        % AR(1)-GARCH(5,1) model
        [EstMdl_agt_ctb, EstParamCov, LLF, info] = estimate(mdlAGT_ctb, series_fit(:,2), 'display', 'off');
        [e_agt_ctb, v_agt_ctb, logL] = infer(EstMdl_agt_ctb, series_fit(:,2));
        sd_agt_ctb = sqrt(v_agt_ctb); dfAGt_ctb = EstMdl_agt_ctb.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_ctb, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,2), 'display', 'off');
        v_gjrt_ctb = infer(EstMdl_gjrt_ctb, series_fit(:,2)); 
        sd_gjrt_ctb = sqrt(v_gjrt_ctb); dfGJRt_ctb = EstMdl_gjrt_ctb.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_ctb = EstMdl_gt_ctb.Offset; p0GJRt_ctb = EstMdl_gjrt_ctb.Offset;
               
        % 3. PSX series
        % GARCH(1,1) model
        [EstMdl_gt_psx, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,3), 'display', 'off');
        v_gt_psx = infer(EstMdl_gt_psx, series_fit(:,3));
        sd_gt_psx = sqrt(v_gt_psx); dfGt_psx = EstMdl_gt_psx.Distribution.DoF;
        % AR(1)-GARCH(4,2) model
        [EstMdl_agt_psx, EstParamCov, LLF, info] = estimate(mdlAGT_psx, series_fit(:,3), 'display', 'off');
        [e_agt_psx, v_agt_psx, logL] = infer(EstMdl_agt_psx, series_fit(:,3));
        sd_agt_psx = sqrt(v_agt_psx);  dfAGt_psx = EstMdl_agt_psx.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_psx, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,3), 'display', 'off');
        v_gjrt_psx = infer(EstMdl_gjrt_psx, series_fit(:,3));
        sd_gjrt_psx = sqrt(v_gjrt_psx); dfGJRt_psx = EstMdl_gjrt_psx.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_psx = EstMdl_gt_psx.Offset; p0GJRt_psx = EstMdl_gjrt_psx.Offset;

        % 4. JNJ series
        % GARCH(1,1) model
        [EstMdl_gt_jnj, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,4), 'display', 'off');
        v_gt_jnj = infer(EstMdl_gt_jnj, series_fit(:,4));
        sd_gt_jnj = sqrt(v_gt_jnj); dfGt_jnj = EstMdl_gt_jnj.Distribution.DoF;
        % AR(1)-GARCH(1,1) model
        [EstMdl_agt_jnj, EstParamCov, LLF, info] = estimate(mdlAGT_jnj, series_fit(:,4), 'display', 'off');
        [e_agt_jnj, v_agt_jnj, logL] = infer(EstMdl_agt_jnj, series_fit(:,4));
        sd_agt_jnj = sqrt(v_agt_jnj);  dfAGt_jnj = EstMdl_agt_jnj.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_jnj, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,4), 'display', 'off');
        v_gjrt_jnj = infer(EstMdl_gjrt_jnj, series_fit(:,4)); 
        sd_gjrt_jnj = sqrt(v_gjrt_jnj); dfGJRt_jnj = EstMdl_gjrt_jnj.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_jnj = EstMdl_gt_jnj.Offset; p0GJRt_jnj = EstMdl_gjrt_jnj.Offset;
        
        % 5. LBYTA series
        % GARCH(1,1) model
        [EstMdl_gt_lbyta, EstParamCov, LLF, info] = estimate(mdlGT, series_fit(:,5), 'display', 'off');
        v_gt_lbyta = infer(EstMdl_gt_lbyta, series_fit(:,5));
        sd_gt_lbyta = sqrt(v_gt_lbyta); dfGt_lbyta = EstMdl_gt_lbyta.Distribution.DoF;
        % AR(1)-GARCH(1,1) model
        [EstMdl_agt_lbyta, EstParamCov, LLF, info] = estimate(mdlAGT_lbyta, series_fit(:,5), 'display', 'off');
        [e_agt_lbyta, v_agt_lbyta, logL] = infer(EstMdl_agt_lbyta, series_fit(:,5));
        sd_agt_lbyta = sqrt(v_agt_lbyta); dfAGt_lbyta = EstMdl_agt_lbyta.Distribution.DoF;
        % GJR-GARCH(1,1) MODEL
        [EstMdl_gjrt_lbyta, EstParamCov, LLF, info] = estimate(mdlGJRt, series_fit(:,5), 'display', 'off');
        v_gjrt_lbyta = infer(EstMdl_gjrt_lbyta, series_fit(:,5)); 
        sd_gjrt_lbyta = sqrt(v_gjrt_lbyta); dfGJRt_lbyta = EstMdl_gjrt_lbyta.Distribution.DoF;
        % Save phi0 = constant in mean equations
        p0Gt_lbyta = EstMdl_gt_lbyta.Offset; p0GJRt_lbyta = EstMdl_gjrt_lbyta.Offset;
        
        % 1. Generate VaR, volatility and return forecasts for BX
        [ret_bx(t+1:t+21, 1), YMSE_bx , v_bx] = forecast(EstMdl_agt_bx, 21, 'Y0', series_fit(:,1));
        upper_bx(t+1:t+21, 1) = ret_bx(t+1:t+21, 1) + tinv(0.95, dfAGt_bx)*sqrt(YMSE_bx); 
        lower_bx(t+1:t+21, 1) = ret_bx(t+1:t+21, 1) - tinv(0.95, dfAGt_bx)*sqrt(YMSE_bx);
        sig_bx(t+1:t+21, 2) = sqrt(v_bx); 
        VaR1_bx(t+1:t+21, 2) = ret_bx(t+1:t+21, 1)+tinv(0.01, dfAGt_bx)*sig_bx(t+1:t+21, 2)*sqrt((dfAGt_bx-2)/dfAGt_bx);
        VaR5_bx(t+1:t+21, 2) = ret_bx(t+1:t+21, 1)+tinv(0.05, dfAGt_bx)*sig_bx(t+1:t+21, 2)*sqrt((dfAGt_bx-2)/dfAGt_bx);
        sig_bx(t+1:t+21, 3) = sqrt(forecast(EstMdl_gjrt_bx, 21, 'Y0', series_fit(:,1)));      
        VaR1_bx(t+1:t+21, 3) = p0GJRt_bx+tinv(0.01, dfGJRt_bx)*sig_bx(t+1:t+21, 3)*sqrt((dfGJRt_bx-2)/dfGJRt_bx);
        VaR5_bx(t+1:t+21, 3) = p0GJRt_bx+tinv(0.05, dfGJRt_bx)*sig_bx(t+1:t+21, 3)*sqrt((dfGJRt_bx-2)/dfGJRt_bx);
        sig_bx(t+1:t+21, 1) = sqrt(forecast(EstMdl_gt_bx, 21, 'Y0', series_fit(:,1)));
        VaR1_bx(t+1:t+21, 1) = p0Gt_bx+tinv(0.01, dfGt_bx)*sig_bx(t+1:t+21, 1)*sqrt((dfGt_bx-2)/dfGt_bx);     
        VaR5_bx(t+1:t+21, 1) = p0Gt_bx+tinv(0.05, dfGt_bx)*sig_bx(t+1:t+21, 1)*sqrt((dfGt_bx-2)/dfGt_bx);

        % Generate VaR, volatility and return forecasts for CTB
        [ret_ctb(t+1:t+21, 1), YMSE_ctb, v_ctb] = forecast(EstMdl_agt_ctb, 21, 'Y0', series_fit(:,2));
        upper_ctb(t+1:t+21, 1) = ret_ctb(t+1:t+21, 1) + tinv(0.95, dfAGt_ctb)*sqrt(YMSE_ctb); 
        lower_ctb(t+1:t+21, 1) = ret_ctb(t+1:t+21, 1) - tinv(0.95, dfAGt_ctb)*sqrt(YMSE_ctb);
        sig_ctb(t+1:t+21, 2) = sqrt(v_ctb); 
        VaR1_ctb(t+1:t+21, 2) = ret_ctb(t+1:t+21, 1)+tinv(0.01, dfAGt_ctb)*sig_ctb(t+1:t+21, 2)*sqrt((dfAGt_ctb-2)/dfAGt_ctb);
        VaR5_ctb(t+1:t+21, 2) = ret_ctb(t+1:t+21, 1)+tinv(0.05, dfAGt_ctb)*sig_ctb(t+1:t+21, 2)*sqrt((dfAGt_ctb-2)/dfAGt_ctb);
        sig_ctb(t+1:t+21, 3) = sqrt(forecast(EstMdl_gjrt_ctb, 21, 'Y0', series_fit(:,2)));
        VaR1_ctb(t+1:t+21, 3) = p0GJRt_ctb+tinv(0.01, dfGJRt_ctb)*sig_ctb(t+1:t+21, 3)*sqrt((dfGJRt_ctb-2)/dfGJRt_ctb);
        VaR5_ctb(t+1:t+21, 3) = p0GJRt_ctb+tinv(0.05, dfGJRt_ctb)*sig_ctb(t+1:t+21, 3)*sqrt((dfGJRt_ctb-2)/dfGJRt_ctb);
        sig_ctb(t+1:t+21, 1) = sqrt(forecast(EstMdl_gt_ctb, 21, 'Y0', series_fit(:,2)));
        VaR1_ctb(t+1:t+21, 1) = p0Gt_ctb+tinv(0.01, dfGt_ctb)*sig_ctb(t+1:t+21, 1)*sqrt((dfGt_ctb-2)/dfGt_ctb);     
        VaR5_ctb(t+1:t+21, 1) = p0Gt_ctb+tinv(0.05, dfGt_ctb)*sig_ctb(t+1:t+21, 1)*sqrt((dfGt_ctb-2)/dfGt_ctb);

        % Generate VaR, volatility and return forecasts for PSX
        [ret_psx(t+1:t+21, 1), YMSE_psx, v_psx] = forecast(EstMdl_agt_psx, 21, 'Y0', series_fit(:,3));
        upper_psx(t+1:t+21, 1) = ret_psx(t+1:t+21, 1) + tinv(0.95, dfAGt_psx)*sqrt(YMSE_psx);
        lower_psx(t+1:t+21, 1) = ret_psx(t+1:t+21, 1) - tinv(0.95, dfAGt_psx)*sqrt(YMSE_psx);
        sig_psx(t+1:t+21, 2) = sqrt(v_psx); 
        VaR1_psx(t+1:t+21, 2) = ret_psx(t+1:t+21, 1)+tinv(0.01, dfAGt_psx)*sig_psx(t, 2)*sqrt((dfAGt_psx-2)/dfAGt_psx);
        VaR5_psx(t+1:t+21, 2) = ret_psx(t+1:t+21, 1)+tinv(0.05, dfAGt_psx)*sig_psx(t, 2)*sqrt((dfAGt_psx-2)/dfAGt_psx);
        sig_psx(t+1:t+21, 3) = sqrt(forecast(EstMdl_gjrt_psx, 21, 'Y0', series_fit(:,3)));
        VaR1_psx(t+1:t+21, 3) = p0GJRt_psx+tinv(0.01, dfGJRt_psx)*sig_psx(t+1:t+21, 3)*sqrt((dfGJRt_psx-2)/dfGJRt_psx);
        VaR5_psx(t+1:t+21, 3) = p0GJRt_psx+tinv(0.05, dfGJRt_psx)*sig_psx(t+1:t+21, 3)*sqrt((dfGJRt_psx-2)/dfGJRt_psx);
        sig_psx(t+1:t+21, 1) = sqrt(forecast(EstMdl_gt_psx, 21, 'Y0', series_fit(:,3)));
        VaR1_psx(t+1:t+21, 1) = p0Gt_psx+tinv(0.01, dfGt_psx)*sig_psx(t+1:t+21, 1)*sqrt((dfGt_psx-2)/dfGt_psx);     
        VaR5_psx(t+1:t+21, 1) = p0Gt_psx+tinv(0.05, dfGt_psx)*sig_psx(t+1:t+21, 1)*sqrt((dfGt_psx-2)/dfGt_psx);

        % Generate VaR, volatility and return forecasts for JNJ
        [ret_jnj(t+1:t+21, 1), YMSE_jnj, v_jnj] = forecast(EstMdl_agt_jnj, 21, 'Y0', series_fit(:,4));
        upper_jnj(t+1:t+21, 1) = ret_jnj(t+1:t+21, 1) + tinv(0.95, dfAGt_jnj)*sqrt(YMSE_jnj); 
        lower_jnj(t+1:t+21, 1) = ret_jnj(t+1:t+21, 1) - tinv(0.95, dfAGt_jnj)*sqrt(YMSE_jnj);
        sig_jnj(t+1:t+21, 2) = sqrt(v_jnj); 
        VaR1_jnj(t+1:t+21, 2) = ret_jnj(t+1:t+21, 1)+tinv(0.01, dfAGt_jnj)*sig_jnj(t+1:t+21, 2)*sqrt((dfAGt_jnj-2)/dfAGt_jnj);
        VaR5_jnj(t+1:t+21, 2) = ret_jnj(t+1:t+21, 1)+tinv(0.05, dfAGt_jnj)*sig_jnj(t+1:t+21, 2)*sqrt((dfAGt_jnj-2)/dfAGt_jnj);
        sig_jnj(t+1:t+21, 3) = sqrt(forecast(EstMdl_gjrt_jnj, 21, 'Y0', series_fit(:,4)));
        VaR1_jnj(t+1:t+21, 3) = p0GJRt_jnj+tinv(0.01, dfGJRt_jnj)*sig_jnj(t+1:t+21, 3)*sqrt((dfGJRt_jnj-2)/dfGJRt_jnj);
        VaR5_jnj(t+1:t+21, 3) = p0GJRt_jnj+tinv(0.05, dfGJRt_jnj)*sig_jnj(t+1:t+21, 3)*sqrt((dfGJRt_jnj-2)/dfGJRt_jnj);
        sig_jnj(t+1:t+21, 1) = sqrt(forecast(EstMdl_gt_jnj, 21, 'Y0', series_fit(:,4)));
        VaR1_jnj(t+1:t+21, 1) = p0Gt_jnj+tinv(0.01, dfGt_jnj)*sig_jnj(t+1:t+21, 1)*sqrt((dfGt_jnj-2)/dfGt_jnj);     
        VaR5_jnj(t+1:t+21, 1) = p0Gt_jnj+tinv(0.05, dfGt_jnj)*sig_jnj(t+1:t+21, 1)*sqrt((dfGt_jnj-2)/dfGt_jnj);
        
        % Generate VaR, volatility and return forecasts for LBYTA
        [ret_lbyta(t+1:t+21, 1), YMSE_lbyta, v_lbyta] = forecast(EstMdl_agt_lbyta, 21, 'Y0', series_fit(:,5));
        upper_lbyta(t+1:t+21, 1) = ret_lbyta(t+1:t+21, 1) + tinv(0.95, dfAGt_lbyta)*sqrt(YMSE_lbyta); 
        lower_lbyta(t+1:t+21, 1) = ret_lbyta(t+1:t+21, 1) - tinv(0.95, dfAGt_lbyta)*sqrt(YMSE_lbyta);
        sig_lbyta(t+1:t+21, 2) = sqrt(v_lbyta); 
        VaR1_lbyta(t+1:t+21, 2) = ret_lbyta(t+1:t+21, 1)+tinv(0.01, dfAGt_lbyta)*sig_lbyta(t+1:t+21, 2)*sqrt((dfAGt_lbyta-2)/dfAGt_lbyta);
        VaR5_lbyta(t+1:t+21, 2) = ret_lbyta(t+1:t+21, 1)+tinv(0.05, dfAGt_lbyta)*sig_lbyta(t+1:t+21, 2)*sqrt((dfAGt_lbyta-2)/dfAGt_lbyta);
        sig_lbyta(t+1:t+21, 3) = sqrt(forecast(EstMdl_gjrt_lbyta, 21, 'Y0', series_fit(:,5)));  
        VaR1_lbyta(t+1:t+21, 3) = p0GJRt_lbyta+tinv(0.01, dfGJRt_lbyta)*sig_lbyta(t+1:t+21, 3)*sqrt((dfGJRt_lbyta-2)/dfGJRt_lbyta);
        VaR5_lbyta(t+1:t+21, 3) = p0GJRt_lbyta+tinv(0.05, dfGJRt_lbyta)*sig_lbyta(t+1:t+21, 3)*sqrt((dfGJRt_lbyta-2)/dfGJRt_lbyta);
        sig_lbyta(t+1:t+21, 1) = sqrt(forecast(EstMdl_gt_lbyta, 21, 'Y0', series_fit(:,5)));
        VaR1_lbyta(t+1:t+21, 1) = p0Gt_lbyta+tinv(0.01, dfGt_lbyta)*sig_lbyta(t, 1)*sqrt((dfGt_lbyta-2)/dfGt_lbyta);     
        VaR5_lbyta(t+1:t+21, 1) = p0Gt_lbyta+tinv(0.05, dfGt_lbyta)*sig_lbyta(t, 1)*sqrt((dfGt_lbyta-2)/dfGt_lbyta);
    end
end

%%
%==================
% +++ Part Six +++
%==================

% Combine all monthly return and volatility forecasts under the same model
ret_Mf_AGt = [ret_bx ret_ctb ret_psx ret_jnj ret_lbyta];
sig_Mf_Gt = [sig_bx(:, 1) sig_ctb(:, 1) sig_psx(:, 1) sig_jnj(:, 1) sig_lbyta(:, 1)];
sig_Mf_AGt = [sig_bx(:, 2) sig_ctb(:, 2) sig_psx(:, 2) sig_jnj(:, 2) sig_lbyta(:, 2)];
sig_Mf_GJRt = [sig_bx(:, 3) sig_ctb(:, 3) sig_psx(:, 3) sig_jnj(:, 3) sig_lbyta(:, 3)];
% Combine all monthly VaR forecasts under the same model for later portfolio
% construction
ARCH_VaR_mf                        = zeros(length(f_ret),5);

for i=1:5
    series = is_ret(:,i);
    for t=1:length(f_ret)
        ret_is = full_ret(t:length(is_ret)+t-1,i);
        if mod(t,21)==0|t==1
            Mdl = garch(0,i_sic);Mdl.Offset=NaN;Mdl.Distribution='t'; %ARCH(5)
            [EstMdl,EstParamCov,LLF,info]=estimate(Mdl,ret_is,'display','off');
            sigma(t:t+20,i)=forecast(EstMdl,21,'Y0',ret_is);
            p0Gt=EstMdl.Offset;
            dfGt   = EstMdl.Distribution.DoF;
            SFAOgt(t:t+20,i) = sqrt(sigma(t:t+20,i));
            ARCH_VaR_mf(t:t+20,i) = p0Gt+tinv(0.05,dfGt)*SFAOgt(t:t+20,i)*sqrt((dfGt-2)/dfGt);
        end
    end
end
VaR5_Mf_ACt = ARCH_VaR_mf(1:end-20,:);
%
VaR1_Mf_Gt = [VaR1_bx(:,1) VaR1_ctb(:,1) VaR1_psx(:,1) VaR1_jnj(:,1) VaR1_lbyta(:,1)];
VaR5_Mf_Gt = [VaR5_bx(:,1) VaR5_ctb(:,1) VaR5_psx(:,1) VaR5_jnj(:,1) VaR5_lbyta(:,1)];
VaR1_Mf_AGt = [VaR1_bx(:,2) VaR1_ctb(:,2) VaR1_psx(:,2) VaR1_jnj(:,2) VaR1_lbyta(:,2)];
VaR5_Mf_AGt = [VaR5_bx(:,2) VaR5_ctb(:,2) VaR5_psx(:,2) VaR5_jnj(:,2) VaR5_lbyta(:,2)];
VaR1_Mf_GJRt = [VaR1_bx(:,3) VaR1_ctb(:,3) VaR1_psx(:,3) VaR1_jnj(:,3) VaR1_lbyta(:,3)];
VaR5_Mf_GJRt = [VaR5_bx(:,3) VaR5_ctb(:,3) VaR5_psx(:,3) VaR5_jnj(:,3) VaR5_lbyta(:,3)];
VaR5_Df_ACt = ARCH_VaR_f;


% Construct portfolio based on VaR5 at monthly updating periods
%load DFVaR5ARGt.mat
%load DFRetARGt.mat

weights_VaR5d_ACt = 0; weights_VaR5m_ACt = 0; 
weights_VaR5d_Gt = 0; weights_VaR5m_Gt = 0; 
weights_VaR5d_ARGt = 0; weights_VaR5m_ARGt = 0; 
weights_VaR5d_GJRt = 0; weights_VaR5m_GJRt = 0; 
weights_Ret_d = 0; weights_Ret_m = 0;
weights_Equ = 0;
for t=1:length(f_ret)   
    for i = 1:5
        weights_Ret_d(t,i) = ret_Df_AGt(t,i) / sum(ret_Df_AGt(t, :));
        % Calculate weights at the minimum daily VaR5 criterion
        VaR5_Df_ACt_t = 1./VaR5_Df_ACt; VaR5_Df_Gt_t = 1./VaR5_Df_Gt;
        VaR5_Df_AGt_t = 1./VaR5_Df_AGt; VaR5_Df_GJRt_t = 1./VaR5_Df_GJRt;
        weights_VaR5d_ACt(t,i) = VaR5_Df_ACt_t(t, i) / sum(VaR5_Df_ACt_t(t, :));
        weights_VaR5d_Gt(t,i) = VaR5_Df_Gt_t(t, i) / sum(VaR5_Df_Gt_t(t, :));  
        weights_VaR5d_ARGt(t,i) = VaR5_Df_AGt_t(t, i) / sum(VaR5_Df_AGt_t(t, :));  
        weights_VaR5d_GJRt(t,i) = VaR5_Df_GJRt_t(t, i) / sum(VaR5_Df_GJRt_t(t, :)); 
        
        % Calculate weights at the minimum monthly VaR5 criteron
        VaR5_Mf_ACt_t = 1./VaR5_Mf_ACt; VaR5_Mf_Gt_t = 1./VaR5_Mf_Gt; 
        VaR5_Mf_AGt_t = 1./VaR5_Mf_AGt; VaR5_Mf_GJRt_t = 1./VaR5_Mf_GJRt;        
        if t == 1
            weights_Ret_m(t:t+20,i) = mean(ret_Mf_AGt(t:t+20,i))/sum(mean(ret_Mf_AGt(t:t+20,:)));
            weights_VaR5m_ACt(t:t+20,i) = mean(VaR5_Mf_ACt_t(t:t+20,i)) / sum(mean(VaR5_Mf_ACt_t(t:t+20, :)));
            weights_VaR5m_Gt(t:t+20,i) = mean(VaR5_Mf_Gt_t(t:t+20,i)) / sum(mean(VaR5_Mf_Gt_t(t:t+20, :)));
            weights_VaR5m_ARGt(t:t+20,i) = mean(VaR5_Mf_AGt_t(t:t+20,i)) / sum(mean(VaR5_Mf_AGt_t(t:t+20, :)));
            weights_VaR5m_GJRt(t:t+20,i) = mean(VaR5_Mf_GJRt_t(t:t+20,i)) / sum(mean(VaR5_Mf_GJRt_t(t:t+20, :)));
        end
        if mod(t, 21) == 0 && t <= 231
            weights_Ret_m(t+1:t+21,i) = mean(ret_Mf_AGt(t+1:t+21,i))/sum(mean(ret_Mf_AGt(t+1:t+21,:)));
            weights_VaR5m_ACt(t+1:t+21,i) = mean(VaR5_Mf_ACt_t(t+1:t+21,i)) / sum(mean(VaR5_Mf_ACt_t(t+1:t+21, :)));
            weights_VaR5m_Gt(t+1:t+21,i) = mean(VaR5_Mf_Gt_t(t+1:t+21,i)) / sum(mean(VaR5_Mf_Gt_t(t+1:t+21, :)));
            weights_VaR5m_ARGt(t+1:t+21,i) = mean(VaR5_Mf_AGt_t(t+1:t+21,i)) / sum(mean(VaR5_Mf_AGt_t(t+1:t+21, :)));
            weights_VaR5m_GJRt(t+1:t+21,i) = mean(VaR5_Mf_GJRt_t(t+1:t+21,i)) / sum(mean(VaR5_Mf_GJRt_t(t+1:t+21, :)));
        end
    end               
end

% Calculate returns under the equal weights portfolio
retP_Equ = f_ret*[0.2 0.2 0.2 0.2 0.2]';
cum_retP_Equ = 1+cumsum(retP_Equ);

% Calculate daily returns and cumulative returns under each portfolio
retP_VaRd_ACt = sum((weights_VaR5d_ACt.*f_ret)')';
cum_retP_VaRd_ACt = 1+cumsum(retP_VaRd_ACt);
retP_VaRd_Gt = sum((weights_VaR5d_Gt.*f_ret)')';
cum_retP_VaRd_Gt = 1+cumsum(retP_VaRd_Gt);
retP_VaRd_ARGt = sum((weights_VaR5d_ARGt.*f_ret)')';
cum_retP_VaRd_ARGt = 1+cumsum(retP_VaRd_ARGt);
retP_VaRd_GJRt = sum((weights_VaR5d_GJRt.*f_ret)')';
cum_retP_VaRd_GJRt = 1+cumsum(retP_VaRd_GJRt);
retP_Ret_d = sum((weights_Ret_d.*f_ret)')';
cum_retP_Ret_d = 1+cumsum(retP_Ret_d);

% Calculate monthly returns and cumulative returns under each portfolio
retP_VaRm_ACt = sum((weights_VaR5m_ACt.*f_ret)')';
cum_retP_VaRm_ACt = 1+cumsum(retP_VaRm_ACt);
retP_VaRm_Gt = sum((weights_VaR5m_Gt.*f_ret)')';
cum_retP_VaRm_Gt = 1+cumsum(retP_VaRm_Gt);
retP_VaRm_ARGt = sum((weights_VaR5m_ARGt.*f_ret)')';
cum_retP_VaRm_ARGt = 1+cumsum(retP_VaRm_ARGt);
retP_VaRm_GJRt = sum((weights_VaR5m_GJRt.*f_ret)')';
cum_retP_VaRm_GJRt = 1+cumsum(retP_VaRm_GJRt); 
retP_Ret_m = sum((weights_Ret_m.*f_ret)')';
cum_retP_Ret_m = 1+cumsum(retP_Ret_m);

% Calculate Average returns under each portfolio
mean_equP = [mean(retP_Equ); mean(retP_Equ)];
mean_VarP_ACt = [mean(retP_VaRd_ACt); mean(retP_VaRm_ACt)];
mean_VarP_Gt = [mean(retP_VaRd_Gt); mean(retP_VaRm_Gt)];
mean_VarP_ARGt = [mean(retP_VaRd_ARGt); mean(retP_VaRm_ARGt)];
mean_VarP_GJRt = [mean(retP_VaRd_GJRt); mean(retP_VaRm_GJRt)];
mean_sum = table(mean_equP, mean_VarP_ACt, mean_VarP_Gt, mean_VarP_ARGt, mean_VarP_GJRt, ...
    'VariableNames', {'Equal_Weights', 'Min_VaR_ACt_Portfolio', ...
    'Min_VaR_Gt_Portfolio', 'Min_VaR_ARGt_Portfolio', 'Min_VaR_GJRt_Portfolio'}, ...
    'RowNames', {'Daily_Update', 'Monthly_Update'})

% Calculate SDs under each portfolio
std_equP = [std(retP_Equ); std(retP_Equ)];
std_VarP_ACt = [std(retP_VaRd_ACt); std(retP_VaRm_ACt)];
std_VarP_Gt = [std(retP_VaRd_Gt); std(retP_VaRm_Gt)];
std_VarP_ARGt = [std(retP_VaRd_ARGt); std(retP_VaRm_ARGt)];
std_VarP_GJRt = [std(retP_VaRd_GJRt); std(retP_VaRm_GJRt)];
std_sum = table(std_equP, std_VarP_ACt, std_VarP_Gt, std_VarP_ARGt, std_VarP_GJRt, ...
    'VariableNames', {'Equal_Weights', 'Min_VaR_ACt_Portfolio', ...
    'Min_VaR_Gt_Portfolio', 'Min_VaR_ARGt_Portfolio', 'Min_VaR_GJRt_Portfolio'}, ...
    'RowNames', {'Daily_Update', 'Monthly_Update'})

% Calculate Sharpe Ratio under each portfolio
sr_equP = [mean(retP_Equ)/std(retP_Equ); mean(retP_Equ)/std(retP_Equ)];
sr_VarP_ACt = [mean(retP_VaRd_ACt)/std(retP_VaRd_ACt); mean(retP_VaRm_ACt)/std(retP_VaRm_ACt)];
sr_VarP_Gt = [mean(retP_VaRd_Gt)/std(retP_VaRd_Gt); mean(retP_VaRm_Gt)/std(retP_VaRm_Gt)];
sr_VarP_ARGt = [mean(retP_VaRd_ARGt)/std(retP_VaRd_ARGt); mean(retP_VaRm_ARGt)/std(retP_VaRm_ARGt)];
sr_VarP_GJRt = [mean(retP_VaRd_GJRt)/std(retP_VaRd_GJRt); mean(retP_VaRm_GJRt)/std(retP_VaRm_GJRt)];
sr_sum = table(sr_equP, sr_VarP_ACt, sr_VarP_Gt, sr_VarP_ARGt, sr_VarP_GJRt, ...
    'VariableNames', {'Equal_Weights', 'Min_VaR_ACt_Portfolio', ...
    'Min_VaR_Gt_Portfolio', 'Min_VaR_ARGt_Portfolio', 'Min_VaR_GJRt_Portfolio'}, ...
    'RowNames', {'Daily_Update', 'Monthly_Update'})

% Create plots for cumulative returns at daily update
figure; plot(Date(s+1:end), cum_retP_Equ, 'y', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_VaRd_ACt, 'r', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_VaRd_Gt, 'c', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_VaRd_ARGt, 'b', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_VaRd_GJRt, 'g', 'LineWidth', 1.5); hold off
datetick('x', 'mmm/yy'); xlim([min(Date(s+1:end)) max(Date(s+1:end))]); 
title('Cumulative Returns under Daily Updating Portfolios'); 
legend('Equal-Weights', 'Min-VaR5-ACt Portfolio', 'Min-VaR5-Gt Portfolio', 'Min-VaR5-ARGt Portfolio', 'Min-VaR5-GJRt Portfolio', 'location', 'northwest');

% Create plots for cumulative returns at monthly update
figure; plot(Date(s+1:end), cum_retP_Equ, 'y', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_VaRm_ACt, 'r', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_VaRm_Gt, 'c', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_VaRm_ARGt, 'b', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_VaRm_GJRt, 'g', 'LineWidth', 1.5); hold off
datetick('x', 'mmm/yy'); xlim([min(Date(s+1:end)) max(Date(s+1:end))]); 
title('Cumulative Returns under Monthly Updating Portfolios'); 
legend('Equal-Weights', 'Min-VaR5-ACt Portfolio',  'Min-VaR5-Gt Portfolio', 'Min-VaR5-ARGt Portfolio', 'Min-VaR5-GJRt Portfolio', 'location', 'northwest');

%%
% Try Black-Litterman Model to dynamically update portfolio weights

% load returns_df.mat; load returns_mf.mat; 
% ret_df_AM = [ARMA_ret_df(:,1:2) ARMA_ret_df(:,5) ARMA_ret_df(:,3) ARMA_ret_df(:,4)]; clear ARMA_ret_df
% ret_mf_AM = [ARMA_ret_mf(:,1:2) ARMA_ret_mf(:,5) ARMA_ret_mf(:,3) ARMA_ret_mf(:,4)]; clear ARMA_ret_mf
% ret_df_AD = [ADHOC_ret_df(:,1:2) ADHOC_ret_df(:,5) ADHOC_ret_df(:,3) ADHOC_ret_df(:,4)]; clear ADHOC_ret_df
% ret_mf_AD = [ADHOC_ret_mf(:,1:2) ADHOC_ret_mf(:,5) ADHOC_ret_mf(:,3) ADHOC_ret_mf(:,4)]; clear ADHOC_ret_mf
% ret_df_CB = [COMB_ret_df(:,1:2) COMB_ret_df(:,5) COMB_ret_df(:,3) COMB_ret_df(:,4)]; clear COMB_ret_df
% ret_mf_CB = [COMB_ret_mf(:,1:2) COMB_ret_mf(:,5) COMB_ret_mf(:,3) COMB_ret_mf(:,4)]; clear COMB_ret_mf

% Initilise fixed parameters first 
tau = 0.5;%1/length(p_mat);
p_matrix = eye(5,5);
Am = 3;
ci = 0.95;

% Incorporate AR-GARCH returns as views
weights_blD = zeros(252,5); weights_blM = zeros(252,5);
for t=1:length(f_ret)  
    % Create the training set to fit models
    series_fit = full_ret(t:t+length(is_ret)-1, :);
    % Generate the historical covariance matrix
    cov_mat = cov(series_fit);
    % Calculate the omega matrix
    omega = (1/ci) * p_matrix*cov_mat*p_matrix';
    % Calculate mkt cap based on stock prices at time t
    mkt_cap = f_p(t,:);
    for i=1:5 
        X(i) = mkt_cap(i)/sum(mkt_cap); 
    end
    Xm = X';
    % Calculate equillibrium returns 
    pie = Am*cov_mat*Xm;
    q_vec = ret_Df_AGt(t, :)'; % Incorporate daily forecasting returns as views
    r_hat_left = inv((inv(tau*cov_mat))+p_matrix'*(inv(omega))*p_matrix);
    r_hat_right = ((inv(tau*cov_mat)) * pie) + (p_matrix'*inv(omega)*q_vec);
    r_hat = r_hat_left*r_hat_right;
    % Calculate the weights at each asset
    weights_D = ((1/Am) * inv(cov_mat) * r_hat)';
    for i = 1:5
        weights_blD(t,i) = weights_D(i)/sum(weights_D(:));
    end
    
    % Update portfolio weights monthly 
    if t == 1
        % Incorporate monthly forecasting returns as views
        views=0;
        for i=1:5
            views(t,i) = mean(ret_Mf_AGt(t:t+20,i))/sum(mean(ret_Mf_AGt(t:t+20,:)));
        end
        q_vec = views(t,:)';
        r_hat_left = inv((inv(tau*cov_mat))+p_matrix'*(inv(omega))*p_matrix);
        r_hat_right = ((inv(tau*cov_mat)) * pie) + (p_matrix'*inv(omega)*q_vec);
        r_hat = r_hat_left*r_hat_right;
        % Calculate the weights at each asset
        weights_M = ((1/Am) * inv(cov_mat) * r_hat)';
        for i = 1:5
            weights_blM(t,i) = weights_M(i)/sum(weights_M(:));
        end
    elseif mod(t,21) == 0 && t <= 232 
        % Incorporate monthly forecasting returns as views
        views=0;
        for i=1:5
            views(t,i) = mean(ret_Mf_AGt(t+1:t+21,i))/sum(mean(ret_Mf_AGt(t+1:t+21,:)));
        end
        q_vec = views(t,:)';
        r_hat_left = inv((inv(tau*cov_mat))+p_matrix'*(inv(omega))*p_matrix);
        r_hat_right = ((inv(tau*cov_mat)) * pie) + (p_matrix'*inv(omega)*q_vec);
        r_hat = r_hat_left*r_hat_right;
        % Calculate the weights at each asset
        weights_M = ((1/Am) * inv(cov_mat) * r_hat)';
        for i = 1:5
            weights_blM(t,i) = weights_M(i)/sum(weights_M(:));
        end
    else
        for i = 1:5
            weights_blM(t,i) = weights_M(i)/sum(weights_M(:));
        end
    end
end




retP_blD = sum((weights_blD.*f_ret)')';
cum_retP_blD = 1+cumsum(retP_blD);
retP_blM = sum((weights_blM.*f_ret)')';
cum_retP_blM = 1+cumsum(retP_blM);
% Calculate the portfolio cumulative returns under the max-return portfolio
retP_max_retD = sum((ARGARCH_Daily_optimal.*f_ret)')';
cum_retP_max_retD = 1+cumsum(retP_max_retD);
retP_max_retM = sum((ADHOC_Monthly_optimal.*f_ret)')';
cum_retP_max_retM = 1+cumsum(retP_max_retM);

% Create plots for cumulative returns at daily update
figure; plot(Date(s+1:end), cum_retP_Equ, 'y', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_max_retD, 'r', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_VaRd_ARGt, 'b', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_blD, 'k', 'LineWidth', 1.5); hold off
datetick('x', 'mmm/yy'); xlim([min(Date(s+1:end)) max(Date(s+1:end))]); 
title('Cumulative Returns under Daily Updating Strategies'); 
legend('Equal-Weights', 'Max-Ret-ARGt Portfolio', 'Min-VaR5-ARGt Portfolio', ...
    'AR-GARCH-t Corrected Black-Litterman Portfolio', 'location', 'northwest');

% Create plots for cumulative returns at monthly update
figure; plot(Date(s+1:end), cum_retP_Equ, 'y', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_max_retM, 'r', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_VaRm_ARGt, 'b', 'LineWidth', 1.5); hold on
plot(Date(s+1:end), cum_retP_blM, 'k', 'LineWidth', 1.5); hold off
datetick('x', 'mmm/yy'); xlim([min(Date(s+1:end)) max(Date(s+1:end))]); 
title('Cumulative Returns under Monthly Updating Strategies'); 
legend('Equal-Weights', 'Max-Ret-AdHoc Portfolio', 'Min-VaR5-ARGt Portfolio', ...
    'AR-GARCH-t Corrected Black-Litterman Portfolio', 'location', 'northwest');

% Calculate Sharpe Ratio under each portfolio
sr_equP = [mean(retP_Equ)/std(retP_Equ); mean(retP_Equ)/std(retP_Equ)];
sr_retP = [mean(retP_Ret_d)/std(retP_Ret_d); mean(retP_Ret_m)/std(retP_Ret_m)];
sr_VarP_Gt = [mean(retP_VaRd_Gt)/std(retP_VaRd_Gt); mean(retP_VaRm_Gt)/std(retP_VaRm_Gt)];
sr_VarP_ARGt = [mean(retP_VaRd_ARGt)/std(retP_VaRd_ARGt); mean(retP_VaRm_ARGt)/std(retP_VaRm_ARGt)];
sr_VarP_GJRt = [mean(retP_VaRd_GJRt)/std(retP_VaRd_GJRt); mean(retP_VaRm_GJRt)/std(retP_VaRm_GJRt)];

sr_bl = [mean(retP_blD)/std(retP_blD); mean(retP_blM)/std(retP_blM)];
sr_sum = table(sr_equP, sr_retP, sr_VarP_Gt, sr_VarP_ARGt, sr_VarP_GJRt, sr_bl, 'VariableNames', ...
    {'Equal_Weights', 'Max_Ret_Portfolio', 'Min_VaR_Gt_Portfolio', 'Min_VaR_ARGt_Portfolio', 'Min_VaR_GJRt_Portfolio', 'BL'}, ...
    'RowNames', {'Daily_Update', 'Monthly_Update'})

