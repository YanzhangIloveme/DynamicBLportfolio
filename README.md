# DynamicBLportfolio
Hi all, if you are studying quantatitive finance or time series techniques, it may bring you a new idea about how to use models to build the Black-Litterman model.

Here, Assume you have already known some basic knowledge about Black-Litterman (BL) portfolio, ARIMA model and Garch model.

We know, BL model is a powerful model to take personal information into account. Practically, these infomation are based on industry reseaerches or personal insights. What if we use quantitative model such as ARIMA or AR-garch to focast the returns and use these returns as 'information'? 


I have tried this idea and use maximum return portfolio and mimimum risk portfolio as comparison. This repository also includes a complete analysis process such as EDA, LB test and JB test. Then I build the model using 80% of data and left 20% of data for back-testing. The results look good. The daily AR-GARCH corrected Black-Litterman portfolio tends to outperform other portfolios, with a higher Sharpe ratio at around 0.11308.  While, the monthly BL portfolio is not that good due to its inaccurate forecasting.

It's easy to use and see the results. Open "GrpAsgmData.mat" and " OpenHighLowPrice" (They are data) into your matlab and just run "fullcode.m". 

