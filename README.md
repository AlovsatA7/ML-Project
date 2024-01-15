# Economic Background

Bitcoin, the leading cryptocurrency, has been a subject of intense attention. Its decentralized nature and high volatility make it an intriguing asset for traders, economists, and data scientists as well. Unlike traditional currencies, Bitcoin operates independently of a central bank and is instead governed by complex algorithms and cryptography. This digital currency's price is influenced by a lot of factors, including market demand, investor sentiment, regulatory changes, and broader economic indicators. As a result, predicting its price movements presents a unique challenge ripe for advanced machine learning techniques.

Bitcoin has captured the attention of the financial world with its unprecedented growth and the significant fluctuations in its value. As a digital currency independent of central financial institutions, Bitcoin's market value is subject to rapid changes influenced by factors such as market trends, investor confidence, global economic events, and technological advancements. Understanding and predicting Bitcoin's price movements is a complex task that calls for innovative approaches.

# Goal of the Project

This project aims to predict future Bitcoin prices by employing machine learning models that can handle the intricate 
patterns seen in historical price data. Specifically, we will predict the logarithmic returns of Bitcoin prices, providing 
a nuanced view of its potential future valuation. Accurate predictions would be of immense value to investors and could 
serve as a benchmark for further financial modeling in the cryptocurrency domain.

# Extra Key Information

The models utilized in this project—Decision Trees, Random Forest, and Support Vector Regression (SVR)—have been
selected for their ability to capture nonlinear patterns and interactions between features. A wide array of features 
derived from Bitcoin's historical trading data are used, such as:

- Logarithmic Returns (log_return)
- Various Moving Averages (WMA_7, EMA_20, etc.)
- Volatility Indices (7_day_volatility, 30_day_volatility)
- Technical Indicators (RSI, MACD, Stoch_k, Stoch_d)
- Cyclical Time Features (day_sin, day_cos)
 
Forecasting Bitcoin's price is not only of academic interest but also of practical significance for investors and financial analysts. The project aims to contribute to the growing field of financial machine learning, providing insights that could enhance trading strategies and risk management.

The key evaluation metrics selected for this analysis are:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Coefficient of Determination (R^2)

These metrics were chosen for their ability to measure forecast accuracy in units of price returns.

The data was splitted on:

- taining data (01.01.2017-31.12.2019)
- testing data (01.01.2020-31.05.2020)
- validation data (01.06.2020-31.12.2020)

# Project Structure
`````
├── README.md
├── data
│   ├── input
│   ├── models_output
│   └── output
├── models
├── notebooks
│   ├── data_preparation.ipynb
│   ├── feature-engineering.ipynb
│   ├── decision_tree_model.ipynb
│   ├── random_forest_model.ipynb
│   ├── svr_model.ipynb
`````


# Observations

After running each of the model we can conclude that:

###  Decision Tree

The initial model results states that the model achieved perfect scores on the **training dataset** with MSE, MAE, and R^2 being 0, 0, and 1 respectively. Which indicates that the model has perfectly fit the data, however there is also possibility of overfitting. 
On the **testing dataset**, the model exhibited high MSE and MAE values (approximately 79695.75 and 198.29 respectively) but a relatively high R^2 value of 0.9529, suggesting that while the model captures the variance in the data well, it also makes large errors in some predictions.
The **validation dataset** showed similar results to the testing dataset with an MSE of approximately 66850.84, MAE of 175.65, and R^2 of 0.9564.
The model used a variety of features including various moving averages, volatility measures, Bollinger Bands, MACD, ATR, Stochastic Oscillator values, and lag features among others. 

#### Cross-validation Scores: 

Cross-validation was performed with negative MSE scores ranging from approximately -1.30 to -0.06, and the mean CV score for MSE was approximately 0.82. This indicates that the model's performance varied significantly across different folds, suggesting variability in the model's predictive performance.

#### Hyperparameter Tuning:

The model was tuned with a maximum depth of 10 and a minimum sample leaf of 4, which likely aimed to prevent overfitting. The resultant model's performance on the validation dataset had an R^2 of approximately 0.9474, showing a good fit.

#### Summary


The **Decision Tree** model seems to overfit the training data but still performs reasonably well on unseen data, capturing a significant proportion of the variance. The high MSE and MAE values on the testing and validation datasets suggest that while the model's general direction of predictions is correct, there are instances where it makes large errors. This may indicate that the model's complexity needs further tuning.

# Random Forest 

Similar to Decision Tree model, The Random Forest regressor achieved an almost perfect fit on the **training dataset** with very low MSE and MAE, and an R^2 score close to 1. This indicates that the model was overfitted with the high probability. 
On the **testing dataset**, the model showed good performance with an MSE of 0.0074, MAE of 0.0365, and an R^2 score of approximately 0.986, indicating that the model has generalized well to unseen data.
The **validation dataset** showed slightly less performance compared to the testing dataset with an MSE of 0.0329, MAE of 0.0501, and an R^2 score of about 0.960.
The model identified Closing_Price_Diff (the feautre that was added for stationarity) as the most important feature, followed by other features like Bollinger Bands, 24h Low , and various lagged features. This suggests that the difference in closing price is the most significant predictor of future returns, according to the model.


#### Cross-validation Scores: 

Cross-validation resulted in a mean MSE of 0.3720 and a standard deviation of 0.3954, indicating some variability in model performance across different folds.

#### Hyperparameter Tuning:

Hyperparameter tuning identified that a max depth of 20, min samples leaf of 4, and n_estimators of 300 were the best parameters for the random forest model. Also the RFECV process was applied to select the most relevant features, potentially improving model robustness by removing irrelevant features.

After final model evaluation and tuning, the random forest showed strong performance on the test set with MSE and MAE values being low and an R^2 score indicating that the model explains a high percentage of the variance in the data. However, there is a slight decrease in performance when moving from the test to the validation set, as indicated by higher MSE and MAE values and a lower R^2 score on the validation set.

#### Cross-validation Scores after Hyperparameter Tuning

The cross-validated MAE, MSE, and R^2 scores provide a more generalized view of the model's performance across different subsets of the data. The high cross-validated R^2 score of approximately 0.921 indicates the model's robustness, despite variability in individual folds.

#### Summary


**The Random forest** regressor has demonstrated strong predictive capabilities in forecasting logarithmic returns of Bitcoin prices. It shows good generalization from the training set to unseen data, but there is room for improvement in its performance on the validation set. The feature importance results provide valuable insights into the drivers of Bitcoin's price movements, with price difference being a key factor.

The variability in cross-validation scores suggests that the model's performance could be sensitive to the specific subsets of the data used for training and validation. However, the model's ability to capture a significant amount of variance in the test data suggests it could be a useful tool for predicting Bitcoin prices.


# SVR

The SVR model initially achieved an MSE of approximately 0.0242, an MAE of 0.0569, and an R^2 of 0.9758 on the **training data**. These metrics indicate an excellent fit to the training data, capturing a significant amount of variance.
On the **testing dataset**, the model performed great with an MSE of 0.0063, MAE of 0.0619, and an R^2 score of 0.988.
In the **validation set**, the model also demonstrated strong performance, albeit slightly lower than the testing set, with an MSE of 0.0494, MAE of 0.0839, and an R^2 of 0.9393.

#### Hyperparameter Tuning:
The grid search identified the optimal hyperparameters for the SVR model: C=10, epsilon=0.01, and the rbf kernel. These parameters were used to train the final model, which suggests a balance between model complexity (C) and the allowance for errors within the epsilon margin.

With the final hyperparameters, the SVR model achieved an MSE of approximately 0.2327, an MAE of 0.2857, and an R^2 of 0.7673 on the **training dataset**. This indicates a reasonable fit to the training data, though not as perfect as the initial fit before hyperparameter tuning, which is actually a good sign as it suggests less overfitting.
On the **testing dataset**, the final SVR model showed an MSE of 0.0286, MAE of 0.1324, and an R^2 score of 0.9451.
The **validation dataset** results were slightly less favorable with an MSE of 0.0505, MAE of 0.1293, and an R^2 of 0.9380. However, these metrics indicate that the model maintains a good performance level even on the validation set, which is critical for the model's use in a real-world setting.

#### Cross-validation Scores: 
The cross-validated MSE was reported to be 0.3836, with an R^2 of 0.5790. These metrics are not as strong as those obtained on the single test set, suggesting that the model's performance may vary depending on the specific subset of the data it's trained on.

#### Summary

**The SVR model** with the best parameters from hyperparameter tuning demonstrates good predictive ability, as shown by the testing and validation performance. While there is some variance in performance across different data subsets indicated by the cross-validation results, the model is generally robust and performs well on unseen data.
The relatively high R^2 scores on the test and validation sets suggest that the SVR model can capture a significant proportion of the variance in the logarithmic returns of Bitcoin prices. However, the less consistent cross-validation results imply that the model could benefit from further validation across different time periods or market conditions to assess its robustness.



# Simple Linear Regression Model

**The Linear Regression model** showed a reasonable fit to the **training data** without signs of overfitting, unlike the Decision Tree model which showed a perfect fit indicating potential overfitting. 
On the **testing data** with high R^2 values, slightly outperforming the Random Forest and SVR models in terms of R^2, but with higher MSE and MAE compared to Random Forest.
On the **validation dataset** it maintained a high R^2 score, similar to its testing performance, passing the Decision Tree and being comparable to the Random Forest and SVR models.

- Training Performance: MSE: 0.2076, MAE: 0.3031, R^2: 0.7924
- Testing Performance: MSE: 0.0492, MAE: 0.1818, R^2: 0.9056
- Validation Performance: MSE: 0.0836, MAE: 0.1762, R^2: 0.8974

# Conclusion

### Decision Tree Model Performance:

- Training Performance: Achieved perfect scores with MSE and MAE at 0, and R^2 at 1, indicating a complete fit to the training data.
- Testing Performance: MSE 79695.75, MAE 198.30, R^2 0.9529.
 - Validation Performance: MSE 66850.84, MAE 175.65, R^2 0.9564.

#### Random Forest Model Performance:

- Training Performance: MSE (0.0056) MAE (0.0306), R^2 (0.9944).
- Testing Performance: MSE 0.0136, MAE 0.0614, R^2 0.9739.
- Validation Performance: MSE 0.0603, MAE 0.0728, R^2 0.9259.

#### SVR Model Performance:

- Training Performance: MSE 0.2327, MAE 0.2857, and R^2 0.7673.
- Testing Performance: MSE 0.0286, MAE 0.1324, R^2 0.9451.
- Validation Performance: MSE 0.0505, MAE 0.1293, R^2 of 0.9380.



**Overfitting** - The Decision Tree model showed signs of overfitting the training data, given its perfect training scores and lower performance on the test and validation sets. In contrast, the Random Forest and SVR models demonstrated a better balance between training performance and generalization to unseen data, with SVR showing the most significant reduction in potential overfitting after hyperparameter tuning.

**Generalization** - Both the Random Forest and SVR models generalized well to unseen data, with Random Forest having a slight edge in terms of lower MSE and higher R^2 scores on the test data. However, the SVR model showed better generalization in the validation set compared to the Random Forest model.

**Predictive Performance** - The Random Forest model had the best overall predictive performance among the three, as indicated by its consistent MSE, MAE, and R^2 scores across the test and validation datasets. The SVR model, after tuning, also performed well but with a slight decrease in performance metrics compared to Random Forest.

In conclusion, while all three models have their strengths, the Random Forest model stands out for its ability to generalize well to new data while maintaining strong predictive performance, making it the most suitable model for predicting Bitcoin's logarithmic returns.
However, the Linear Regression model's strong performance, especially in terms of R^2, makes it a valuable and simpler alternative for predictions 


