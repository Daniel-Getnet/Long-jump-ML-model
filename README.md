# Long jump performance prediction using machine learning models<br>
This repository contains the code and data used in my manuscript titled “Predicting Long Jump Performance Using Biomechanical Features: A Machine Learning Approach.” The study explores how biomechanical features pridict long jump performance and utilizes machine learning approch to predict performance.<br>

###  four Model were used for this study:<br>
1. Random Forest<br>
2. CatBoost<br>
3. Gradient Boosting<br>
4. XGBoost<br>

### Feature Analysis<br>
A total of nineteen (19) biomechanical features from elite long jumpers were analyzed for importance using the Random Forest model.<br> The feature selection process identified the following as the five most influential factors in predicting long jump performance:<br>

- Gender<br>
- Horizontal Velocity during the Third-to-Last Stride Before Take-off (3rd LS HV BTO)<br>
- Horizontal Velocity during the Final Stride Before Take-off (LS HV BTO)<br>
- second last step horizontal velocity before take off (2nd LS HV BTO)<br>
- Vertical Velocity at Take-off (VV at take-off)<br>

Conversely, the least influential features were:<br>

- Take-off Angle<br>
- Flight Time During the Second-to-Last Stride (2nd LS Flight Time)<br>
- Center of Mass Lowering (CM Lowering)<br>

### Model Performance Evaluation<br>

Model performance was evaluated using the following metrics:<br>

- Mean Absolute Error (MAE)<br>
- Root Mean Squared Error (RMSE)<br>
- Coefficient of Determination (R²)<br>

Among the models, **XGBoost** demonstrated the best predictive accuracy, achieving:<br>

- **R²:** 0.9454<br>
- **MAE:** 0.1552<br>
- **RMSE:** 0.1791<br>

## Hyperparameter Tuning<br>

To improve model performance, hyperparameter tuning was performed. The final model was evaluated on unseen data, demonstrating strong predictive accuracy and further confirming its robustness.<br>

## Conclusion<br>

The finding of this study offer valuable insights into to what extent biomechanical features pridict long jump performance. moreover, The study provides practical applications for athletes and coaches seeking data-driven approaches to optimize long jump performance performance.





