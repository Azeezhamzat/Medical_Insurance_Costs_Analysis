# Performance Analysis of Supervised Learning for Medical Insurance Costs

<center>Author: Jinghui Yao, Qiaolin Liang, Yifan Wang, Ruizhi Xu, Haofei Lai, Yonghao Li</center>

### Abstract

Insurance companies devise health plans based on the insurer's personal conditions. Finding out the factors with higher impact on the insurer’s medical expenditures can help the insurance company to generate more reasonable health plans and subsequently maximize revenue. In this paper, we explore influential factors contributing to each individual’s annual medical expenditure. Using descriptive analysis and supervised learning including multiple linear regression and classification methods, we found that smokers tend to have a higher medical cost while other factors (such as region and obesity) are less impactful. Most importantly, we are able to visualize and predict the medical expenditures spent for each individual based on characteristic features, which might be of interest to insurance companies for future improvements to healthcare plans.

### Introduction

Healthcare is an act of using necessary medical procedures to improve one’s well-being, which may include surgeries, therapies and medications. These services usually are provided by a healthcare system that consists of hospitals and physicians. According to *Time*, currently the medical cost spent per person in the U.S is five times more than that in Canada, and the U.S administrative costs for healthcare came out to $812 billion in 2017. The amount spent in healthcare is significant, not only does each individual pay a lot for their insurance plans, the government has also been devoting resources to medical expenditures for a long time.

Under the healthcare law, insurance companies are not allowed to set the plan for each individual depending on their health, medical history and gender (Healthcare.gov). Therefore, we are interested in what factors contribute more to the medical costs of each individual. Our research questions include the following:

1. How much does obesity or smoking status affect medical expenditures?
2. What is the most significant factor that affects a patient’s medical expenditures?
3. Does the difference in residential regions impact medical costs?
4. Can we predict personal medical expenditures using supervised learning models?

To understand more about the dataset, we first explore the relationships between the dependent variable (medical charges) and each independent variable (number of children, smoking status, BMI, etc.) to visualize the amount of influence of each IV. In order to further explore our research questions, we then build a multiple linear regression model to better understand the underlying relationships between all independent variables and the dependent variable medical cost. We also apply classification to explore our research questions as another approach. We use individual classifiers (including random forest, SVM, naive bayes, and logistic regression) along with an ensemble model and decision tree model to thoroughly study the dataset. Additionally, we build a prediction function using the model with the highest accuracy to predict whether an individual’s medical expenditures will be higher or lower than the mean of log-scale charges.

