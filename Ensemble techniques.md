## Ensemble techniques
Ensemble techniques are used to generate outputs with greater accuracy by stacking multiple models 
and taking out weighted mean, majority voting or training another model through obtained results. 

To clearly understand whats the need for ensemble techniques we can go back to the task 1, which was 
regarding decision trees.
| Outlook | Temperature | Humidity | Wind | Play Tennis |
|---|---|---|---|---|
| Sunny | Hot | High | Weak | Yes |
| Sunny | Hot | High | Strong | No |
| Overcast | Hot | High | Weak | Yes |
| Rain | Mild | High | Weak | Yes |
| Rain | Cool | Normal | Weak | No |
| Rain | Cool | Normal | Strong | Yes |
| Overcast | Cool | Normal | Strong | Yes |
| Sunny | Mild | High | Weak | Yes |
| Sunny | Cool | Normal | Weak | No |
| Rain | Mild | Normal | Weak | Yes |
| Sunny | Mild | Overcast | Mild | High | Yes |
| Overcast | Hot | Normal | Yes | No |
| Rain | Mild | High | Strong | No |
In it we try to decide whether the boy will play or not by analysing the 
whether. He plays everytime its overcast,if its raining, we must ask whether its windy or not,
if windy he wont play. When making a decision tree several factors we must take into consideration:
On what features do we make our decisions on? What is the threshold for classifying each question 
into yes or no answer? In the first decision tree what if we wanted to ask ourselves if we had friends 
to play with or not. If we have friends, we will play every time. If not, we might continue to ask
ourselves questions about the whether. By adding an additional question, we hope to greater define the 
yes or no classes.

This is where Ensemble Methods come in handy! Rather than just relying on one Decision Tree and hoping we made the right decision at each split, Ensemble Methods allow us to take a sample of Decision Trees into account, calculate which features to use or questions to ask at each split, and make a final predictor based on the aggregated results of the sampled Decision Trees.

# Clear Implementation
