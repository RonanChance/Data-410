# Research Proposal

My research goal is to find the most effective method of classifying URLs as benign or malicious. To achieve my research goal, I will use boosting methods to create strong learners from weak learners. I will expand on current research by exploring the effectiveness of Random Forest, Adaboost, XGBoost, and LightGBM. Additionally, I will examine the benefit of implementing a Synthetic Minority Oversampling Technique (SMOTE) on the imbalanced dataset.

The dataset of interest is the Malicious Urls Dataset, found on Kaggle:
https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset

It contains 651,191 URLs in total. Of these, the majority are completely safe, but the remaining URLs are either defacement, phishing, or malware.

The percentage breakdown of the URLs is as follows:
- Safe (~65.74%)
- Defacement (~14.81%)
- Phishing (~14.45%)
- Malware (~4.99%)

Since my goal is to create a binary classification of whether a link is malicious, I will remove the defacement links from the dataset, and group the phishing and malware links into one category. 

The given data file only contains two columns, one for the URL, and one for the classification. The first step of data processing will be to create uniformity in the URLs. This means removing the ‘http://’ and the ‘www’ from all of the URLs.

After completion of this step, I plan to extract feature data from these individual URLs. My current features include, but are not limited to:
- Count of each special character present (e.g. ‘%’, ‘$’, ‘#’, ‘@’, …)
- Total length of URL
- Ratio of letters to special characters
- Ratio of capitalized to uncapitalized letters
- Contains a URL shortening service
