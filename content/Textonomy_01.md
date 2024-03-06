---
Title: Blog Post One - Initial Thoughts (by Group "Textonomy")
Date: 2024-03-04 22:30
Category: Progress Report
Tags: Group Textonomy
---


## The Process We Come Up With Our Topic

Without a doubt, we are now in an Artificial Intelligence (AI) era. Within a few years, we went from traditional Deep Learning models to seemingly close to being near Artificial General Intelligence, and OpenAI just released Sora which amazed the world. With the increasing popularity of AI, the demand for AI-related products has also increased, including Nvidia’s high-performance advanced graphics processing units (GPUs). The advanced chips have become a critical component in AI development, powering deep learning algorithms and accelerating computational tasks in different fields, ranging from healthcare and finance to manufacturing and transportation. As a result, Nvidia’s stock reached its historical peak of USD 823.94 on 23-02-2024 from USD 492.44 on 02-01-2024, a substantial increase of 67.32%, making it an attractive investment option for those seeking exposure to the flourishing AI sector.

The recent phenomenal surge of demand in overall AI technology and Nvidia's stock price rise have inspired our group to apply text analytics and Natural Processing Language (NLP) techniques to analyze investors’ sentiment toward AI development and its influences on AI sector stocks.

Our group believes the analysis of investors' sentiment toward AI development is important for understanding market dynamics and identifying trends that can potentially impact asset allocations and trading decisions. By leveraging text analytics techniques, such as sentiment analysis and opinion mining, our group aims to extract valuable financial insights from textual data sources, including news articles and social media posts, to analyze how positive or negative sentiment towards AI development and consider it as a systematic risk of AI sector stocks.

## The Potential Ways to Achieve Our Goals

### Data Collection

For our sentiment analysis project focusing on AI discourse, we aim to collect data from sources talking about financial information regarding AI-related companies such as news websites like [Yahoo Finance](https://finance.yahoo.com/), which provides extensive financial news coverage, including AI developments, or the Financial Times. To diversify our dataset, we consider platforms like [Reddit](https://www.reddit.com/), where community discussions can offer valuable insights into public sentiment.

Our web scraping strategy involves tools such as [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for parsing HTML/XML documents, Requests for HTTP requests, and [Selenium](https://www.selenium.dev/) for dynamic content. Compliance with each platform's guidelines ensures ethical data collection. We'll organize the scraped data, including article headlines, publication dates, and texts, for preprocessing.

The preprocessing phase is crucial for cleaning and standardizing the data. We plan to remove HTML tags, special characters, and whitespace, normalize text to lowercase, and apply tokenization, stop words removal, and lemmatization using libraries like [NLTK](https://www.nltk.org/) and [spaCy](https://spacy.io/). These steps are designed to refine the dataset, making it suitable for sentiment analysis through iterative scripting and quality checks.

For pretraining data, we have searched online and found some potentially useful labeled datasets, such as [Sanders](https://github.com/zfz/twitter_corpus) (Twitter sentiment corpus), [Taborda](https://ieee-dataport.org/open-access/stock-market-tweets-data) (stock market tweets data), [financial phrase-bank](https://huggingface.co/datasets/financial_phrasebank) (financial news headings and sentiments), and [FiQA](https://sites.google.com/view/fiqa/) (financial news, microblog messages, and their sentiments). They may have different data structures and label methods, and we will preprocess them to form a large uniform dataset for training, validation, and testing.

### Model Selection & Training

Considering the limited time and computing resources, we plan to select a middle-size language model such as Bert, or FinBert (Bert with some financial knowledge pretraining). The goal is to train the model to take in texts and produce a sentiment score (e.g., 1 for very negative and 10 for very positive) with our collected pretraining datasets. Another potential way to train the model is to make use of large language models like ChatGPT and Bard. We can use the large ones as teachers producing fake training data and teach our middle-size model how to predict. In this way, we expect to get a model performing slightly worse than the large ones or on par with them. In the last stage, we will apply the trained Bert model to our collected real-life data from the Internet.

### Potential Uses of Results

With the sentiment scores, we can get a general idea of the market sentiment toward AI development. For example, we may use the weighted average score of all texts analyzed as the overall market sentiment. We are then able to predict whether the opening price of AI sector stocks will rise or fall on the next day based on the overall sentiment on that day. In addition, we intend to find the correlation between the sentiment and the price change so that we can even predict how much will the price rise or fall. If time permits, we will also design trading strategies based on the results of the model predictions.

## Expected Challenges During The Project

As now the project is in its early stages, we expect to encounter some challenges in detailed tasks. Firstly, the data sources we selected may have an anti-crawler mechanism (our project is for academic purposes only), which limits our ability to obtain large datasets. Secondly, the collected data may not be suitable for model training or prediction. Thirdly, how we weigh the scores across all texts needs to be discussed. Lastly, the correlation between the market sentiment toward AI development and the price of AI sector stocks may not be significant as there are also other factors influencing the market.