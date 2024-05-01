---
Title: Limitations of traditional NLP, Rise of GPT-based algorithms (Group "NewPotential")
Date: 2024-04-29 16:30
Category: Progress Report
Tags: Group NewPotential
---

## Introduction 

To recapitulate our project, we delve into analyzing the sentiment of gold. Through trial and error, we initially set our goal on analyzing sentiments around various assets. Our first trial involved developing an NLP model to categorize news articles about "Tesla" as either bullish or bearish. However, we noticed that our model wasn't just picking up articles directly related to "Tesla" the company; it was also gathering content that merely mentioned "Tesla" in passing, leading to collecting some irrelevant data. 

This challenge prompted us to pivot our focus. Precious metals, and gold in particular, stood out as an appropriate topic to analyze for our sentiment analysis. Unlike other stocks or bonds, which are often mentioned in various non-relevant contexts, gold presented a more straightforward subject matter for sentiment analysis. Furthermore, our decision was bolstered by the current economic climate. In times of heightened inflation and uncertainty, gold traditionally emerges as a safe asset, so-called “safe-haven.” People's inclination towards gold in these conditions made it an even more compelling subject for our analysis. 

One of the most notable challenges we encountered was the initial difficulty in scraping the most recently published articles. We discovered that the layout and format of articles could vary slightly, even within the same CNBC publication. For instance, a standard CNBC article might be structured quite differently from a CNBC article that has been selected in the web scraping process. This variability initially hindered our ability to consistently scrape the latest news. In response, we refined our code to accommodate these variations, enhancing our model to fetch timely information. This adjustment was crucial in ensuring that the articles we scraped remain up to date, so that they can capture the most relevant sentiment around gold. 

During the web scraping process in our project, the team relied on searching CNBC by the keyword “gold” in hopes of finding the sentiment of gold through news articles. However, it came to our attention that irrelevant news articles were also included, as titles of those articles had the word “gold.” Instead of talking about the commodity, they were often about credit cards or company news with “gold” in their names. This blurred the focus of our sentiment analysis as not much could be done to deal with this issue. 

Our decision to focus on gold, rather than other assets like "Apple" stocks, was not made lightly. Beyond the practical considerations around data relevance and clarity, gold's unique position in the financial market offered rich insights to mine. Unlike other assets, gold tends to retain value or even appreciate even in the fluctuating market, making it a unique asset class. Its role as a hedge asset against inflation and economic uncertainty makes the sentiment around gold a particularly intriguing subject to choose.  

## Lack granularity required to study a particular subject 

It was observed that the sentiment scores obtained were distracted by different noises. The mechanism behind NLP is simple as it gives out predefined score to words in a sentence according to information from database.  If the piece of news article was comparison between two different commodities, it will carry on giving out score based on the context, disregarding the comments on the commodity that was irrelevant to gold and possibly computed a result contrary to this project’s goal. Putting the effort into trending asset classes, such as stock of Apple, this limitation may have more significant hindering effect as it is often evaluated alongside big tech companies like Tesla. To remove such noises, Named Entity Recognition (NER) may help to some extent, as a technique within NLP that identifies and categorizes named entities present in text, such as company names, financial indicators, or relevant keywords. By implementing NER algorithms tailored to financial news, it becomes possible to extract specific entities and their associated sentiments more accurately.  

Another similar cause to weak and ineffective analysis in the project is the incapability of probing from perspective of finance. The financial domain encompasses a wide range of industry-specific jargon, acronyms, and terminology. These special terms may convey different meanings compared to daily and conventional use. Hence, relying solely on pre-existing and generic NLP packages for understanding these terms may lead to inaccuracies or incomplete comprehension. Therefore, a question has risen. How do we let the machine think in the perspective of finance as humanly as possible? Proper data processing will be key to ensure qualitative input is thrown inside NLP packages. Addressing this with Bag-of-Words (BoW), text is represented as a collection of individual words, disregarding grammar and word order. By creating a thematic corpus of financial terms and phrases, and mapping them to corresponding concepts, a custom BoW model can be developed. A tailored BoW model shall enhance the understanding of financial language, enabling more precise analysis and interpretation of financial news.  

To touch up the outcome, appropriate NLP packages shall be carefully picked and implemented, while one may also consider available packages particularly for certain subjects. For example, ```NLTK Vader``` performs well in understanding social media context and ```afinn``` is strong in blogs-like platform like Twitter. With much work and fine-tuning to be done, there will be lots of obstacles for first timers to overcome while conduting a proper text analytics, without fearing to lose the analyzing target midway.  

## The Challenge of Sarcasm and Irony 

The use of sarcasm and irony can be particularly problematic for NLP models when analyzing market news. These rhetorical devices often convey the opposite of what is literally stated, making it extremely challenging for algorithms to accurately interpret the intended sentiment. 

Consider a headline that reads: "Investors Rejoice as Housing Prices Soar to Unaffordable Levels." On the surface, the use of the word "Rejoice" suggests a positive sentiment, implying that investors are happy about the rising housing prices. However, the underlying sarcasm becomes evident when we consider the second part of the headline, which states that the prices have reached "unaffordable levels." The juxtaposition of "Rejoice" and "unaffordable levels" is a clear indication of sarcasm, where the author is actually expressing a negative sentiment about the housing market situation, despite the seemingly positive wording. 

An NLP model that solely relies on the literal interpretation of the words may incorrectly categorize this headline as conveying a positive sentiment. It would fail to recognize the sarcastic undertone and the true negative sentiment expressed by the author. 

The challenge lies in the fact that sarcasm and irony often rely on nuanced linguistic cues, such as tone, context, and implicit meaning, which can be difficult for algorithms to detect and interpret accurately. Unlike clear-cut emotional expressions, sarcasm and irony are highly contextual and may not follow a predictable set of rules, making it particularly challenging for NLP models to handle.

## Solution: Developing Specialized Sarcasm Detection Models 

Researchers have explored the use of machine learning techniques, such as deep learning, to build models specifically trained to detect sarcasm and irony in text. These models can learn to identify linguistic patterns, tone, and other cues that indicate the use of sarcasm. 

The key idea behind this approach is to develop machine learning models that are specifically trained to identify linguistic patterns, tonal cues, and other indicators of sarcastic or ironic expressions in text. 

Here's how this solution can be implemented: 

First, you would need to curate a dataset of market news articles and headlines, where human annotators have manually labeled instances of sarcasm and irony. This annotated corpus would serve as the training data for the sarcasm detection model. 

Next, you would utilize advanced natural language processing techniques, such as deep learning, to build a model that can learn to recognize the distinctive features of sarcastic and ironic language. This could involve training the model to detect subtle linguistic cues, understand contextual information, and identify patterns that signal the use of sarcasm or irony. 

Once the sarcasm detection model is trained, it would be integrated as a separate component within the overall market sentiment analysis pipeline. When analyzing new market news, this specialized model would be used to identify and flag any instances of sarcastic or ironic language. 

Finally, the output from the sarcasm detection model would be leveraged to enhance the sentiment analysis process. For example, if the model identifies sarcasm in a headline, the sentiment score could be adjusted to reflect the underlying negative sentiment, rather than the literal positive wording. 

By implementing this specialized sarcasm detection solution, we can significantly improve the accuracy and reliability of your market sentiment analysis, helping us avoid the pitfalls of misinterpreting the true sentiment expressed in market news.