---
Title: Review & Reflection Part 1 - ESG News and Stock Volatility (by Group "ESG & Volatility")
Date: 2024-04-30 19:12
Category: Progress Report
Authors: Group ESG & Volatility, Seobin, Lindsey, Hongyu
Tags: Group ESG & Volatility
---

For this blog post, we will provide a cumulative review and reflection on our project, 'ESG & Volatility,' specifically on the Data Collection and Text Preprocessing and Cleaning portion of our project. 

## Data Collection:

Our project aims to explore how stock price volatility in a company reacts to positive and negative Environmental, Social, and Governance (ESG) news. Specifically, we want to assess whether stock prices are more sensitive to positive or negative ESG sentiments. To achieve this, we utilized Natural Language Processing (NLP) and text analytics to analyze the relationship between ESG news sentiment and stock price movements.

Initially, selecting a focus company for our research was challenging. We reviewed over 20 companies listed on the NASDAQ (i.e., MSFT, AAPL) and S&P 500 (i.e., AMZN, GOOGL) examining their recent positive and negative events. Ultimately, we chose Starbucks Corp (SBUX) because it demonstrated a clear distinction between positive and negative ESG-related events, making it an ideal candidate for our study.

We dedicated a significant portion of our project timeline to selecting a focus company, which caused us to accelerate the remaining tasks to meet our final deadline. Additionally, as beginner programmers, we spent considerable time learning to code while progressing through our project.

Once we selected our focus company, we began compiling URLs of news articles related to positive and negative ESG events to scrape text data for sentiment analysis using Javascript and Python. This phase presented several challenges and required multiple iterations. Initially, we used BeautifulSoup to filter through the URLs, but it scraped all text from the news websites, not just the news content. As a result, we attempted to combine Selenium with BeautifulSoup to target only the news text but encountered numerous errors due to difficulties in integrating these two packages. Ultimately, after consulting with our professor, we decided to switch to Newspaper3k, a package specifically designed for scraping news text data, which proved more suitable for our project. 

```javascript
var links = document.getElementsByTagName('a');
var linksArr = [];

for (var i = 0; i < links.length; i++) {
  if (links[i].ping && 
     !links[i].href.includes('google')) {
     linksArr.push('<p>' + links[i].href + '</p>');
  }
}

var newWindowContent = linksArr.join('');
var newWindow = window.open();
newWindow.document.write(newWindowContent);
newWindow.document.close();
```

If we were to restart this process, it would be more beneficial to select news sources with a consistent format rather than using various outlets with different formats. Opting for articles from a single database, such as Factiva, would streamline the data collection and web scraping stages of our project. Additionally, during our initial attempt, we encountered several URLs with firewalls or subscription requirements that hindered our scraping efforts, as the packages we used were unable to bypass these barriers. A consistent format and source would also mitigate these issues, allowing for smoother data collection. 

Moreover, considering the large number of news article URLs we handled, it would have been more efficient to save these URLs in a CSV file. Parsing through the URLs from the CSV file would have allowed for a more organized and condensed overview, facilitating easier management and access during our project. However, we did manage to save the extracted details from each positive and negative ESG event into CSV files. This step was crucial for data preprocessing, allowing us to organize and analyze the information more effectively.

Below is a snippet of our code used to handle the large number of news article URLs and extract details from each positive ESG event. We applied a similar process for the negative ESG events, effectively managing and analyzing data from both types of events:

```python
pip install newspaper3k pandas

from newspaper import Article
import csv
import pandas as pd

urls = [ '...' ]

def extract_details(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            'link': url,
            'title': article.title,
            'source': article.source_url,
            'date': article.publish_date.strftime('%Y-%m-%d') if article.publish_date else 'No Date Found',
            'text': article.text
        }
    except Exception as e:
        print(f"Failed to process {url}: {str(e)}")
        return None


# List to store results
news_results = []


# Process each URL
for url in urls:
    result = extract_details(url)
    if result:
        news_results.append(result)


# Save the results to a CSV file
with open("positive_news_data.csv", "w", newline='', encoding='utf-8') as csv_file:
    fieldnames = ["link", "title", "source", "date", "text"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(news_results)


print("Data saved to positive_news_data.csv")
```

## Text Preprocessing and Cleaning:

After we scraped the news articles, we moved on to text preprocessing. For text cleaning, we converted all text into lowercase and removed unnecessary non-word characters like punctuations in order to remove noise. Initially, we removed all punctuation marks; however, we discovered that this approach also removes the dash, combining the words that were separated by dashes into a single word. For example, 'company-operated' was combined into one word 'companyoperated' when the dash was removed, which has the potential to be misinterpreted when we run sentiment analysis. Therefore, we adjusted our code to remove all punctuation marks except for the dash as well as the exclamation mark since exclamation marks could convey sentiment. Below is the code snippet of our adjusted code:

```python
# Define the clean_text() function
def clean_text(text):
   cleaned_text = re.sub(r'[^a-zA-Z\s!-]', '', text)
   cleaned_text = cleaned_text.strip()
   return cleaned_text.lower()
```
Yet, we still faced some limitations of the outcome and one example is that 'U.S.' was converted to 'us' when we removed the punctuations. This could completely change the meaning of the word and hence, limit the reliability of our results in subsequent stages of sentiment analysis. We also acknowledged some other limitations in our outcomes in further steps of text preprocessing. For instance, when we attempted stemming, we realized that stemming is not always perfect and can remove word components, leaving behind stems that are grammatically incorrect. Overall, our attempts in text preprocessing have taught us about the complexities involved in preparing for language analysis and how difficult it is to clean the text data perfectly, especially when dealing with a substantial volume of text data that cannot be manually cleaned individually. We also learned the importance of finding a balance between completeness and practicality in text preprocessing. While it is important to aim for the best possible data cleaning, we should also consider the feasibility and efficiency of the data cleaning methods being used. 

