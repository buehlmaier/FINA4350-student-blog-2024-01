---
Title: Blog Post Two - Encounter, Analyze, and Solve (by Group "Textonomy")
Date: 2024-04-30 22:30
Category: Reflection Report
Tags: Group Textonomy
---


As we are all new to NLP and related techniques, we believe sharing the challenges we met along the way is valuable. We hope these experiences can serve as a reference for future NLP learners.

## Data Collection and Preprocessing (by Zepa)

In the data processing part for pre-training data, our group has adopted data from 3 sources, namely **Financial phrasebank** (financial news headings and sentiments), **Sanders** (Twitter sentiment corpus), and **Taborda** (stock market tweets data). 

One challenge here is that our group needs to select accurate, relevant, and useful data, and it will be best if those data are well-packed, cleaned, and tidied up already. Our group spent a certain amount of time selecting high-quality data input to ensure potentially high-quality output.

Another little challenge here is that there may be an extra space at the last line of each individual processed sentence due to the usage of `\n`, which is undesired, so our group modified the previous code of `code_taborada.py` and utilized the `if-else` statement to specifically make it excluded. Below is an example from `code_taborada.py`.

![code snippet in code_taborada.py]({static}/images/Textonomy_02_code-snippet.png)

## Model Training (By Bosco)

In this part, I will illustrate the challenges I face when training our BERT model with reference to the ["Classify text with BERT"](https://www.tensorflow.org/text/tutorials/classify_text_with_bert) tutorial on TensorFlow.

The most challenging part is to read the dataset in python, which was collected and preprocessed by Zepa. Zepa combined the three datasets and transformed them into one consistent format, which consists of some lists of `[sentence, label]`. The dataset is then exported as a `.txt` file for later use. My job is to read the text file in Python and use the data to train the BERT model. Below is the code I used to read the lines:
```python
with open('combined_result.txt', 'r') as file:
    for line in file:
        # Remove leading and trailing whitespace and newline character
        line = line.strip()
        # Remove the square brackets
        line = line.strip("[]")
        # Find the index of the last comma in the line
        last_comma_index = line.rindex(',')
        sentence = line[:last_comma_index].strip()
        label = int(line[last_comma_index + 1:].strip())
        data.append([sentence, label])
```
The code assumes that each line in the text file is in the form of `[sentence, label]`. It then proceeds to find the position of the last comma. Everything before the last comma is assigned to the sentence variable, and the integer after the last comma represents the label. One can easily extract the  i<sup>th</sup> sentence by `data[i][0]` and the  i<sup>th</sup> label by `data[i][1]`. The code functions effectively when the dataset is in a consistent format. I test the code with just one dataset and it works well. However, when I test it with the combined dataset, it is not the case. After some eyeball checking, it may be due to some occasional gaps within the list that are caused by some bugs on Zepa's part. To resolve this issue, I tried deleting the gaps manually, but the sample size is large, so that this method is quite inefficient.

![example of special cases]({static}/images/Textonomy_02_special-cases.png)

Finally, I came up with a solution, which is to add a line of code: `match = re.match(r'^(.+?),\s*(\d+)$', line)`, which can validate and extract the relevant information from each line. 

- `^`: Anchors the pattern to the start of the string.
- `(.+?)`: Matches and captures one or more characters (except a newline) lazily. The `+` indicates one or more occurrences, and the `?` makes the matching lazy, meaning it captures as few characters as possible.
- `,`: Matches a comma character.
- `\s*`: Matches zero or more whitespace characters. The `\s` represents any whitespace character (spaces, tabs, etc.), and the `*` indicates zero or more occurrences.
- `(\d+)`: Matches and captures one or more digits. The `\d` represents any digit character (0-9), and the `+` indicates one or more occurrences.
- `$`: Anchors the pattern to the end of the string.

The `re.match()` function is used to check whether the line matches the specified pattern. If and only if a match is found, the code proceeds to extract the sentence and label. As a result, any lines with anomalies are ignored. However, two new issues have arisen. 

Firstly, there are some lines that match the specified pattern, but they do not actually represent the desired sample data and have been mistakenly included in the dataset. Fortunately, there are only a few lines with this issue, and it can be resolved manually.

![example of special cases 2]({static}/images/Textonomy_02_special-cases-2.png)

Another issue is that the code can only read the lines after the gap in those gapped lists, as the lines before the gap do not match the expected format. There are numerous lines affected by this problem, and due to time constraints, I have decided to retain them in the dataset. I believe that including this noise data might potentially improve the training process. 

Nonetheless, if time permits, we aim to address the issue at its root cause, which is the data preprocessing stage. Our goal is to ensure the dataset is consistent and properly formatted, thereby eliminating the need to handle inconsistencies during the training process.

## Web Scraping and Result Analysis (By Marcel, Wenkai, and Wanqian)

In order to get current text data to use for our sentiment analysis, we decided to scrape the [Yahoo Finance search site](https://news.search.yahoo.com/search?p=search_example). For that, we used a set of AI-related keywords.

A few challenges arose during web scraping development:
- (presumably) temporary IP blocking
- duplicate data
- imprecise datetime values in text format.

We started testing our scraper by running the scraping functions in a for loop for each keyword, trying to maximize AI-related news data. We noticed that the csv files we put the date in were alternating between having no data whatsoever and having (expected) hundreds to thousands of news entries. The pattern suggested a temporary inability to access the search site which we assumed to be due to temporary IP blocking. Our solution to this problem was to implement a timer that waits after each scraping iteration by keyword before starting the next iteration, which solved this issue. Considerations for a more sophisticated solution in the future are using proxy servers or rotating IPs and adjusting the timer dynamically based on the server’s response (minimizing the timer duration).

As all keywords are related to each other (because we were using AI-related keywords) and because we were scraping every day to get the newest data, naturally, we expected duplicate data. To solve this we developed scripts that would parse the data of the different keywords and combine it into a combined file while eliminating duplicates, as well as add the newest distinct data to these files every day. 

Lastly, the search site did not return actual datetime values for the post date of the articles but returned text values such as “3 days ago”, “45 minutes ago”, and “1 week ago”. The higher the aggregation level of this time information, the more imprecise a conversation to a datetime value was. Example: If we have a data entry saying the post was released 45 minutes ago, we can parse it into a datetime value that is `now() - 45 minutes`, meaning the exact value would only be off by seconds. 
However, for values where the date posted was “2 weeks ago”, `now() - 2 weeks` could potentially be off by days, making these data points less helpful.

When using the scraped data for analysis, the datetime imprecision problem got worse. The target AI index we analyzed was a US index, and we needed to further convert the time into Eastern Time. Therefore, we cannot guarantee that the news was published on that day. For example, we scraped data at 12:30 P.M. on April 20th and one piece of news showed “one day ago.” We then assumed the news was published at 12:30 P.M. on April 19th in Hong Kong time, and therefore, 12:30 A.M. on April 19th in  Eastern Time. But the news could be published at night on April 18th in Eastern Time.

Despite the time imprecision, we also encountered data imbalance. The Yahoo Finance website tends to return more news published recently and less old news. Therefore, the sentiment scores may be less accurate for long ago dates due to inadequate data, while it takes more computing resources to produce sentiment scores for recent dates with thousands of pieces of news. However, this problem will diminish in the long run as we will collect more and more data.