## README for `summarization`

### Overview:
This script focuses on processing and summarizing large articles or meeting transcripts. It consists of methods to split long articles, summarize the chunks of information, and recombine the summarized information for an overall coherent summary. It interfaces with OpenAI's GPT models to perform summarization.

### Classes and Methods:

1. **Split**
   - `split_string(string, num_segments)`: Splits a string into approximately `num_segments` based on sentence endings.
   - `split_meeting(articles)`: Returns a list of articles split into chunks so that each chunk is less than 6800 tokens.
   - `make_split_meeting_files()`: Processes the files in the `SQuALITY` directory and generates split versions of articles.

2. **Summarize**
   - `traverse_path(folder_path)`: Traverses through a directory to find files and folders to summarize.
   - `traverse_sub_path(path)`: For a given path, load the queries and articles and generate both intermediate and final summaries.
   - `intermediate_summary(query, docs)`: Generates intermediate summaries for each document based on the provided query.
   - `refine_summary(query, docs, l2h=False)`: Refines the summaries based on prior generated summaries. Supports low-to-high and high-to-low processing.
   - `final_summary(query, intermediate_outputs)`: Combines intermediate outputs into a final summary.

3. **SelectSummary**
   - `random_select(count)`: Randomly selects a given number (`count`) of indexes from a range and stores them.
   - `load_query_article(path)`: Loads query and article data based on pre-selected random indexes.

### Important Notes:

- Before running this script, ensure the OpenAI API key is provided.
- The script processes articles in the `SQuALITY` directory.
- Summaries are saved in the same directory structure under the `summary` subfolder.
- The script currently has hardcoded paths and filenames. Adjust these as necessary for your use case.

### Usage:

1. Ensure you have the required packages installed.
2. Populate the `../../keys.json` file with your OpenAI API keys.
3. Place the articles or transcripts to be summarized in the `SQuALITY` folder (or adjust paths as required).
4. Call the appropriate class methods depending on your desired operation, e.g., to split articles, call `Split.make_split_meeting_files()`.

### Dependencies:
Ensure the following Python packages are installed:
- json
- bert_score (optional based on further usage)
- tqdm (optional based on further usage)
- langchain
- OpenAI Python SDK

### Future Enhancements:
1. Make the directory structures and file paths more configurable.
2. Integrate `bert_score` and `tqdm` if they have specific functionalities tied to this script.
3. Add exception handling for better error feedback and resilience.
4. Optimize the token calculations for better accuracy.