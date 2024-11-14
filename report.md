# Report

This report is shortened to save the reader time. More details can be discussed later.

## Summary  
The deliverable includes:  
* A functional web UI to suggest queries  
* A functional API that runs around 80 ms for each request, assuming normal conditions and average-length questions  
* The "good" queries are in the top 10 returned results from this system/API  
* The algorithm runs entirely on consumer-grade CPUs, with no GPU involved.  

Some missing aspects:  
- Benchmarking the quality of queries: currently using human evaluation and a small dataset, but a more systematic approach is needed.  
- The system's launch time is long, but after that, it is instant thanks to the system being fast by design.  

## Problem Analysis  
This project focuses on achieving fast run times on the CPU. With this in mind, I assume that you would like the algorithm to run entirely on the CPU. Therefore, I have no plans to use a GPU for this project. However, if a GPU were used, it could significantly improve both the run time and the quality of the system.

From the problem definition and examples of the expected output, we can see that the ideal solution must not only understand the user's request but also the user's intent and the context surrounding the request. The example below illustrates this clearly:

Input: Create a table for top noise-canceling headphones that are not expensive  
Output: Top noise-canceling headphones under $100

In this example, the ideal algorithm must understand that we are looking for "top noise-canceling headphones" and not a "table" even though both are main nouns in this case. The importance of a word is highly context-dependent. To convert "not expensive" to "under $100," the model or algorithm must understand the context of the United States, because "$100" or "$100-$200" is considered expensive in some other countries. All five examples demonstrate that the ideal solution here is a language model (LM) that is heavily trained to understand the meaning and context of the question being asked. However, achieving this with a runtime below 100ms on a consumer CPU is very challenging. Even with the small models, we can process about 20 tokens per second, which means that 2 tokens in 100ms is insufficient for this project. The smallest LMs could meet the time constraints, but it performs poorly at predicting the next tokens or refining a phrase. We might need to train them with a specialized dataset, but there is a high risk of dismissing returns in that case. 

For the scope of this project, I have chosen to follow a less robust method with the goal of getting as close as possible to the ideal solution while still meeting the time constraints.   

## Thought Process  
From the provided examples, we can observe that the output can be modeled in two parts:  
- **Keywords**: Extracted from the original request, mostly noun phrases.  
- **Refinement**: Extra keywords that guide the system to more precise or related information, which can be nouns or adjectives.

Based on this, I propose an algorithm that can be roughly described as follows:  
1. Given an input, extract keywords from the input.  
2. Generate refined keywords from the input and the keywords extracted in step 1.  
3. Combine the keywords from the two steps above to generate the candidates for final results and rank them.

There are some challenges with this approach:  
1. Keywords represent the main idea or object from the user's input, the source of information to search for. From the example, we can see that most of the keywords are noun phrases that can be found in the input. The first solution is extracting noun phrases from sentences. However, not all noun phrases are equally important. In the above example, why is "table" less important than "headphones"? One idea to solve this problem is to use dependency parsing, which (maybe) could provide a tree structure to show that "table" is less important than "headphones." Another challenge is that "top noise canceling" depends on "headphones," and together they form a single noun phrase. From my benchmarking, I found that all dependency parsers have one or two problems: they are either too expensive or too inaccurate.

    Therefore, I decided to build my own token classification model to determine whether a token in an input has the potential to become a keyword, and I retain the keywords according to my model. The results from this model are good enough to suggest keywords from user inputs. The model is fine-tuned on my own collected data using the BERT architecture.

2. Once we have the keywords, we have the first part of the query. However, we might need a refinement part. Refinement can have 2 possibilities:  
   - **Change/delete the keywords** from step 1. The best keyword will depend heavily on the corpus, document, or user group we are serving, as each group might have its own terms for specific concepts. It is challenging to model this without a corpus for that specific domain or user group. In this project, we don’t have that information yet. Therefore, I decided to follow the approach below.  
   - **Adding extra keywords**: We assume that the keywords from the input capture the most important part of the query. The refinement part will provide extra details to help narrow down the search space. Since these details are not in the keywords extracted from the input, I model these extra details as coming from what remains of the input after removing the keywords. These are typically sub-phrases, such as "has collaborated with," for example. To generate the keywords, we need to find a word or phrase with a similar meaning. While we have synonyms at the word level, we don’t have them at the sub-phrase level. If we had time and resources, we could mine these phrases to construct a good set of sub-phrase synonyms, but we don’t have that here. To overcome this issue, I chose to model this problem as finding the most similar sub-phrase using sentence embeddings. I have a corpus of the most common words as a base, and we find the words from this corpus that have similar meanings to the sub-phrase from the input.  

3. The candidates are generated by combining the keywords from the two steps above. These candidates are then ranked against the input using a cross-encoder to find the best matches and return the top results.  

    As you can see, the quality of the output depends heavily on:  
    - The corpus of most common words/sub-phrases  
    - The quality of sentence embeddings  
    - The quality of cross-encoders for re-ranking  

    Unfortunately, I don’t have enough resources to fine-tune these models to optimize system performance, as we need to collect high-quality data that reflects the business value/usecase of this system. However, if we had enough resources, fine-tuning these models would definitely improve the system as a whole. The main reason we need high-quality data is that it is challenging to identify which sub-phrases are more related without using data to illustrate the concept of relatedness. 

## Other Aspects:

1. **Performance Improvements**:

    Applying these approaches naively will not provide good performance. Many refinements are needed to optimize both running time and the quality of outputs. I have made several optimizations to bring the running time below 100 ms as required:  
    - Implemented most operations with O(N) complexity  
    - Took advantage of ordered samples, etc.  
    - Optimized the inference time of the ML model on the CPU

    If I had enough time and resources, the next steps would be:  
    - Collect a dataset to improve the quality of sub-phrase embeddings and the cross-encoder, targeting this use case specifically.  
    - Collect common terms from internet sources (which should not be too hard).

2. **Development and Deployment Aspects**:  
    - Add guardrails depending on what is permitted: detect sensitive information, and if using an LLM, detect prompt injection by running prompt classification, etc.  
    - Track the performance of ML models after deployment to detect any degradation in performance with new data.  
    - Add a logging system.  
    - Track the performance of the API and set up alerts if issues occur after deployment.  
    - Set up CI/CD with GitHub Actions (staging/production separation).  
    - Deploy a load balancer.  

    Not all aspects of production deployment are listed here.