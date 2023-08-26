# Coefficient of Conflict: A Computational Approach to Contextually Analyzing Team Conflict 

## Motivation

### Why Does Understanding Conflict Matter?

Conflict is an awareness on the part of the parties involved of discrepancies, incompatible wishes, or irreconcilable desires (Boulding, 1963). Conflict is a frequent occurrence in interpersonal dynamics, and it carries the potential for both destructive consequences and constructive growth.
Conflict is the cause of up to 50% of voluntary and up to 90 % of involuntary departures (excluding departures due to downsizing, mergers, and restructuring) (Dana, 2001). It was estimated in a certain study that replacing an engineer costs the company 150% of the departing employee's total annual compensation—the combination of salary and benefits(Dana, 2001) .

Poorly understood and unresolved conflict entails high costs and consequences to an organization and its stakeholders. Watson & Hoffman (1996) anticipate that  managers spent nearly 42% of their time on informal negotiations. Unresolved conflict can be a major cause of employee absenteeism, affecting productivity. (The Blackwell Encyclopedia of Sociology,2007).
On the flipside, conflict can also stimulate constructive growth. Conflict challenges conventional thinking and encourages the exploration of novel solutions. Diverse perspectives can lead to innovative ideas that drive the team forward (Amason et al., 1997). Actively managing task conflict is beneficial to team performance (DeChurch and Marks, 2001). 


It is clear that conflict, if unresolved, can have serious consequences, while conflict, if managed well, can create value for teams and organizations at large. A deeper and possibly contextual understanding of conflict, computed with minimal human intervention to ensure robustness and low bias can help in both these scenarios. 
De Dreu (2006) suggests that different types of conflict, such as process, relational, and task conflicts, have unique implications. Process conflict, arising from disagreements about how to execute tasks, can spur critical evaluation of methodologies and enhance decision-making. Relational conflict, centered on interpersonal issues, necessitates sensitivity and empathy to preserve healthy working relationships. Task conflict, pertaining to differing opinions on goals, can promote deeper understanding and alignment among team members. In each of these, the topic being discussed will vary. For example, consider a team of software engineers. Intuitively, process conflict would include topics like the coding language, the method for testing the code, timelines for the project etc. Relation conflict would include topics around their designations and experience. Task conflict would include topics around the problem they are trying to solve through the coding exercise. Therefore, a topic-centric approach will help to computationally identify the different types of conflicts.

### Proposed Method: Computationally Quantifying Conflict

Previous ways of measuring conflict have relied on self-report measures. Popular works on understanding team conflict, such as Jehn and Mannix, (2001) , DeChurch and Marks ( 2001) and De Dreu (2006) use self-reported surveys to gauge the level of conflict amongst a team.The inherent nature of social sciences poses problems such as difficulty in defining concepts, vast variations in concepts by context and difficulty in reproducibility, which necessitates quantitative or verifiable qualitative analysis. (Smith, 1997) 

Work in computational social science has attempted to computationally measure previously-qualitative measures, such as Receptiveness (Yeomans et al., 2020), Forward Flow (Gray et al.,2019) , Discursive Diversity (Lix et al.,2020) and Certainty (Rocklage et al.) . The techniques to achieve quantification include lexical word counts (Certainty) and comparison of embeddings (Free Flow and Discursive Diversity). The Receptiveness feature is measured using an existing computational model Politeness (Danescu-Niculescu-Mizil et al. ,2013). However, qualitative and subjective methods like surveys and observations still continue to be widely used. We aim to change this paradigm. 


### Applications

González-Romá and Hernández (2014) emphasize that effective conflict management can lead to improved team cohesion and performance. Addressing conflicts through open communication and resolution strategies can enhance trust and mutual respect, ultimately leading to a harmonious work environment. In a world increasingly driven by technology, where team interactions are often mediated through digital platforms, a manager cannot always be present to gauge and understand conflict amongst their team. This necessitates a need to measure conflict computationally, in order to better understand and manage teams. By quantifying conflict levels, we gain the ability to track conflict dynamics over time, and correlate them to specific contexts, enabling effective conflict resolution.

In addition to advancing the science of team conflict, our work would also have additional applications in an industry setting. Social media is a potential application of a computational system for measuring conflict. Interactions in social media are usually public and conflicts typically involve the masses (Gebauer et al. 2013) The landscape of social media is rife with potential for miscommunication and misunderstandings, often exacerbated by the absence of non-verbal cues typical in face-to-face interactions. Conflict greatly influences value formation on social media; it can create as well as destroy value. (Husemann et al. 2015) 

Trying to understand social media without considering the underlying media dynamics and incentives is problematic (Zeitzoff, 2017). Existing methods for content moderation, such as hashtag moderation, provides limited understanding on the origin of problematic content (Gerrard, 2018). Sibai et al., 2017 make an attempt to use machine learning tools to identify and classify different posts based on the types of conflict, however they do not compute a conflict score or measure conflict in contextually. Computational and contextual analysis of team conflict can help decode underlying tensions and misinterpretations in textual communication, flagging potential conflicts for intervention, before they can exacerbate into violent situations. 

Zeitzoff, 2017 enlists certain real-life situations (shown below), where social media was used to foment or exacerbate conflict, and enlists different theories of conflict that can be tested in those situations. The research questions enlisted here are important questions in social science, particularly conflict research, that can be effectively studied via computational methods, considering the huge volume and dynamic nature of the data involved.

![Alt text](https://github.com/PriyaDCosta/coefficientofconflict/blob/d4a2e1be0b9ef1a2465f5fe789cf48bddd9cd9c6/graphs/Screen%20Shot%202023-08-25%20at%206.56.03%20PM.png)

## Proposed Research 

In our ongoing research on team communication processes, we have been studying how features of team conversations differ across tasks. Our work involves building a Python-based conversational analysis framework that generates quantifiable features from conversations of real teams. These features are then used to identify how, across different tasks, teams use distinct conversational strategies to succeed.

We apply our conversational analysis framework to chat-based datasets from real, online team interactions. The nature of the task varies in each dataset. Our dataset currently consists of ~2600 chats across 5 datasets, each with a different type of task. Each dataset has a specific metric to determine performance that differs based on the task, which we use for building across-task models.

As part of this research, we have also engaged in early-stage exploratory data analysis, examining the prevalence of different topics over the course of a conversation. Below, we provide an overview of our existing work; we then proceed to outlining how this work can be extended to examining conflict through a computational lens.

### Existing Progress: Exploring Topics Over Time

Measuring conflict computationally first requires an understanding of the crux of a team’s conversation. Intuitively, one would expect the crux of the conversation to be around agreeing, affirming, supporting etc. in a low/no conflict scenario, and around disagreement in a conflicted scenario. A conversation is not static, and the crux of the conversation evolves over time, which is why we perform our analysis on time-based “chunks” of the conversation, rather than on the conversation as a whole. Temporally slicing the crux of the conversation helps to identify stages in the conversation. For example, if the crux of the conversation is exchanging pleasantries, sharing designations and disagreeing, it is likely that it is the introduction stage and conflict here may be relational conflict (Jehn and Mannix, 2001) , i.e. conflict about the relation between two people in a team. 

Accordingly, we explored developing the framework “Topics Over Time”, using the BERTopic model published by Grootendorst, 2021. The objective of this framework is to slice a conversation into logical “chunks” of time, and compare how important different topics are to that chunk of the conversation, to obtain the crux of each chunk. 

This comparison is done by first obtaining the top topics for each task i.e. each dataset. We follow two approaches here: (1) The bottom-up approach, whereby BERTopic itself suggests topics, and (2) The top-down approach, whereby humans with domain knowledge about the dataset suggest topics and guide BERTopic on topic modeling. We then chunk each conversation in the dataset into logical chunks of time, and obtain BERT Vector embeddings for each conversation-chunk and each topic. These vectors are compared using cosine similarity, giving us a “similarity score” of how each topic relates to each chunk of the conversation. The figure below helps to visualize our approach:

![Alt text](https://github.com/PriyaDCosta/coefficientofconflict/blob/3aa752629ba966600e94d9a97577be994d434f06/graphs/Screen%20Shot%202023-08-24%20at%208.17.58%20PM.png)

We normalized the performance metric for each dataset, and grouped the conversations i.e. teams into below and above performance categories to identify if there exist any peculiarities in what high performing and low performing teams talk about. We also bootstrapped confidence intervals to give a range of the similarity scores for new datasets with similar tasks

![Alt text](https://github.com/PriyaDCosta/coefficientofconflict/blob/3aa752629ba966600e94d9a97577be994d434f06/graphs/Screen%20Shot%202023-08-24%20at%202.45.28%20PM.png)

The above table contains an example from the Juries dataset (Hu et al., 2021) where the task on hand is determining whether a certain person should learn to speak English. The image above highlights the different topics, and their similarity scores at different chunks of time in the conversation. The results shown above can be tied back to a sample conversation from the dataset:

![Alt text](https://github.com/PriyaDCosta/coefficientofconflict/blob/3aa752629ba966600e94d9a97577be994d434f06/graphs/Screen%20Shot%202023-08-24%20at%202.45.41%20PM.png)

*Results presented  are using the top-down approach, which gives more intuitive topics. 

### Research Questions
Our key research goal is to derive a computational conflict coefficient (conflict score). Accordingly, we seek to answer the following research questions:

#1 Can team conflict be automatically detected through text-based features?
#2 What features of communication predict different kinds of conflict (e.g., task, process, and relational conflict)?
#3 Do computationally-derived conflict features correspond to human intuitions, or perceptions, of conflict?
#4 Do computationally-derived conflict features predict other team outcomes (e.g., viability, performance)?

## Proposed Methodology
Our current work lays a foundation for determining how different topics relate to a conversation over time. Our research questions seek to understand conflict can be contextually and computationally measured through text, and it can be used to predict team performance. Accordingly, we propose the following methodology to build on our existing framework to answer our research questions:

### Step 1: Initial creation of measure at the chat level.
Identify an initial dataset from our existing datasets to run this on. For each concept, we chose (e.g., "task conflict"), we will generate a custom word list using our domain knowledge of the datasets. Using the “Topics Over Time” Framework”, we will calculate a similarity score for the embeddings of each concept and the embeddings of each chat, to determine the conflict type and conflict score.We expect an output in the format as follows:

![Alt text](https://github.com/PriyaDCosta/coefficientofconflict/blob/3aa752629ba966600e94d9a97577be994d434f06/graphs/Screen%20Shot%202023-08-26%20at%202.06.49%20PM.png)

### Step 2: Validate with human labels.
This is an important step in answering Research Question #3 :“Do computationally-derived conflict features correspond to human intuitions, or perceptions, of conflict?”. We consider the following options:

#### Option 1: Validation Using MTurker
Generate a rating pipeline where we give MTurkers chats --> ask them to rate (process conflict versus relation conflict versus task conflict.) Get the correlation between our measures and human measures

RISK: Human raters are really noisy, and this may fail even if our measure is good.

#### Option 2: Validation by Running New Experiments
Run experiments of our own where we force people to talk about planning/transition at a specific time ("In the first 2 minutes, you are only allowed to talk about ....") or prevent them from planning (e.g., "you are not allowed to plan, you must discuss ONLY the task itself without talking about how you do it") See if the metric is able to detect that we gave them this instruction

RISK: Expensive, time-consuming, as we have to run brand-new studies. We might need an IRB. People might not listen to our instructions, and they may discuss something else, which adds noise.

### Step 3: Show that the validated measurement is also useful.
This step is relevant to Research Question #4: "Do computationally-derived conflict features predict other team outcomes (e.g., viability, performance)?". It ties back to our objectives of the team process mapping project, which led to the inception of this project. Once a conflict score is computed, we will explore causal relationships between team performance and computationally measured conflict. For example, intuitively, we may think that high levels of relationship conflict predict low performance. This can be validated using the conflict score and the performance metric for the respective task.

We will  first run the models with ONLY the new features i.e. conflict features inside them. We will then run the new feature alongside our existing features in team process mapping (i.e. politeness, assertiveness, hedging etc.) and see whether it is more predictive.
Another method We also have a dataset of "team viability" --- where the dependent variable is a survey of whether the team is willing to work with each other again, and not their objective performance. Intuitively, teams with high levels of conflict will not want to work together again.


