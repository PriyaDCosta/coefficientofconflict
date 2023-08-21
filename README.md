# Coefficient of Conflict: A Computational Approach to Contextually Analyzing Team Conflict 

## Motivation

Communication forms the bedrock of human interaction, and conversations form an integral part of communication. Conversations have been studied by social scientists much before advancements in technology. Many theoretically important features that relate to the topic being discussed, and usually, social scientists ask (1) a human to rate the topic manually; (2) ask people what they talked about; or (3) require people to talk about limited topics from a list, in order to study topics in conversation. The inherent nature of social sciences poses problems such as difficulty in defining concepts, vast variations in concepts by context and difficulty in reproducibility, which necessitates quantitative or verifiable qualitative analysis. (Smith, 1997) 

Work in computational social science has attempted to computationally measure previously-qualitative measures, such as Receptiveness (Yeomans et al.), Forward Flow (Gray et al.) , Discursive Diversity (Lix et al.) and Certainty (Rocklage et al.) . The techniques to achieve quantification include lexical word counts (Certainty) and comparison of embeddings (Receptiveness , Free Flow, Discursive Diversity). However, qualitative and subjective methods like surveys and observations still continue to be widely used.


In our current work on teams , we have identified conflict as an area where there have not been attempts to develop computational measurements.


### Why does understanding conflict matter?

Conflict, a natural occurrence in interpersonal dynamics, carries the potential for both constructive growth and destructive consequences.
Conflict is the cause of up to 50 percent of voluntary and up to 90 percent involuntary departures (excluding departures due to downsizing, mergers, and restructuring). Theft and damages costs 2 percent to a company of the staff total cost. It was estimated in a certain study that replacing an engineer costs the company 150% of the departing employee's total annual compensation—the combination of salary and benefits. Analysts arrived at this figure by accounting for lost productivity, recruiting fees, interviewing time, staffing department employees' salaries, and orientation and training costs. (Dana, 2001) 


Poorly understood and unresolved conflict entails high costs and consequences to an organization and its stakeholders. Watson & Hoffman (1996) anticipate that  managers spent nearly 42% of their time on informal negotiations. Unresolved conflict can be a major cause of employee absenteeism, affecting productivity. (The Blackwell Encyclopedia of Sociology,2007).  
It is clear that conflict, if unresolved, can have serious consequences.


### Why should we measure conflict contextually (i.e using topic modeling) and computationally?

González-Romá and Hernández (2014) emphasize that effective conflict management can lead to improved team cohesion and performance. Addressing conflicts through open communication and resolution strategies can enhance trust and mutual respect, ultimately leading to a harmonious work environment. In a world increasingly driven by technology, where team interactions are often mediated through digital platforms, a manager cannot always be present to gauge and understand conflict amongst their team. This necessitates a need to measure conflict computationally, in order to better understand and manage teams. By quantifying conflict levels, we gain the ability to track conflict dynamics over time, and correlate them to specific contexts, enabling effective conflict resolution.

De Dreu (2006) suggests that different types of conflict, such as process, relational, and task conflicts, have unique implications. Process conflict, arising from disagreements about how to execute tasks, can spur critical evaluation of methodologies and enhance decision-making. Relational conflict, centered on interpersonal issues, necessitates sensitivity and empathy to preserve healthy working relationships. Task conflict, pertaining to differing opinions on goals, can promote deeper understanding and alignment among team members. In each of these, the topic being discussed will vary. For example, consider a team of software engineers. Intuitively, process conflict would include topics like the coding language, the method for testing the code, timelines for the project etc. Relation conflict would include topics around their designations and experience. Task conflict would include topics around the problem they are trying to solve through the coding exercise. Therefore, a topic-centric approach will help to computationally identify the different types of conflicts .

### Industry Perspective

Social media platforms have become pivotal in human lives. Understanding team dynamics takes on a new level of importance. It is common for groups of users to also represent themselves as teams on social media. For example, many political leaders, affiliated to a certain party, may also maintain their personal social media accounts. 

The landscape of social media is rife with potential for miscommunication and misunderstandings, often exacerbated by the absence of non-verbal cues typical in face-to-face interactions. Computational analysis of team conflict can help decode underlying tensions and misinterpretations in textual communication, flagging potential conflicts for intervention, before they can exacerbate into violent situations. This aids in maintaining a positive user experience and preventing public disagreements that may tarnish the platform's reputation.

Using a topic-based approach can help social media companies identify potential sources or reasons of discord, whether they arise from differences in approaches, personal dynamics, or cultural nuances. This insight enables these companies to preemptively address conflicts as well as safeguard themselves from possible legal issues.
Our Current Work

In our work on team process mapping, we have 5 datasets, each of which contain a number of conversations amongst teams who are supposed to perform a particular task. The nature of the task varies according to the dataset, for example in the Juries dataset (Hu et al., 2021), the task on hand is adjudication, whereas in the CSOP dataset (Almaatouq et al., 2021), the task on hand is optimization within constraints. Naturally, each dataset has a specific metric to determine performance. The performance metric for the Juries dataset is % in Majority, while it is Efficiency (Score / Task Duration) for CSOP.

We have built a Python-based framework to computationally derive topics spoken at different points of time in a conversation.
Developing the Framework

##### Step 1: Extracting Top Topics (For each dataset)
We use two approaches (1) Bottom Up Approach (Unguided Topic Modeling), where the base model i.e. BERTopic (Grootendorst, 2022) is used to generate topics without any prompting and (2) Top-Down Approach (Guided Topic Modeling), where we prompt the BERTopic model to generate topics based on our domain knowledge.

##### Step 2: Chunking
We chunk the conversation into equal units of time based on timestamps. The number of chunks are user-defined. In case of datasets which do not capture timestamps, we chunk the conversation in such a way that each chunk 

##### Step 3 : Creating and Comparing Embeddings
We generate BERT embeddings, which capture context, for each topic and also for each chunk of each conversation in the dataset. We compute the cosine similarity between each topic and each conversation embedding for each chunk.

##### Step 4 : Grouping Conversations and Bootstrapping Confidence Intervals 
We normalize this metric and label a conversation (task) as “below average performance” or “above average performance”. We then bootstrap confidence intervals for the cosine similarity for each topic-chunk-label.

### Proposed Methodology

##### Step 1: Initial creation of measure at the chat level.
Identify an initial dataset from our existing datasets to run this on. For each concept we chose in (e.g., "task conflict"), generate a custom word list using our domain knowledge of the datasets (refer previous bullet). Using the Topic modeling framework, create BERT Embeddings for each topic. 

We expect an output in the format as follows:
| Message                                                  | Task Conflict Score | Process Conflict Score | Relation Conflict Score |
|----------------------------------------------------------|---------------------|------------------------|-------------------------|
| "I think we should structure the presentation like this. Your approach seems off" | 0.0                 | 0.9                    | 0.0                     |
| "I love your presentation"                              | 0.0                 | 0.0                    | 0.0                     |
| "I'm trying my best, but it seems like you're not listening to my ideas either." | 0.0                 | 0.0                    | 0.8                     |
| “I like Pizza”                                         | 0.0                 | 0.0                    | 0.0                     |
| “I think we should start by designing the experiment. Data analysis does not seem right at this time." | 0.85                | 0.0                    | 0.0                     |
| “We all agree on starting with data analysis”            | 0.0                 | 0.0                    | 0.0                     |


##### Step 2: Validate with human labels.
###### Option 1: Validation Using MTurker

Generate a rating pipeline where we give MTurkers chats --> ask them to rate (process conflict versus relation conflict versus task conflict.) Get the correlation between our measures and human measures

** RISK:** Human raters are really noisy, and this may fail even if our measure is good.

###### Option 2: Validation by Running New Experiments
Run experiments of our own where we force people to talk about planning/transition at a specific time ("In the first 2 minutes, you are only allowed to talk about ....") or prevent them from planning (e.g., "you are not allowed to plan, you must discuss ONLY the task itself without talking about how you do it") See if the metric is able to detect that we gave them this instruction

** RISK:**  expensive, time-consuming, as we have to run brand-new studies. Also, people might not listen to our instructions, and they may discuss something else, which adds noise.

###### Option 3: Validation by Running New Experiments
"Synthetic" data related to process/relation/task conflict. Ask an LLM to generate some data and see if our model can detect whether it is process/relation/task conflict  or not
RISK: not realistic and LLMs are subject to hallucinations. 

##### Step 3: Show that the validated measurement is also useful.
Take our datasets --> Run the actual feature --> Run the models with ONLY the new features i.e. conflict features inside them.
Run the new feature alongside our features and see whether it is more predictive than other stuff


Plot the feature over time and show how it captures known patterns in how people execute/talk during projects.


