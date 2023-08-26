# Coefficient of Conflict: A Computational Approach to Contextually Analyzing Team Conflict 

## Motivation

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

Zeitzoff, 2017 enlists certain real-life situations (Table 1), where social media was used to foment or exacerbate conflict, and enlists different theories of conflict that can be tested in those situations. The research questions enlisted here are important questions in social science, particularly conflict research, that can be effectively studied via computational methods, considering the huge volume and dynamic nature of the data involved.

![Alt text]([https://assets.digitalocean.com/articles/alligator/boo.svg ](https://drive.google.com/file/d/1R1rJgYs9N9Eww1NQVY5EZ6KwWksnFCeg/view?usp=drive_link))

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


