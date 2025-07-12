"""
Please answer the following thought questions. These are questions that help you reflect
on the process of building this chatbot and about ethics.


We are expecting at least three complete sentences for each question for full credit.


Each question is worth 2.5 points.
"""


######################################################################################
"""
QUESTION 1 - Anthropomorphizing


Is there potential for users of your chatbot possibly anthropomorphize
(attribute human characteristics to an object) it?
What are some possible ramifications of anthropomorphizing chatbot systems?
Can you think of any ways that chatbot designers could ensure that users can easily
distinguish the chatbot responses from those of a human?
"""


Q1_your_answer = """


Yes, there is potential/places for anthropomorphism. Because our chatbot uses human speech and complete sentences, we believe that some can think
that our chatbot is human. However, we believe that it is unlikely, as our current chatbot is *very* deterministic and does not have many variations
in its responses. Thus, we think it  would quickly be recognized as a program. If we wanted to combat that, we can make our AI emulate human speech better, maybe with informal
or varied sentences and slang, but computationally the costs would be extremely high and/or we would lower the accuracy of our responses as a result (i.e. the
slang is used wrong, and the sentence response doesn't make sense). As a result, anthropomorphism in its speech (particularly slang) is not worthwhile. A potential ramification of anthropomorphizing chatbots is that, if users believe they are talking to a human, they may be more likely to give sensitive information that could be used against them.


Chatbot designers can make chatbot responses distinguishable by outright stating within the message that the content is made by a bot Otherwise, they can
program bots to follow a specific response format or style of speech/tone. Making the bot consistently respond in the same manner is often enough
of a signal for users to know they are speaking to a bot.




"""


######################################################################################


"""
QUESTION 2 - Privacy Leaks


One of the potential harms for chatbots is collecting and then subsequently leaking
(advertently or inadvertently) private information. Does your chatbot have risk of doing so?
Can you think of ways that designers of the chatbot can help to mitigate this risk?
"""


Q2_your_answer = """


Our chatbot does not collect any private information about someone's identity - but it could give insight into someone's preferences. We do "collect" information that users
give us in the form of movie titles and what they like or dislike –  this information can be used for aggressive/targeted advertising.
However, our chatbot doesn't save the user’s response over extended periods of time. As it does not collect data over different users and save it elsewhere, it is difficult for
this information to be accessed by a malicious actor (unless they are local) online. Similarly, if the user accidentally provided sensitive information in their response,
the chatbot would "wipe" itself once execution finishes, making it difficult for the sensitive information to leak.


However, a general chatbot may possibly collect sensitive information if a user provides it. To mitigate this, developers should train the bot
to automatically filter out sensitive information - either by simply deleting collected information it deems it uneccessary to finding a movie , or collecting
less information from the user by restricting the input. An example could be making the user only state the name of the movie and a yes or no if they liked it.
With less information collected, developers would be trading accuracy/performance for potential safety.




"""


######################################################################################


"""
QUESTION 3 - Classifier


When designing your chatbot, you had the choice of using a lexicon-based sentiment analysis module
or a statistical module (logistic regression). Which one did you end up choosing for your
system. Why?
"""


Q3_your_answer = """


We ended up choosing the rule-based sentiment analysis . We realized that our data and inputs follow a similar
structure - where a person states a movie name and always includes some word that describes a positive or negative emotion. With how simple the inputs are, we theorized
that our lexicon-based sentiment analysis might work better than a statistical model of analysis (which could get bogged down by extra exposition in the input).
After we ran tests with the data, we got slightly higher accuracy with lexicon-based analysis, and we decided to utilize that for our project!


"""


"""
QUESTION 4 - Refelection


You just built a frame-based dialogue system using a combination of rule-based and machine learning
approaches. Congratulations! Reflect on the advantages and disadvantages of this paradigm.
"""


Q4_your_answer = """


We think a core advantage of the paradigm is its simplicity - both for the user and the programmer (and the AI). We prompt the user to answer a small set of specific
question - like their opinions of a movie or a clarification- and thus we can focus on training our model and responses on specific movie-based recommendations. We don't need to 
consider responses that go beyond that (i.e. if they give opinions on a video game instead of a movie) as we have the chatbot ask the question again until their input is in our domain
This doubles as a disadvantage, however. The chatbot serves only one specific purpose, and as a result, could not be used outside of its niche. Another disadvantage is that it is difficult 
to scale the  chatbot and build more functionality. This is because our current bot works on a "frame-based" analysis.
If we add an option for the bot to recommend books, we would need to add a new "frame" and train an entirely new operation for the bot. This would entail a new
dataset on books, and a new trained module to link book preferences to book recommendations.
Because this workflow is similar to our current movie recommendation frame, it would be feasible to add the new functionality, but it would add more size and computation to our bot.


"""

