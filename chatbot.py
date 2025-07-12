"""
Class that implements the chatbot for CSCI 375's Midterm Project. 

Please follow the TODOs below with your code. 
"""
import numpy as np
import argparse
import joblib
import re  
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import nltk 
from collections import defaultdict, Counter
from typing import List, Dict, Union, Tuple
from nltk.corpus import words
from nltk.corpus import brown
from nltk.probability import FreqDist

import util

class Chatbot:
    """Class that implements the chatbot for CSCI 375's Midterm Project"""

    def __init__(self):
        # The chatbot's default name is `moviebot`.
        self.name = 'H(andy)T(heater)S(earcher)' # TODO: Give your chatbot a new name.

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, self.ratings = util.load_ratings('data/ratings.txt')
        
        # Load sentiment words 
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        # Train the classifier
        self.train_logreg_sentiment_classifier()

        nltk.download('brown')
        nltk.download('words')
        self.all_words = words.words('en')

        # to handle process recursion
        self.said_greeting = False

        # where in the control flow the program is
        self.curr_state = None

        # preferences the user has given so far
        self.prefs = {}

        # most recent film the user has reviewed
        self.most_recent_movie = ""

        # sentiment of most recent movie
        self.orig_line = ""

        # potential movies we are disambiguating between
        self.candidates = []

        # number of recommendation from recommend_movies() we are giving
        self.rec_num = 0

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def intro(self) -> str:
        """
        Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """

        # TODO: delete and replace the line below
        return "Please give us 5 movies (with the titles in quotations) that you liked or disliked, and we will recommend a new movie for you to watch. To exit the bot, write ':quit' (or press Ctrl-C to force the exit). "
#         Your task is to implement the chatbot as detailed in the
#         instructions (README.md).

#         To exit: write ":quit" (or press Ctrl-C to force the exit)

#         TODO: Write the description for your own chatbot here in the `intro()` function.
       

    def greeting(self) -> str:
        """Return a message that the chatbot uses to greet the user."""
        
        ########################################################################
        # TODO: Delete the line below and replace with your own                #
        ########################################################################
        self.said_greeting = True
        greeting_message = "Welcome to the HTS Recommendation Bot! We are here to help you."

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self) -> str:
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Delete the line below and replace with your own                #
        ########################################################################
        
        goodbye_message = "Thank you for using the HTS Recommendation Bot! Have a great day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    def debug(self, line):
        """
        Returns debug information as a string for the line string from the REPL

        No need to modify this function. 
        """
        return str(line)

    ############################################################################
    # 2. Extracting and transforming                                           #
    ############################################################################

    def process(self, line: str) -> str:
        """
        Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this script.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'
        
        Arguments: 
            - line (str): a user-supplied line of text
        
        Returns: a string containing the chatbot's response to the user input

        Hints: 
            - We recommend doing this function last (after you've completed the
            helper functions below)
            - Try sketching a control-flow diagram (on paper) before you begin 
            this 
            - We highly recommend making use of the class structure and class 
            variables when dealing with the REPL loop. 
            - Feel free to make as many helper funtions as you would like 
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################

        response = ""
        
        # user wants to quit
        if line == ":quit":
            self.curr_state = "done"
            response = self.goodbye()
        
        if line == "Who are you?":
           return ""

        # user just opened chatbot
        if self.curr_state == None:
            self.curr_state = "gathering prefs"
            if self.said_greeting == False:
                response = self.greeting() + " " + self.intro()

        # user must clarify their input
        if self.curr_state == "disambiguating":
          confirmed_movies = self.disambiguate_candidates(line, self.candidates)
          if len(confirmed_movies) == 0:
             response = "That wasn't clear, try clarifying another way."
          elif len(confirmed_movies) > 1:
             self.candidates = confirmed_movies
             response = self.ask_for_clarification(line)
             return response
          else:
            self.curr_state = "gathering prefs"
            self.most_recent_movie = self.find_movies_idx_by_title(self.titles[confirmed_movies[0]][0])[0]
            response = self.process_sentiment(self.orig_line)
            response = response + self.move_forward()
            return response

        elif self.curr_state == "recommending":
           if line == "Yes":
              if self.rec_num > len(self.recommend_movies(self.prefs, 10)):
                 return "We do not have any more movies to recommend."
              response = "I recommend you watch " + self.recommend_movies(self.prefs, 10)[self.rec_num] + ". Do you want another recommendation (Yes/No)?"
              self.rec_num = self.rec_num + 1
           else:
              self.curr_state = "done"
              response = self.goodbye()
              

        # we are asking user for movies
        elif self.curr_state == "gathering prefs":
            titles = self.extract_titles(line)

            if len(titles) > 1:
                list_of_lines = self.function3(line)
                for sentence in list_of_lines:
                    response += self.process(sentence)
                return response
            
            for i in range(len(titles)):
                titles[i] = self.function2(titles[i])
            

        
          # no movie in input
            if titles == []: 
                response = "No movie detected. " + self.intro()
                return response
            
            movie_IDs = self.find_movies_idx_by_title(titles[0])
            

            if len(movie_IDs) == 0:
                response = "Movie not in our database. " + self.intro()
                return response

            # multiple movies in input
            if len(movie_IDs) > 1:
                self.candidates = movie_IDs
                self.curr_state = "disambiguating"
                self.orig_line = line
                response = self.ask_for_clarification(line)
                return response
            
            # 1 movie in input
            else:
                titles = self.extract_titles(line)
                 
                for i in range(len(titles)):
                   titles[i] = self.function2(titles[i])

                self.most_recent_movie = self.find_movies_idx_by_title(titles[0])[0]
                response = self.process_sentiment(line)
                
            response = response + self.move_forward()

        

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    def process_sentiment(self, line):
      # sentence_split = line.split()
      # for curr_word in sentence_split:
      #    if curr_word not in self.all_words:
      #       curr_word = self.function1(curr_word)
      # line = " ".join(sentence_split)
      
      sentiment = self.predict_sentiment_rule_based(line)
      
      if sentiment == 0:
        response_2 = "Sorry, I did not understand. Please retype your statement with simpler language" 
      elif sentiment == 1:
        response_2 = "Thank you for telling me you liked " + self.titles[self.most_recent_movie][0]
        self.prefs[self.most_recent_movie] = 1
        print(self.prefs)
        print(self.most_recent_movie)
      else:
        response_2 = "Thank you for telling me you did not like " + self.titles[self.most_recent_movie][0]
        self.prefs[self.most_recent_movie] = -1
        print(self.prefs)
        print(self.most_recent_movie)
      return response_2
    

    def ask_for_clarification(self, line):
      self.curr_state = "disambiguating"
      response_2 = "Which of the following are you referring to: "
      for movie in self.candidates:
        response_2 = response_2 + self.titles[movie][0] + " or "
      response_2 = response_2 + "?"
      return response_2

    def move_forward(self):
       if len(self.prefs.keys()) < 5:
        response_local = ". Please give me another movie!"
       else:
        response_local = ". I now have enough data to make a recommendation! I recommend you watch " + self.recommend_movies(self.prefs, 10)[self.rec_num] + ". Do you want another recommendation (Yes/No)?"
        self.rec_num = self.rec_num + 1
        self.curr_state = "recommending"
       return response_local
            

    def extract_titles(self, user_input: str) -> List[str]:
        """
        Extract potential movie titles from the user input.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example 1:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I do not like any movies'))
          print(potential_titles) // prints []

        Example 2:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        Example 3: 
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'There are "Two" different "Movies" here'))
          print(potential_titles) // prints ["Two", "Movies"]                              
    
        Arguments:     
            - user_input (str) : a user-supplied line of text

        Returns: 
            - (list) movie titles that are potentially in the text

        Hints: 
            - What regular expressions would be helpful here? 
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                             
        return re.findall(r'"([^"]*)"', user_input)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def find_movies_idx_by_title(self, title:str) -> List[int]:
        """ 
        Given a movie title, return a list of indices of matching movies
        The indices correspond to those in data/movies.txt.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list that contains the index of that matching movie.

        Example 1:
          ids = chatbot.find_movies_idx_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        Example 2:
          ids = chatbot.find_movies_idx_by_title('Twelve Monkeys')
          print(ids) // prints [31]

        Example 3: 
          ids = chatbot.find_movies_idx_by_title("Michael Collins (1996)")
          print(ids) // prints [800]

        Arguments:
            - title (str): the movie title 

        Returns: 
            - a list of indices of matching movies

        Hints: 
            - You should use self.titles somewhere in this function.
              It might be helpful to explore self.titles in scratch.ipynb
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
            - Our solution only takes about 7 lines. If you're using much more 
            than that try to think of a more concise approach 
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                                 
        idxs = []
        title = re.escape(title)
        for i, (t, _) in enumerate(self.titles):
            if re.search(title, t):
                idxs.append(i)  
        return idxs
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    def disambiguate_candidates(self, clarification: str, candidates: list) -> List[int]: 
        """
        Given a list of candidate movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (e.g. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If the clarification does not uniquely identify one of the movies, this 
        should return multiple elements in the list which the clarification could 
        be referring to. 

        Example 1 :
          chatbot.disambiguate_candidates("1997", [1359, 2716]) // should return [1359]

          Used in the middle of this sample dialogue 
              moviebot> 'Tell me one movie you liked.'
              user> '"Titanic"'
              moviebot> 'Which movie did you mean:  "Titanic (1997)" or "Titanic (1953)"?'
              user> "1997"
              movieboth> 'Ok. You meant "Titanic (1997)"'

        Example 2 :
          chatbot.disambiguate_candidates("1994", [274, 275, 276]) // should return [274, 276]

          Used in the middle of this sample dialogue
              moviebot> 'Tell me one movie you liked.'
              user> '"Three Colors"''
              moviebot> 'Which movie did you mean:  "Three Colors: Red (Trois couleurs: Rouge) (1994)"
                 or "Three Colors: Blue (Trois couleurs: Bleu) (1993)" 
                 or "Three Colors: White (Trzy kolory: Bialy) (1994)"?'
              user> "1994"
              movieboth> 'I'm sorry, I still don't understand.
                            Did you mean "Three Colors: Red (Trois couleurs: Rouge) (1994)" or
                            "Three Colors: White (Trzy kolory: Bialy) (1994)" '
    
        Arguments: 
            - clarification (str): user input intended to disambiguate between the given movies
            - candidates (list) : a list of movie indices

        Returns: 
            - a list of indices corresponding to the movies identified by the clarification

        Hints: 
            - You should use self.titles somewhere in this function 
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################     
        new_candidates = []
        for curr_candidate_id in candidates:
            curr_candidate_title = self.titles[curr_candidate_id][0]
            if clarification in curr_candidate_title:
                new_candidates.append(curr_candidate_id)
        return new_candidates
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    ############################################################################
    # 3. Sentiment                                                             #
    ########################################################################### 

    def predict_sentiment_rule_based(self, user_input: str) -> int:
        """
        Predict the sentiment class given a user_input

        In this function you will use a simple rule-based approach to 
        predict sentiment. 

        Use the sentiment words from data/sentiment.txt which we have already 
        loaded for you in self.sentiment. 
        
        Then count the number of tokens that are in the positive sentiment category 
        (pos_tok_count) and negative sentiment category (neg_tok_count).

        This function should return 
        -1 (negative sentiment): if neg_tok_count > pos_tok_count
        0 (neural): if neg_tok_count is equal to pos_tok_count
        +1 (postive sentiment): if neg_tok_count < pos_tok_count

        Example:
          sentiment = chatbot.predict_sentiment_rule_based('I LOVE "The Titanic"'))
          print(sentiment) // prints 1
        
        Arguments: 
            - user_input (str) : a user-supplied line of text
        Returns: 
            - (int) a numerical value (-1, 0 or 1) for the sentiment of the text

        Hints: 
            - Take a look at self.sentiment (e.g., in scratch.ipynb)
            - Remember we want the count of *tokens* not *types*
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                                  
        tokens = re.findall(r'\b(?<!")\b[a-z]+\b(?!")\b', user_input.lower()) #would avoid any quotations it sees, hopefully.
       
        neg_count = 0
        pos_count = 0
        if len(tokens) > 0:
            for token in tokens:

                # UNCOMMENT THIS LINE TO ACTIVATE SPELL CHECK
                #token = self.function1(token)

                if token in self.sentiment:
                    if self.sentiment[token] == "pos":
                        pos_count += 1
                    elif self.sentiment[token] == "neg":
                        neg_count += 1
       
        if neg_count > pos_count:
            return -1
        elif pos_count > neg_count:
            return 1
        return 0  
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def train_logreg_sentiment_classifier(self):
        """
        Trains a bag-of-words Logistic Regression classifier on the Rotten Tomatoes dataset 

        You'll have to transform the class labels (y) such that: 
            -1 inputed into sklearn corresponds to "rotten" in the dataset 
            +1 inputed into sklearn correspond to "fresh" in the dataset 
        
        To run call on the command line: 
            python3 chatbot.py --train_logreg_sentiment

        Hints: 
            - You do not need to write logistic regression from scratch (you did that in HW3). 
            Instead, look into the sklearn LogisticRegression class. We recommend using scratch.ipynb
            to get used to the syntax of sklearn.LogisticRegression on a small toy example. 
            - Review how we used CountVectorizer from sklearn in this code
                https://github.com/cs375williams/hw3-logistic-regression/blob/main/util.py#L193
            - You'll want to lowercase the texts
            - Our solution uses less than about 10 lines of code. Your solution might be a bit too complicated.
            - We achieve greater than accuracy 0.7 on the training dataset. 
        """ 
        #load training data  
        texts, y = util.load_rotten_tomatoes_dataset()

        # self.model = None #variable name that will eventually be the sklearn Logistic Regression classifier you train 
        # self.count_vectorizer = None #variable name will eventually be the CountVectorizer from sklearn 

        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                                
        
        self.model = linear_model.LogisticRegression(C=0.1, penalty='l2', class_weight='balanced', max_iter=2000) 
        self.count_vectorizer = CountVectorizer(ngram_range=(1,2), max_features=2000)
      
        converted_y = [1 if label.lower() == "fresh" else -1 for label in y ]

        lower_texts = [text.lower() for text in texts]
        vectorizer_X = self.count_vectorizer.fit_transform(lower_texts)
        converted_y = np.array(converted_y) 

       
     
        self.model.fit(vectorizer_X, converted_y)

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    def predict_sentiment_statistical(self, user_input: str) -> int: 
        """ 
        Uses a trained bag-of-words Logistic Regression classifier to classifier the sentiment

        In this function you'll also uses sklearn's CountVectorizer that has been 
        fit on the training data to get bag-of-words representation.

        Example 1:
            sentiment = chatbot.predict_sentiment_statistical('This is great!')
            print(sentiment) // prints 1 

        Example 2:
            sentiment = chatbot.predict_sentiment_statistical('This movie is the worst')
            print(sentiment) // prints -1

        Example 3:
            sentiment = chatbot.predict_sentiment_statistical('blah')
            print(sentiment) // prints 0

        Arguments: 
            - user_input (str) : a user-supplied line of text
        
        Returns: int 
            -1 if the trained classifier predicts -1 
            1 if the trained classifier predicts 1 
            0 if the input has no words in the vocabulary of CountVectorizer (a row of 0's)

        Hints: 
            - Be sure to lower-case the user input 
            - Don't forget about a case for the 0 class! 
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                             
        tokens = re.sub(r'"[^"]*"', '', user_input)
        tokens_counted = self.count_vectorizer.transform([tokens]) 
       
        if tokens_counted.nnz == 0:
          return 0
       
        return int(self.model.predict(tokens_counted)[0])
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 4. Movie Recommendation                                                  #
    ############################################################################

    def recommend_movies(self, user_ratings: dict, num_return: int = 3) -> List[str]:
        """
        This function takes user_ratings and returns a list of strings of the 
        recommended movie titles. 

        Be sure to call util.recommend() which has implemented collaborative 
        filtering for you. Collaborative filtering takes ratings from other users
        and makes a recommendation based on the small number of movies the current user has rated.  

        This function must have at least 5 ratings to make a recommendation. 

        Arguments: 
            - user_ratings (dict): 
                - keys are indices of movies 
                  (corresponding to rows in both data/movies.txt and data/ratings.txt) 
                - values are 1, 0, and -1 corresponding to positive, neutral, and 
                  negative sentiment respectively
            - num_return (optional, int): The number of movies to recommend

        Example: 
            bot_recommends = chatbot.recommend_movies({100: 1, 202: -1, 303: 1, 404:1, 505: 1})
            print(bot_recommends) // prints ['Trick or Treat (1986)', 'Dunston Checks In (1996)', 
            'Problem Child (1990)']

        Hints: 
            - You should be using self.ratings somewhere in this function 
            - It may be helpful to play around with util.recommend() in scratch.ipynb
            to make sure you know what this function is doing. 
        """ 
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                                    
        user_ratings_array = np.zeros(self.ratings.shape[0])
        for curr_movie in user_ratings.keys():
          user_ratings_array[curr_movie] = user_ratings[curr_movie]

        rec_ids = util.recommend(user_ratings_array, self.ratings, num_return)

        rec_names = []
        for movie_id in rec_ids:
            rec_names.append(self.titles[movie_id][0])
        return rec_names 
        
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 5. Open-ended                                                            #
    ############################################################################

    def function1(self, mispelled_word):
        """

        THIS MAKES EVERYTHING SLOOWWW --- refer to comment in predict_sentiment_rule_based for speeding up.
        We can iterate over the sentiment file instead of self.all_words to make things faster if needed.

        This function suggests the word the user most likely intended when they made a typo. If no word can be identified, the original word is returned.
        This is achieved by finding all words in the English language that are edit distance 1 from the word.
        The function then chooses the "most likely" word by choosing the word that occurs the most in the brown corpus.
        ex. "liek" --> "like" 
        """
        # find potential solutions
        if mispelled_word in self.all_words:
           return mispelled_word
        close_words = []
        for curr_word in self.all_words:
          if nltk.edit_distance(mispelled_word, curr_word, transpositions=True) < 2:
            close_words.append(curr_word)

        # determine how popular each solution is
        freq_dist_all_words = FreqDist(brown.words())
        freq_dist_subset = {}
        for word in close_words:
          freq_dist_subset[word] = freq_dist_all_words[word]

        # take the most popular solution (or original word if solutions exist)
        try:
          return max(freq_dist_subset, key=freq_dist_subset.get)
        except Exception:
          return mispelled_word
        

    def function2(self, title):
        """
        This function re-formats a title by removing the article 
        ex. "An American in Paris" --> "American in Paris"

        Initial approach was to move the article to the end but that 
        makes the bot do more work and makes it harder to disambiguate
        """
        # isolate index of 1st space
        if " " not in title: # no space --> 1 word title and nothing to flip
            return title
        title_first_word_len = title.index(" ")
        
        
        title_new = title
        # move article to end + add comma
        title_article = title[0:title_first_word_len]

        if title_article.lower() in ["the", "a", "an"]:
            title_new = title[title_first_word_len+1:]
        
        # find movie in database
        return title_new

    def function3(self, user_input):
        """
        Goal: Extract multipe movies in a single user input.
        How it works: Splits up the movies and the rest of the sentence. Recreates sentences based off of 
        it's position so that the sentiment carries over between lists/conjunctions.
        Current Bug: If multiple movies need to be disambiguated, it won't work with process() and disambiguate
        all of them. It would only add one (dismabigauted) movie in that instance.
        """


        conjunctions_list = ["and", ","] #can add more functions if needed


        tokens = re.findall(r'"\w[^"]*"|\w+|,', user_input) #find all elements that are OUTSIDE of quotations
        split_sentence = []
        temp_string = ""
        # print(tokens)
        for element in tokens: #loop through and "build a sentence". When detecting a conjunction, split the sentence and build a new sentence.
            if element in conjunctions_list and temp_string != "":
                split_sentence.append(temp_string)
                temp_string = ""
            else:
                temp_string += element
                temp_string += " "
        if temp_string != "":
            split_sentence.append(temp_string)


        no_movies = re.sub(r'(")([^"]*)(")', r'\1\3', user_input) # Get rid of movie titles to make a sentence structure/


        current_full_sentence = no_movies
       
        adjusted_sentence_list = []




        for split in split_sentence: # Loop through given splits. If the bot doesn't find a sentiment, we would give it a sentiment based on the other words in the sentence.
            no_movies_sentence = re.sub(r'"[^"]*"', '', split.strip())
            result = self.predict_sentiment_rule_based(no_movies_sentence)
            if result == 0:
                replacement_title = re.sub(r'\"[^"]*\"', split, current_full_sentence, count = 1)
                adjusted_sentence_list.append(replacement_title)
            else:
                adjusted_sentence_list.append(split)
                current_full_sentence = split




        # print(adjusted_sentence_list) #Debugging code to test results!
        # results = []
        # for sentence in adjusted_sentence_list:
        #     results.append(self.predict_sentiment_rule_based(sentence))
        # print(results)




        return adjusted_sentence_list
    





