# ITS-POCs
Intelligent Tutoring System

Blog- https://medium.com/@ashrivastava_40082/building-an-intelligent-tutoring-system-from-scratch-irt-bkt-83693fb21286

Steps to run the above code- 
1. If your dataset is in the format "Data format generally received from e-learning platforms.csv" then using "preprocessing.py" to make that data reusable and then follow from step 2 else move directly to step 2.
2. Now your dataset would appear like "data (BKT).csv", so use "Contextual Estimation of Guess Slip method (BKT).py" to find labell each response as guess and slip.
3. Now we will operate on "Demo1 (IRT).csv" which is transpose of the data in "data (BKT).csv". Use "ItemResponseTheory.py" to find the shortest learning path and your model is ready.  

For deeper understanding- 
In this new learning era along with the course structures and new teaching methodologies, we need intelligent and adaptive tutoring systems also. This brings the concept of the Intelligent Tutoring System. An intelligent tutoring system (ITS) is a computer system that aims to provide immediate and customized instruction or feedback to learners, usually without requiring intervention from a human teacher. Feedback to the learners involve the knowledge tracing of each student for each knowledge component (KC), customized instructions are the adaptivity of the system to define a Shortest Learning Path for each student to enhance the speed as well as the quality of learning and these actions happen immediately with the help of some very easy to comprehend yet so powerful Machine Learning Algorithms.

There are a lot of ML algorithms that assert to be the foundation of ITS yet the two most popular ones are the Item Response Theory (IRT) and Bayesian Knowledge Tracing (BKT). 

The item response theory (IRT), also known as the latent response theory refers to a family of mathematical models that attempt to explain the relationship between latent traits (unobservable characteristic or attribute) and their manifestations (i.e. observed outcomes, responses or performance).
Bayesian Knowledge Tracing (BKT) is an algorithm used in many intelligent tutoring systems to model each learner's mastery of the knowledge being tutored. It models student knowledge in a Hidden Markov Model as a latent variable, updated by observing the correctness of each student's interaction in which they apply the skill in question. 

So I tried to build an Intelligent Tutoring System that integrates both IRT and BKT by picking the parts at which both provide good results. IRT provides better results in adaptive systems in which subsequent actions depend on the current response. This was used to build the adaptive learning part of the system, which defines the shortest learning path. While BKT is efficient in tracing knowledge of the learner for each knowledge component so this was used to trace the student’s knowledge. Being specific 3-parameter IRT and contextual estimation of guess slip method (BKT) was used to make the model a reality.

Shortest Learning Sequence/Path (SLS) consists of a short series of lessons recommended for the students who undergo reinforcement process in order to remediate their learning difficulty.

First, we will talk about the IRT part of the system. The purpose of these models is to probabilistically explain an examinee’s responses to test items via a mathematical function based on his/her ability. The goal of IRT is to estimate the learner’s ability with regards to his/her dichotomous answers to test items. These test items are scored dichotomously: the correct answer receives a score of one, and each of the distractors yields to a score of zero. Items scored dichotomously are often referred to as binary items.
Each lesson has difficulty level based on the incorrect responses on the examination by the learner. Initially, the difficulty level is computed by taking the natural log of the odds of failure (the number of incorrect answers) given by

                                                       bi = ln((1-p)/p)                         -(i)

Equation (i) is the difficulty logit where b is the difficulty of ith lesson and p is the number of correct responses to test items. Negative logit means the lesson is easy, 0 means the lesson is moderately difficult, and positive logit means the lesson is hard.
Student’s ability is estimated based on the correct responses on the examination. Ability  is computed by taking the natural log of the odds of success (the number of correct answers) given by
					                                              𝜽𝒋 = ln(p/(1-p))                         -(ii)
The Ability Logit in (ii) is the ability of jth student and p is the number of correct responses to test items. Negative logit means the ability level is poor, 0 indicates an average ability level, and positive logit means ability level is high. 
In the One-Parameter Logistic (1PL) Model or Rasch Model, the probability of a correct response is determined by the lesson’s difficulty and the student’s ability by
					                                              𝑷(𝜽)= (𝒆^(𝛉𝐣−𝒃𝒊))/(𝟏+𝒆^(𝛉𝐣−𝒃𝒊))          -(iii)

This is one parameter IRT while what I used in the model was 3 parameter that includes ‘slip’ and ‘guess’ probabilities of each student corresponding to a particular topic. So the formula becomes (where c is ‘guess factor’ and s is ‘slip factor’)
 					                                              P(𝜽) = c + (1-c-s)/(1+e^(-a(𝜽-b)))         -(iv)

The IRT model illustrates the relationship between the learner’s answer and a test item through the Item Characteristic Curve (ICC). The standard mathematical model for the ICC is the cumulative form of the logistic function. It shows how the interaction of student ability and item difficulty influences the predicted probability of a correct response to the item.
The Item Information Function (IFF) is related to the accuracy with which ability is estimated. IIF provides information about the ability of the student depending on how closely the difficulty of the item matches the ability of the student.
                                                        𝑰𝒊(𝜽,𝒃𝒊) =𝑷𝒊(𝜽,𝒃𝒊)*𝑸𝒊(𝜽,𝒃𝒊)                   -(iv)
The item information is used to rank the lessons in recommending the shortest learning sequence where 𝐼𝑖(𝜃,𝑏𝑖) is the item information of ith lesson, 𝑃𝑖 𝜃,𝑏𝑖 is the probability of correct responses on the ith lesson, and 𝑄𝑖(𝜃,𝑏𝑖) = 1 – 𝑃𝑖(𝜃,𝑏𝑖) the probability of incorrect responses on the ith lesson. As the ability becomes either smaller or greater than the item difficulty, the item information decreases.

So the question is how should we get ‘c’ (guess factor) and ‘s’ (slip factor). Here we need contextual estimation of guess and slip method. It says that the probabilities of guess and slip of a student attempting a question are a function of external parameter like ‘Time taken’, ‘help request’, ‘number of opportunities student has already used in a current skill’ and ‘total time taken in this skill so far’. So for us to estimate these probabilities dynamically we first need to train the ML model. So we need training data that says the guess and slip probability of action. So we use Bayes theorem to label each action as guess and slip. Algorithmic implementation is shown below,

def P_L_func (correct, P_L_previous, S, G):
  	if correct==1:
    		P_L_obs = (P_L_previous*(1-S))/(P_L_previous*(1-S) + (1-P_L_previous)*(1-G))
  	else:
    		P_L_obs = (P_L_previous*S)/(P_L_previous*S + (1-P_L_previous)*(1-G))
  
  	P_L_current = P_L_obs + (1-P_L_obs)*P_T
  
  	return P_L_current

def P_C_func (P_L_previous, S, G):
  	P_C_current = P_L_previous*(1-S) + (1-P_L_previous)*G
  
  	return P_C_current

def P_S_func (P_L_previous, S, G, A1, A2):
    	if A1==1 and A2==1:
        		P(A+1+2/Ln) = (1-S)*(1-S)
        		P(A+1+2/~Ln) = P_T*(1-S)*(1-S) + (1-P_T)*P_T*G*(1-S) + (1-P_T)*(1-P_T)*G*G
    	elif A1==1 and A2==0:
        		P(A+1+2/Ln)= G*(1-S)
        		P(A+1+2/~Ln) = P_T*(1-S)*S + (1-P_T)*P_T*G*S + (1-P_T)*(1-P_T)*G*(1-G)
    	elif A1==0 and A2==1:
        		P(A+1+2/Ln) = G*(1-S)
        		P(A+1+2/~Ln) = P_T*S*(1-S) + (1-P_T)*P_T*(1-G)*(1-S) + (1-P_T)*(1-P_T)*(1-G)*G
    	else:
        		P(A+1+2/Ln) = G*G
        		P(A+1+2/~Ln) = P_T*S*S + (1-P_T)*P_T*(1-G)*S + (1-P_T)*(1-P_T)*(1-G)*(1-G)
    
    	P(A+1+2) = P_L_previous*P(A+1+2/Ln) + (1-P_L_previous)*P(A+1+2/~Ln)
    
    	return (P(A+1+2/Ln)*P_L_previous)/P(A+1+2)

def P_G_func (P_L_previous, S, G, A1, A2):
    	if A1==1 and A2==1:
        		P(A+1+2/Ln) = (1-S)*(1-S)
        		P(A+1+2/~Ln) = P_T*(1-S)*(1-S) + (1-P_T)*P_T*G*(1-S) + (1-P_T)*(1-P_T)*G*G
    	elif A1==1 and A2==0:
        		P(A+1+2/Ln)= G*(1-S)
        		P(A+1+2/~Ln) = P_T*(1-S)*S + (1-P_T)*P_T*G*S + (1-P_T)*(1-P_T)*G*(1-G)
    	elif A1==0 and A2==1:
        		P(A+1+2/Ln) = G*(1-S)
        		P(A+1+2/~Ln) = P_T*S*(1-S) + (1-P_T)*P_T*(1-G)*(1-S) + (1-P_T)*(1-P_T)*(1-G)*G
    	else:
        		P(A+1+2/Ln) = G*G
        		P(A+1+2/~Ln) = P_T*S*S + (1-P_T)*P_T*(1-G)*S + (1-P_T)*(1-P_T)*(1-G)*(1-G)
    
    	P(A+1+2) = P_L_previous*P(A+1+2/Ln) + (1-P_L_previous)*P(A+1+2/~Ln)
    
    	return 1-((P(A+1+2/Ln)*P_L_previous)/P(A+1+2))

P_L_fun: funtion returns learning probability
P_C_fun: funtion returns correctness probability
P_S_fun: function returns slip probability (depending on the next 2 actions - A+1+2)
P_G_fun: function returns guess probability (depending on the next 2 actions - A+1+2)
P(A+1+2/Ln): next 2 ations (n+1 and n+2) happened when student knew the concept at nth action
P(A+1+2/~Ln): next 2 ations (n+1 and n+2) happened when student did not know the concept at nth action

After labelling we will train the model to predict any action as guess or slip which will further be used in 3 parameter IRT to find the correctness probability and then the item information to define the shortest learning path.


About Dataset and implementation

The dataset used is for 10 students in which each student answers 6 questions of each of the 10 sub-section of a chapter. I kept a threshold of 0.75 correctness probability to consider whether a student learned a particular sub-section and thus removed from the further learning process. Since the guess and slip probabilities calculated is for every action but we need it for a total of 6 questions, so we will take a mean of guess and slip probabilities of a student for a particular sub-section.

 ![lesson difficulty and student ability](https://cdn-images-1.medium.com/max/1000/1*WvgmHoxF6BTa1D80-jbmVw.png)
 
 ![Probability of correctness](https://cdn-images-1.medium.com/max/1000/1*b_Nuy-HOVAn_WFgOr_2xWw.png)
 
 ![Shortest Learning Path](https://cdn-images-1.medium.com/max/1000/1*bak-HLFe3PQDBEWdNxZWEg.png)
 
 ![Question Type in Next Iteration](https://cdn-images-1.medium.com/max/1000/1*829VnFblv1KCU3dKWMbRBA.png)


Now since we arranged these sub-sections according to the increasing item information for each student, we need to ask questions in each section according to that, so I assumed 5 categories in which item information of each student can fall.
Range 1: 0.0<Item Information<=0.05 thus 4-easy, 1-medium and 1-hard
Range 1: 0.05<Item Information<=0.10 thus 3-easy, 2-medium and 1-hard
Range 1: 0.10<Item Information<=0.15 thus 2-easy, 2-medium and 2-hard
Range 1: 0.15<Item Information<=0.20 thus 1-easy, 3-medium and 2-hard
Range 1: 0.20<Item Information<=0.25 thus 0-easy, 3-medium and 3-hard

So the questions asked to each student will be,

So each student will continue solving questions for a particular sub-section till correctness probability reach 0.75.
GitHub repo - https://github.com/myhoggy/ITS-POCs
