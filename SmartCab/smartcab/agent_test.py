import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5, gamma = 0, turn_on_optimization = False):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.number_of_resets = 0                          # making an agent aware of the number of trial re-sets
        self.epsilon_starting = epsilon                    # initial value for epsilon
        self.alpha_starting   = alpha                      # initial value for alpha - may decay over time        
        self.gamma            = gamma                      # assuming no future rewards are included in Q function, effectively discounting factor equals zero
        self.turn_on_optimization  = turn_on_optimization  # boolean flag indicating whether to turn on optimization in the code


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        self.number_of_resets += 1
        
        
        ########### 
        ## TO DO ##     # Done
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        
        
        if self.turn_on_optimization:
            #self.epsilon = 1 / (1+1/self.epsilon)
            self.epsilon = self.epsilon * 0.98
            #self.alpha = self.alpha * 0.99
            #self.alpha = self.alpha_starting * 100 / (100 + self.number_of_resets)
            self.alpha = self.alpha_starting * 10 / (10 + self.number_of_resets)
            
        else:
            self.epsilon = self.epsilon_starting - 0.05 * (self.number_of_resets - 1)
            
            
        if testing: 
            self.epsilon = 0 
            self.alpha = 0
        

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##     # Done, may want to re-visit to simpler version
        ###########
        
        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent        
        #state = None
        state = (waypoint, inputs['light'] , inputs['left'], inputs['oncoming'])
        #state = (waypoint, inputs['light'] == 'green', inputs['left'] == 'forward', inputs['oncoming'] in ('forward','right')) # 4 times smaller state space
                
        #check_direction = waypoint
        #if (waypoint == 'forward'): check_direction = 'oncoming'
        #state = (waypoint, inputs['light'], inputs[check_direction])
        
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##     # Done
        ###########
        # Calculate the maximum Q-value of all actions for a given state

        maxQ = max(self.Q[state].values())

        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##     # Done
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        
        if self.learning:
            if state not in self.Q.keys(): 
                self.Q[state] = {None: 0.0, 'left': 0.0, 'right': 0.0, 'forward': 0.0}
        
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        
        action = None
        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
                         
        if self.learning:
            # Decide whether to exploit the best action available or explore more possible actions        
            if (random.uniform(0,1) < self.epsilon):
                # explore by taking random action
                action = random.choice(self.env.valid_actions) 
            else:
                # exploit best known action(s)
                best_actions_list = []
                for possible_action in self.env.valid_actions:
                    if (max(self.Q[state].values()) == self.Q[state][possible_action]): 
                        best_actions_list.append(possible_action)                        
                action = random.choice(best_actions_list) 
        else: 
           action = random.choice(self.env.valid_actions) #take random action 
       
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        
        if self.learning:
            new_state = self.build_state()
            self.createQ(new_state)
            self.Q[state][action] = (1-self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * self.get_maxQ(new_state))
            #note gamma discount factor is set zero by default, essentially removing the last term
        
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run(learning_version = 'none'):
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    if learning_version == 'none':
        agent = env.create_agent(LearningAgent)
    if learning_version == 'default':
        agent = env.create_agent(LearningAgent, learning = True)
    if learning_version == 'optimized' or learning_version == 'test':
        #agent = env.create_agent(LearningAgent, alpha = 0.1, learning = True, turn_on_optimization = True)
        agent = env.create_agent(LearningAgent, alpha = 0.5, learning = True, turn_on_optimization = True)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name

    if learning_version == 'none':
        sim = Simulator(env, update_delay = 1)
    if learning_version == 'default':
        sim = Simulator(env, update_delay = 0.005, log_metrics=True, display = False)
    if learning_version == 'optimized':
        sim = Simulator(env, update_delay = 0.0001, log_metrics=True, optimized = True, display = False)
    if learning_version == 'test':
        sim = Simulator(env, update_delay = 0.1, log_metrics=True, optimized = True, display = True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    
    if learning_version in ('none','default'):
        sim.run(n_test = 10)
    if learning_version == 'optimized' or learning_version == 'test':
        sim.run(n_test = 10, tolerance = 0.005)

if __name__ == '__main__':

    #run(learning_version = 'none')
    #run(learning_version = 'default')
    #run(learning_version = 'optimized')
    run(learning_version = 'test')
