import numpy as np
import random

points ='ABCDE'

paths = {
		'A':{
				'B':4,'C':4,'D':7,'E':3
		},
		'B':{
				'A':4,'C':2,'D':3,'E':5
		},
		'C':{
				'A':4,'B':2,'D':2,'E':3
		},
		'D':{
				'A':7,'B':3,'C':2,'E':6
		},
		'E':{
				'A':3,'B':5,'C':3,'D':6
		}
}

def get_path_cost(a,b):
	''' Returns the cost of route a -> b'''
	return paths[a][b]


def eval_sample(sample):
	''' Evalutates sample given, based on cost of routes '''
	cost = 0
	for i in range(len(sample)-1):
		cost += get_path_cost(sample[i], sample[i+1])
	return cost


def crossover(parent_a):
	''' Creates new genome by using part of his parent's genome while shuffling the remaining '''
	split = np.random.randint(len(parent_a)-1)
	return parent_a[:split]+''.join(random.sample(parent_a[split:],len(parent_a[split:])))
    


def mutate(s, rate):
	''' Mutates (swaps 2 random points) of given genome s, with rate - rate '''
	if random.uniform(0, 1) <= rate:
		i,j = random.sample(range(len(s)), 2)
		lst = list(s)
		lst[i], lst[j] = lst[j], lst[i]
		new_s = ''.join(lst)
		return new_s

	else :
		return s

def check_end(lst, threshold = 5):
	''' checks if the last #threshold elements of list are identical. Used for early stopping '''
	return len(set(lst[-threshold:])) == 1

def run_sim(epochs, pop_len, elitism_no, points, paths, mut_rate, early_stopping=False):
	''' Runs a simulations with the hyperparameters given ''' 

	# creates population
	population = [''.join(random.sample(points,len(points))) for _ in range (pop_len)]

	# inits scores list for early stopping 
	scores = [0]
	for _ in range (epochs):
		child_population = []
		ratings = []

		# evaluate old population
		for sample in population:
				ratings.append((sample, eval_sample(sample)))

		# sort ratings list of tuples based on rating
		ratings.sort(key=lambda tup: tup[1])
		best_rating = ratings[0][1]
		scores.append(best_rating)

		# checks for early stopping clause
		if check_end(scores) and early_stopping:
			break

		# keep the top #elitism_no number of elements as is
		# these elements will be used to create offspring
		population = ratings[:elitism_no]

		for _ in range(len(ratings)-len(population)):
				# pick a random parent from population
				a = random.choice(population)
				child_population.append(crossover(a[0]))

		new_pop = []

		# mutate and insert offsprings into new_pop list 
		for child in child_population:
			new_pop.append(mutate(child, mut_rate))
		
		# cleanse parents and add them to the new_pop
		for val in population:
			new_pop.append(val[0])

		# create the new population and repeat
		population = new_pop

	# return best score
	return scores[-1]


def run_multiple_sims(no_of_sims, epochs, pop_len, elitism_no, points, paths, mut_rate, early_stopping=True):
	''' Runs multiple sims for a specific set of hyperparameters '''

	sim_scores = [] # used for keeping track of best scores

	for _ in range (no_of_sims):
		sim_scores.append(run_sim(epochs, pop_len, elitism_no, points, paths, mut_rate, early_stopping=early_stopping))

	return no_of_sims - sim_scores.count(11) # return the number of non optimal sims
def find_best_params(no_of_sims=20):
	''' Finds the best set of parameters for a specific range of each crucial parameter 
			(population size, elitism number and mutation rate)'''

	best_non_opt = 100

	for pop_len in range(5,15):
		for elitism_no in range(1,3):
			for mutate in range (1,7):

				hyperparams = {
					'no_of_sims': no_of_sims,
					'epochs': 20,
					'pop_len': pop_len,
					'elitism_no': elitism_no,
					'points': points,
					'paths': paths,
					'mut_rate': mutate*0.1
				}
				non_opt = run_multiple_sims(**hyperparams)
				if non_opt < best_non_opt:
					best_non_opt = non_opt
					params = {}
					params = {
						'best_pop_len':pop_len,
						'best_elitism':elitism_no,
						'best_mutate':mutate*0.1
					}
	print ('best hyperparams are', params, 'with ', no_of_sims-best_non_opt,'/', no_of_sims, 'optimal sims')

if __name__ == '__main__':


	#### uncomment once to run chosen instance ####

	#### 1 simualtion with specific hyperparams  ####

	# hyperparams = {
	# 				'epochs': 20,
	# 				'pop_len': 7,
	# 				'elitism_no': 1,
	# 				'points': points,
	# 				'paths': paths,
	# 				'mut_rate': 0.3
	# 			}
	#run_sim(**hyperparams)

	#### multiple simulations for given hyperparams ####

	# hyperparams = {
	# 				'no_of_sims': 20,
	# 				'epochs': 20,
	# 				'pop_len': 7,
	# 				'elitism_no': 1,
	# 				'points': points,
	# 				'paths': paths,
	# 				'mut_rate': 0.3
	# 			}

	#run_multiple_sims(**hyperparams)

	#### find the best set of hyperparams  ####
	find_best_params()
