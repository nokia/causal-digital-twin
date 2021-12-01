import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 
from networkx.algorithms.dag import is_directed_acyclic_graph
import random
import sklearn
import networkx as nx


def generate_DAG (n, p):
	"""
	function that generates a random (ER) DAG of size n and edge probability p 
	"""
	# create array of len n x n with random 1s and 0s according to probability p
	a = np.random.choice([0, 1], size=(n**2), p=[1-p, p])

	# transform it to matrix of size n x n
	b = np.reshape(a, (-1, n))

	# keep only upper triangle (insures graph will be DAG)
	# also insures that higher ranked edge can not cause lower ranked edge
	c = np.triu(b, k=1)

	# create a DAG from this adjacency matrix
	DAG = nx.from_numpy_matrix(c, create_using=nx.DiGraph)
	
	return DAG



def plot_DAG (graph):
	"""
	function that plots a DAG inline 
	"""
	pos = nx.spring_layout(graph)
	nx.draw_networkx_nodes(graph, pos)
	nx.draw_networkx_labels(graph, pos)
	nx.draw_networkx_edges(graph, pos, arrows=True)
	plt.show()



def print_DAG (graph):
	"""
	function that prints all DAG parameters 
	"""
	print("Nodes: " + str(graph.nodes))
	print("Edges: " + str(graph.edges))

	print("\nNodes (full view): " + str(graph.nodes.data()))
	print("\nEdges (full view): " + str(graph.edges.data()))



def get_parents (DAG):
	"""
	function that return parents of each node in a DAG 
	"""
	adj = nx.adjacency_matrix(DAG)
	matrix = adj.todense()
	for node in DAG.nodes:
		# prints position of 1s in each adj matrix column
		parents_list = np.where(matrix[:,node] == 1)[0]
		print("Parents for node " +str(node) +" : " +str(parents_list))



def get_roots (graph):
	"""
	function that returns root nodes
	"""
	DAG = graph
	adj = nx.adjacency_matrix(DAG)
	matrix = adj.todense()
	roots = np.array([])
	for node in DAG.nodes:
		parents_list = np.where(matrix[:,node] == 1)[0]
		if len(parents_list) == 0:
			roots = np.append(roots, node)
	return roots



def parametrize_DAG (graph, lam, v_self, v_prop, lag, step = 0.01):
	"""
	function that parametrizes DAG with SCM probabilities
	lam: probability that fault appears
	v_self: probability that fault persists on a node
	v_prop: probability that fault propagates to child node
	lag: time (samples) needed for a fault to propagate from parent to child
	step: probability resolution
	"""
	DAG = graph
	
	lam_values = np.arange(lam[0], lam[1] + step, step)
	v_self_values = np.arange(v_self[0], v_self[1] + step, step)
	v_prop_values = np.arange(v_prop[0], v_prop[1] + step, step)
	lag_values = np.arange(1, lag + 1, 1)
	
#	 print("lam_values: " + str(lam_values))
#	 print("v_self_values: " + str(v_self_values))
#	 print("v_prop_values: " + str(v_prop_values))
#	 print("lag_values: " + str(lag_values))
	
	for i in DAG.nodes:
		DAG.nodes[i]['lam'] = np.around(random.choice(lam_values), 3)
		DAG.nodes[i]['v_self'] = np.around(random.choice(v_self_values), 3)
		
	for (u,v) in DAG.edges:
		DAG.edges[u,v]['v_prop'] = np.around(random.choice(v_prop_values), 3)
		DAG.edges[u,v]['lag'] = random.choice(lag_values)
	
	return DAG



def get_parents_parameters (DAG):
	"""
	function that returns parameters of parents for each node 
	"""
	adj = nx.adjacency_matrix(DAG)
	matrix = adj.todense()
	for node in DAG.nodes:
		print("\nNode number " +str(node) + " lam_value: " +str(DAG.nodes[node]['lam']))
		print("Node number " +str(node) + " v_self_value: " +str(DAG.nodes[node]['v_self']))
		
		# prints position of 1s in each adj matrix column
		parents_list = np.where(matrix[:,node] == 1)[0]
		print("Parents for node " +str(node) +" : " +str(parents_list))
		for count, value in enumerate(parents_list):
			print("\tNode number " +str(parents_list[count]) + " lam_value: " +str(DAG.nodes[parents_list[count]]['lam']))
			print("\tNode number " +str(parents_list[count]) + " v_self_value: " +str(DAG.nodes[parents_list[count]]['v_self']))
			print("\tFor this edge v_prop is " + str(DAG.edges[parents_list[count],node]['v_prop']))
			print("\tFor this edge lag is " + str(DAG.edges[parents_list[count],node]['lag']))



def time_series (graph, length):
	"""
	this function builds Causal Model (SCM) and synthesizes time series of desired length
	by taking a graph (DAG) as input (structure + parameters) as depicted on Figure 2 in paper
	"""
	DAG = graph
	adj = nx.adjacency_matrix(DAG)
	matrix = adj.todense()
	df = pd.DataFrame([])
	
	for node in DAG.nodes:
		parents_list = np.where(matrix[:,node] == 1)[0] # list of parents (causes)
		
		# initialize alert with zero (system is without faults):
		currentNode = 0
		alert = np.array([])
		alert = np.append(alert, currentNode) # creation of time series for current node
		
		
		# looping
		for i in range(1, length): # value zero is already initialized
			
			# error terms from Figure 4 (exogenous variables):
			error_gate_self = np.random.choice([0, 1], replace=True, p = [1 - DAG.nodes[node]['v_self'], DAG.nodes[node]['v_self']]) # firing middle node error in SCM
			errorCurrentNode = np.random.choice([0, 1], replace=True, p = [1 - DAG.nodes[node]['lam'], DAG.nodes[node]['lam']]) # firing right node error in SCM
			
			# first endogenous variable from Figure 4:
			gate_self = currentNode & error_gate_self #  determining whether alert keeps previous value

			if len(parents_list) > 0: # if has parents
				# iterate over parents to calculate v_prop values
				v_prop_values = np.array([])
				for count, value in enumerate(parents_list):
					v_prop = DAG.edges[parents_list[count],node]['v_prop']
					lag = DAG.edges[parents_list[count],node]['lag']

					# error terms for cause-effect relation from Figure 4 (exogenous variable):
					error_gate_prop = np.random.choice([0, 1], replace=True, p = [1 - v_prop, v_prop]) # firing error for gate node in SCM

					if i < lag: # no propagation before LAGth sample
						gate_prop = 0
					else:
						gate_prop = int(df[parents_list[count]][i-lag]) & error_gate_prop

					# adding this v_prop to the list of gates    
					v_prop_values = np.append(v_prop_values, gate_prop)

				# the last endogenous variable from Figure 4:
				currentNode = gate_self | errorCurrentNode | any(v_prop_values) # probability that alert becomes active independetly from previous sample

			else: # this means it's root node
				# the second endogenous variable from Figure 4:
				currentNode = gate_self | errorCurrentNode
				
			# appending the value of current node to array (following sample of time series)
			alert = np.append(alert, currentNode) # new alert sample
		print("Simulation for node number " +str(node) +" is successful")
		
		df[node] = alert
	
	return df



def get_ground_truth (graph):
	"""
	function that returns a list of edges in a DAG
	i.e. ground truth for evaluating causal discovery techiques
	"""
	gt = np.array([])
	for (u,v) in graph.edges:
		edge = str(u) +"->" +str(v)
		#print(edge)
		gt = np.append(gt, edge)
	#print(" ")
	#print(gt)
	return gt



def return_f1(precision, recall):
	"""
	function that returns F1 score
	given precision and recall
	"""
	if precision!=0 and recall!=0:
		f1 = (2 * precision * recall ) / (precision + recall)
	else:
		f1 = 0
	return f1



def mean_degree(graph):
	"""
	function that computes mean edge degree for a DAG
	"""
	degree = 0
	for i in graph:
		degree += graph.degree(i)
		#print(graph.degree(i))

	#print("\nIn total: " +str(degree))
	mean = degree / len(graph)
	return mean



def compare(gt, out):
	"""
	function that compares two lists of edges
	ex: ground truth list and list of discovered edges
	and returns precision and recall
	"""
	correct = list(set(gt) & set(out)) # correctly identified
	addit = list(set(out) - set(correct)) # additionally identified
	if len(out) == 0:
		precision = 0
	else:
		precision = len(correct) / len(out)
	recall = len(correct) / len(gt)
	return [precision, recall]



def compare_verbose(gt, out):
	"""
	function that compares two lists of edges
	ex: ground truth list and list of discovered edges
	and returns precision and recall with explanations
	"""
	print("Number of ground truth edges: " +str(len(gt)))
	print("Number of discovered edges: " +str(len(out)))
	print(" ")
	correct = list(set(gt) & set(out))
	print("\tCorrectly discovered: " +str(len(correct)))
	addit = list(set(out) - set(correct))
	print("\tAdditionally discovered: " +str(len(addit)))
	precision = len(correct) / len(out)
	recall = len(correct) / len(gt) 
	print(" ")
	print("Precision: " +str(precision))
	print("Recall: " +str(recall))
	return [precision, recall]