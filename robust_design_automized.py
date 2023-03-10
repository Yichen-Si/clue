import sys, string, itertools, copy, time, argparse
from mip import *
import numpy as np
import scipy.special

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--path", type=str, help="Prefix of the output", default="./")
parser.add_argument("-N", "--indiv", type=int, help="Number of individuals", default=64)
parser.add_argument("-C", "--conditions", type=int, help="Number of experimental conditions", default=6 )
parser.add_argument("-P", "--pools", type=int, help="Fixed number of pools" )
parser.add_argument("-m", "--min_dist", type=int, help="Robust parameter: minimum number of pairwise distance between individual profiles", default=2 )
parser.add_argument("--Gurobi", action='store_true', help="Is the Gurobi solver set up in the system?")
parser.add_argument("-t", "--max_threads", type=int, help="Number of threads to use", default=-1 )
parser.add_argument("--max_seconds", type=int, help="Maximum seconds to run each ILP", default=60 )
args = parser.parse_args()


path = args.path
I = args.indiv
C = args.conditions
m = args.min_dist
max_threads = args.max_threads
max_sec = args.max_seconds
solver_name = "CBC"
if args.Gurobi:
	solver_name = "GRB"

if m % 2 == 1: # pairwise difference is always even
	m += 1
if C >= I:
	sys.exit("If C (number of conditions) >= N (number of donors), >= N pools are needed anyways.")

# Represent experimental conditions (used for output)
code_length = 1
letters = list(string.ascii_letters)
ccode = letters
while C > len(ccode):
	code_length += 1
	ccode = [ ''.join(list(x)) for x in itertools.combinations(letters, code_length) ]


### Helper function

# Output binary representation of all b choose k
def for_all_point_nr(k, b):
	# { (y_1...y_P) \in {0,1}^P | y_1 + ... + y_P == C, y_i >= 0 }
	def dfs(now, i, r):
		if r == 0:
			yield now
		elif i < b:
			for x in range( min(2, r+1) ):
				now[i] = x
				for tmp in dfs(now, i + 1, r - x):
					yield tmp
			now[i] = 0
	for x in dfs([0] * b, 0, k):
		yield x

# Build a graph where each vertex is a pooling profile (potential donor)
# and each edge represents a pair of profiles that cannot co-exist
def build_graph(k,b,c):
	Vset = []
	id = 0
	for v in for_all_point_nr(k,b):
		Vset.append([id, 0, v.copy(),[]])
		id += 1
	print("Number of notes %d" % id)
	for i in range(1, id):
		for j in range(i):
			dis = len([x for x in range(b) if Vset[i][2][x] != Vset[j][2][x] ])
			if dis < c: #
				Vset[i][1] += 1; # Number of neighbors to avoid
				Vset[i][3].append(j) # Neighbors' id (not safe for co-occurence)
				Vset[j][1] += 1;
				Vset[j][3].append(i)
	return Vset


### Feasible assignments

if args.pools is None: # Decide the minimum P
	b = C + 1
	tot = int(scipy.special.comb(b, C))
	while tot < I:
		b += 1
		tot = int(scipy.special.comb(b, C))
	P = b
	candidates = [v.copy() for v in for_all_point_nr(C, P)]
	if m > 2:
		while True:
			Vset = build_graph(C,P,m)
			N = len(Vset)
			ilp = Model(solver_name=solver_name)
			# ilp.verbose = 0
			ilp.threads = max_threads
			x = [ ilp.add_var(var_type=BINARY) for i in range(N) ]
			for i in range(N):
				for j in Vset[i][3]:
					if j < i:
						ilp += xsum( [x[i], x[j] ] ) <= 1
			ilp.objective = maximize(xsum( x[i] for i in range(N) ) )
			status = ilp.optimize(max_seconds = max_sec)
			n_sol = ilp.num_solutions
			candidates = [Vset[i][2] for i,x in enumerate(ilp.vars) if x.xi(0) == 1. ]
			N = len(candidates)
			print(f"Try with {P} pools, {N} feasible solutions (need at least {I})")
			if N >= I:
				break
			else:
				P += 1
	print(f"Decided to use {P} pools")
else: # Check if the number of pool is attainable
	P = args.pools
	tot = int(scipy.special.comb(P, C))
	candidates = [v.copy() for v in for_all_point_nr(C, P)]
	if tot < I:
		sys.exit("No identifiable scheme with the given (N,P,C). Consider increase P or let the program decide")
	if m > 2: # Check if the robustness requirement is attainable
		while True:
			Vset = build_graph(C,P,m)
			N = len(Vset)
			ilp = Model(solver_name=solver_name)
			ilp.verbose = 0
			ilp.threads = max_threads
			x = [ ilp.add_var(var_type=BINARY) for i in range(N) ]
			for i in range(N):
				for j in Vset[i][3]:
					if j < i:
						ilp += xsum( [x[i], x[j] ] ) <= 1
			ilp.objective = maximize(xsum( x[i] for i in range(N) ) )
			status = ilp.optimize(max_seconds = max_sec)
			n_sol = ilp.num_solutions
			candidates = [Vset[i][2] for i,x in enumerate(ilp.vars) if x.xi(0) == 1. ]
			N = len(candidates)
			if N >= I:
				break
			else:
				print("The desired robust parameter is not feasible. Reduced it by 2")
				m -= 2

# Any subset of these candidates works, but we could optimize further
candidates = [np.array(x, dtype=int) for x in candidates]
N = len(candidates) # Candidate


### Optimize

# Partition the m-independent set to (m+1)-independent ones
keepindx = set()
keep = []
n_sol = 1
while n_sol > 0 and len(keepindx) < I:
	ilp = Model(solver_name=solver_name)
	ilp.verbose = 0
	ilp.threads = max_threads
	x = [ ilp.add_var(var_type=BINARY) for i in range(N) ]
	for i in range(1,N):
		for j in range(i):
			if np.abs(candidates[i] - candidates[j]).sum() == m:
				ilp += xsum( [x[i], x[j] ] ) <= 1
	for i in keepindx:
		ilp += xsum( [x[i]] ) == 0
	ilp.objective = maximize(xsum( x[i] for i in range(N) ) )
	status = ilp.optimize(max_seconds = max_sec)
	n_sol = ilp.num_solutions
	new = set([i for i,x in enumerate(ilp.vars) if x.xi(0) == 1. ])
	keep.append(new)
	print(n_sol, len(new))
	if len(new) == 0:
		break
	keepindx = keepindx.union(new)

# Balance pool size while maximize robustness
ct = [len(x) for x in keep]
tot = 0
it = 0
while tot < I:
	tot += ct[it]
	it += 1
it -= 1
arr = np.array(candidates)
overlap = np.dot(arr, arr.transpose())
np.fill_diagonal(overlap, 0)
penalty = np.zeros_like(overlap)
penalty[overlap == m] = 1
S = (I * C) // P # Min pools size
s = S
while True:
	ilp = Model(solver_name=solver_name)
	ilp.verbose = 0
	ilp.threads = max_threads
	x = [[ilp.add_var(var_type=BINARY) for j in range(i+1)] for i in range(N) ]
	ilp += xsum( x[i][0] for i in range(N) ) == I
	for i in range(P):
		ilp += xsum(candidates[j][i] * x[j][0] for j in range(N)) >= s
	for i in range(1, N):
		for j in range(i):
			ilp += xsum( [ x[i][j+1], - x[i][0] ] ) <= 0
			ilp += xsum( [ x[i][j+1], - x[j][0] ] ) <= 0
			ilp += xsum( [ -x[i][j+1], x[j][0], x[i][0] ] ) <= 1
	for j in range(it):
		for i in keep[j]:
			ilp += xsum( [x[i][0] ] ) == 1
	ilp.objective = minimize(xsum( penalty[i,j] * x[i][j+1] for i in range(1,N) for j in range(i) ) )
	status = ilp.optimize(max_seconds = max_sec)
	if status != OptimizationStatus.INFEASIBLE:
		break
	else:
		s -= 1
indicator = [0] * N
for i in range(N):
	if i > 0:
		indx = sum( [ len([j for j in range(k+1)]) for k in range (i) ] )
	else:
		indx = 0
	indicator[i] = int(ilp.vars[ indx ].xi(0))
print(sum(indicator))


mat = [candidates[i] for i,v in enumerate(indicator) if v == 1 ]
outline = '\n'.join( [ ''.join([str(x) for x in y]) for y in np.transpose(np.array(mat)) ] )
print(outline)

n = len(mat)
arr = np.array(mat)
overlap = np.dot(arr, arr.transpose())
ovec = overlap[np.triu_indices(I, 1)].reshape(-1,1)
print( "Max pairwise overlap: ",  np.max(ovec) )
tab = [0] * C
tot = 0
for i in range(C):
	ct = len([ x for x in ovec if x == i ])
	tab[i] = ct
	tot += i * ct

print("spectrum of pairwise overlaps:")
out = ["%d:%d" % (i,x) for i,x in enumerate(tab)]
spectrum = '\t'.join(out)
print(spectrum)
print("average overlap: ", tot/len(ovec))






### Try to assign experimental conditions to minimize batch effect

upper =  int(np.floor(S / C)) + 1
lower =  int(np.floor(S / C))

while True:
	ilp = Model(solver_name=solver_name)
	ilp.verbose = 0
	ilp.threads = max_threads
	x = [[[ilp.add_var(var_type=BINARY) for c in range(C)] for i in range(I) ] for p in range(P)]
	for p in range(P):
		ilp += xsum( mat[i][p] * x[p][i][c] for i in range(I) for c in range(C) ) >= s
		for c in range(C):
			ilp += xsum( mat[i][p] * x[p][i][c] for i in range(I) ) <= upper
			ilp += xsum( - mat[i][p] * x[p][i][c] for i in range(I) ) <= -lower
	for i in range(I):
		for c in range(C):
			ilp += xsum(x[p][i][c] for p in range(P)) == 1
		for p in range(P):
			if mat[i][p] == 0:
				ilp += xsum(x[p][i][c] for c in range(C)) == 0
			else:
				ilp += xsum(x[p][i][c] for c in range(C)) == 1
	status = ilp.optimize(max_seconds = max_sec)
	if status != OptimizationStatus.INFEASIBLE:
		break
	else:
		upper += 1
		lower -= 1
		if lower < int(np.floor(S / C) / 2):
			sys.exit("Error: cannot distribute conditions")
		print(" Best bound is not achieved for batch effect minimization. Consider change other parameters and re-run")


# Result
iden = "N%d_P%d_C%d_m%d" % (I,P,C,m)
cphat = [[[ mat[i][p] * x[p][i][c].xi(0) for c in range(C)] for i in range(I) ] for p in range(P)]

# Print the distribution of conditions over batches
cp = np.zeros((P,C))
for p in range(P):
	for c in range(C):
		cp[p, c] += sum([ cphat[p][i][c] * mat[i][p] for i in range(I) ])


outlist = ['\t'.join( ["Pool"] + [ccode[i] for i in range(C)] )]
for i,v in enumerate(cp) :
	nl = '\t'.join( [str(i)] + [str(int(x)) for x in v ] )
	outlist.append(nl)

ctline = '\n'.join(outlist) + '\n'
print(ctline)

# Output final design
res = []
psize = [0] * P
for p in range(P):
	rec = []
	rsize = [0] * I
	for i in range(I):
		if mat[i][p] == 0:
			rec.append('. ')
		else:
			psize[p] += 1
			for c in range(C):
				if cphat[p][i][c] == 1:
					rec.append( ccode[c] + " " )
					rsize[i] += 1
	# print(sum(rsize))
	nl = ''.join(rec)
	res.append(nl)

dsline = '\n'.join(res) + '\n'
f = path + iden + ".design"
with open(f, 'w') as wf:
	_=wf.write(dsline)

print(dsline)
print(ccode[:C])

f = path + iden + ".txt"
with open(f, 'w') as wf:
	wf.write("Number of pools: " + str(P) + "; minimum pairwise distance: " + str(m) + "\n\n")
	wf.write("Number of occurrences in each pool:\n\n")
	wf.write(ctline)
	wf.write("\nSpectrum of pairwise overlap:\n\n")
	wf.write(spectrum)
	wf.write("\n\nDesign matrix:\n\n")
	wf.write(dsline)
