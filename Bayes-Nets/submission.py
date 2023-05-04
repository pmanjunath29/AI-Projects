import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32, random
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_security_system_net():
    """Create a Bayes Net representation of the above security system problem. 
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    BayesNet = BayesianModel()
    # TODO: finish this function
    BayesNet.add_node("H")
    BayesNet.add_node("C")
    BayesNet.add_node("M")
    BayesNet.add_node("B")
    BayesNet.add_node("Q")
    BayesNet.add_node("K")
    BayesNet.add_node("D")
    
    BayesNet.add_edge("C", "Q")
    BayesNet.add_edge("H", "Q")
    BayesNet.add_edge("B", "K")
    BayesNet.add_edge("M", "K")
    BayesNet.add_edge("Q", "D")
    BayesNet.add_edge("K", "D")    
    
    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the security system.
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    # TODO: set the probability distribution for each node
    cpd_h = TabularCPD('H', 2, values=[[0.5], [0.5]])
    cpd_c = TabularCPD('C', 2, values=[[0.7], [0.3]])
    cpd_m = TabularCPD('M', 2, values=[[0.2], [0.8]])
    cpd_b = TabularCPD('B', 2, values=[[0.5], [0.5]])
    #cpd_q = TabularCPD('Q', 2, values=[[0.5], [0.5]])
    #cpd_k = TabularCPD('K', 2, values=[[0.5], [0.5]])
    #cpd_d = TabularCPD('D', 2, values=[[0.5], [0.5]])
    
    bayes_net.add_cpds(cpd_h, cpd_c, cpd_m, cpd_b)
    
    cpd_qhc = TabularCPD('Q', 2, values = [[0.95, 0.75, 0.45, 0.1], [0.05, 0.25, 0.55, 0.9]], evidence=['H', 'C'], evidence_card=[2, 2])
    cpd_kbm = TabularCPD('K', 2, values = [[0.25, 0.05, 0.99, 0.85], [0.75, 0.95, 0.01, 0.15]], evidence=['B', 'M'], evidence_card=[2, 2])
    cpd_dqk = TabularCPD('D', 2, values = [[0.98, 0.65, 0.4, 0.01], [0.02, 0.35, 0.6, 0.99]], evidence=['Q', 'K'], evidence_card=[2, 2])
    
    bayes_net.add_cpds(cpd_qhc, cpd_kbm, cpd_dqk)

    return bayes_net


def get_marginal_double0(bayes_net):
    """Calculate the marginal probability that Double-0 gets compromised.
    """
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables = ['D'], joint=False)
    double0_prob = marginal_prob['D'].values[1]
    #solver = VariableElimination(bayes_net)
    #marginal_prob = solver.query(variables = ['M'], joint=False)
    #double0_prob = marginal_prob['M'].values[0]
    #print(double0_prob)
    #raise NotImplementedError
    return double0_prob


def get_conditional_double0_given_no_contra(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down.
    """
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'],evidence={'C':0}, joint=False)
    double0_prob = conditional_prob['D'].values[1]
    return double0_prob


def get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down and Bond is reassigned to protect M.
    """
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables = ['D'],evidence={'C':0, 'B':1}, joint=False)
    double0_prob = conditional_prob['D'].values[1]
    return double0_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    # TODO: fill this out
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")
    
    BayesNet.add_edge("A", "AvB")
    BayesNet.add_edge("B", "AvB")
    BayesNet.add_edge("B", "BvC")
    BayesNet.add_edge("C", "BvC")
    BayesNet.add_edge("C", "CvA")
    BayesNet.add_edge("A", "CvA")
    
    cpd_a = TabularCPD('A', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_b = TabularCPD('B', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_c = TabularCPD('C', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    
    BayesNet.add_cpds(cpd_a, cpd_b, cpd_c)
    
    cpd_avb = TabularCPD('AvB', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_bvc = TabularCPD('BvC', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_cva = TabularCPD('CvA', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    
    BayesNet.add_cpds(cpd_avb, cpd_bvc, cpd_cva)
    
    AvB_Awins = [0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1]
    AvB_Bwins = [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1]
    ABTie = [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]
    
    BvC_Bwins = [0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1]
    BvC_Cwins = [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1]
    BCTie = [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]
    
    CvA_Cwins = [0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1]
    CvA_Awins = [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1]
    CATie = [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]
    
    cpd_avb_ab = TabularCPD('AvB', 3, [AvB_Awins, AvB_Bwins, ABTie], 
                            evidence = ['A', 'B'], evidence_card = [4, 4])
    cpd_bvc_bc = TabularCPD('BvC', 3, [BvC_Bwins, BvC_Cwins, BCTie],
                            evidence = ['B', 'C'], evidence_card = [4, 4])
    cpd_cva_ca = TabularCPD('CvA', 3, [CvA_Cwins, CvA_Awins, CATie],
                            evidence = ['C', 'A'], evidence_card = [4, 4])
    
    BayesNet.add_cpds(cpd_avb_ab, cpd_bvc_bc, cpd_cva_ca)
    
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables = ['BvC'],evidence={'AvB':0, 'CvA':2}, joint=False)
    posterior = conditional_prob['BvC'].values
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = tuple(initial_state)
    # TODO: finish this function
    sample = list(initial_state)
    dict = {}
    vars = ['A', 'B', 'C', 'AvB', 'BvC', 'CvA']
    
    for i in range(6):
        if i == 3:
            sample[i] = 0
        elif i == 5:
            sample[i] = 2
        elif sample[i] is None:
            if i == 0 or i == 1 or i == 2:
                sample[i] = random.choice([0, 1, 2, 3])
            elif i == 4:
                sample[i] = random.choice([0, 1, 2])
        dict[vars[i]] = sample[i]
        
    idx_choices = [0, 1, 2, 4]
    idx = random.choice(idx_choices)
    idx_choices.remove(idx)
    skill_levels = [0, 1, 2, 3]
    game_results = [0, 1, 2]
    probs = []
    chosen_var = vars[idx]
    
    if chosen_var == 'A':
        for skill in skill_levels:
            p_chosen = bayes_net.get_cpds('A').values[skill]
            p_AvB_chosen = bayes_net.get_cpds('AvB').values[dict['AvB']][skill][dict['B']]
            p_CvA_chosen = bayes_net.get_cpds('CvA').values[dict['CvA']][dict['C']][skill]
            num = p_AvB_chosen * p_CvA_chosen * p_chosen
            den = 0
            for skill2 in skill_levels:
                p_AvB_chosen2 = bayes_net.get_cpds('AvB').values[dict['AvB']][skill2][dict['B']]
                p_CvA_chosen2 = bayes_net.get_cpds('CvA').values[dict['CvA']][dict['C']][skill2]
                den = den + p_AvB_chosen2 * p_CvA_chosen2 * bayes_net.get_cpds('A').values[skill2]
            prob = num / den
            probs.append(prob)
    elif chosen_var == 'B':
        for skill in skill_levels:
            p_chosen = bayes_net.get_cpds('B').values[skill]
            p_BvC_chosen = bayes_net.get_cpds('BvC').values[dict['BvC']][skill][dict['C']]
            p_AvB_chosen = bayes_net.get_cpds('AvB').values[dict['AvB']][dict['A']][skill]
            num = p_BvC_chosen * p_AvB_chosen * p_chosen
            den = 0
            for skill2 in skill_levels:
                p_BvC_chosen2 = bayes_net.get_cpds('BvC').values[dict['BvC']][skill2][dict['C']]
                p_AvB_chosen2 = bayes_net.get_cpds('AvB').values[dict['AvB']][dict['A']][skill2]
                den = den + p_BvC_chosen2 * p_AvB_chosen2 * bayes_net.get_cpds('B').values[skill2]
            prob = num / den
            probs.append(prob)
    elif chosen_var == 'C':
        for skill in skill_levels:
            p_chosen = bayes_net.get_cpds('C').values[skill]
            p_CvA_chosen = bayes_net.get_cpds('CvA').values[dict['CvA']][skill][dict['A']]
            p_BvC_chosen = bayes_net.get_cpds('BvC').values[dict['BvC']][dict['B']][skill]
            num = p_CvA_chosen * p_BvC_chosen * p_chosen
            den = 0
            for skill2 in skill_levels:
                p_CvA_chosen2 = bayes_net.get_cpds('CvA').values[dict['CvA']][skill2][dict['A']]
                p_BvC_chosen2 = bayes_net.get_cpds('BvC').values[dict['BvC']][dict['B']][skill2]
                den = den + p_CvA_chosen2 * p_BvC_chosen2 * bayes_net.get_cpds('B').values[skill2]
            prob = num / den
            probs.append(prob)
    elif chosen_var == 'BvC':
        for result in game_results:
            prob = bayes_net.get_cpds('BvC').values[result][dict['B']][dict['C']]
            probs.append(prob)
            
    if idx == 0 or idx == 1 or idx == 2:
        sample[idx] = random.choice(skill_levels, 1, p=probs)[0]
    else:
        sample[idx] = random.choice(game_results, 1, p=probs)[0]
        
    return tuple(sample)


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    sample = tuple(initial_state)
    # TODO: finish this function
    sample = list(initial_state)
    
    for i in range(6):
        if i == 3:
            sample[i] = 0
        elif i == 5:
            sample[i] = 2
        elif sample[i] is None:
            if i == 0 or i == 1 or i == 2:
                sample[i] = random.choice([0, 1, 2, 3])
            elif i == 4:
                sample[i] = random.choice([0, 1, 2])
                
    next = []
    skill_levels = [0, 1, 2, 3]
    game_results = [0, 1, 2]
    
    for i in range(6):
        if i <= 2:
            next.append(random.choice(skill_levels))
        elif i == 3:
            next.append(0)
        elif i == 4:
            next.append(random.choice(game_results))
        elif i == 5:
            next.append(2)
            
    p_A_next = team_table[next[0]]
    p_B_next = team_table[next[1]]
    p_C_next = team_table[next[2]]
    
    p_A_prior = team_table[sample[0]]
    p_B_prior = team_table[sample[1]]
    p_C_prior = team_table[sample[2]]
    
    p_AvB_AB_next = match_table[next[3]][next[0]][next[1]]
    p_BvC_BC_next = match_table[next[4]][next[1]][next[2]]
    p_CvA_CA_next = match_table[next[5]][next[2]][next[0]]
    
    p_AvB_AB_prior = match_table[sample[3]][sample[0]][sample[1]]
    p_BvC_BC_prior = match_table[sample[4]][sample[1]][sample[2]]
    p_CvA_CA_prior = match_table[sample[5]][sample[2]][sample[0]]
    
    num = p_A_next * p_B_next * p_C_next * p_AvB_AB_next * p_BvC_BC_next * p_CvA_CA_next
    den = p_A_prior * p_B_prior * p_C_prior * p_AvB_AB_prior * p_BvC_BC_prior * p_CvA_CA_prior
    
    alpha = num / den
    if alpha >= 1:
        sample = next
    else:
        choice = random.choice([0, 1], 1, p = [alpha, 1 - alpha])
        if choice == 0:
            sample = next
    
    return tuple(sample)


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    delta = 0.000001
    N = 100
    Burnout = 10000
    Gibbs_sample = initial_state
    Gibbs_results = [0, 0, 0]
    iters_since_conv = 0
    diffs = []
    
    while iters_since_conv != N:
        Gibbs_sample = Gibbs_sampler(bayes_net, Gibbs_sample)
        Gibbs_count = Gibbs_count + 1
        if Gibbs_count > Burnout:
            Gibbs_results[Gibbs_sample[4]] = Gibbs_results[Gibbs_sample[4]] + 1
            Gibbs_total = Gibbs_results[0] + Gibbs_results[1] + Gibbs_results[2]
            Gibbs_convergence = [Gibbs_results[0] / Gibbs_total, Gibbs_results[1] / Gibbs_total, Gibbs_results[2] / Gibbs_total]
            diffs.append((Gibbs_convergence))
            iters_since_conv = 0
            if Gibbs_count > N + Burnout:
                for i in range(N):
                    avg_conv = (abs(diffs[-i - 1][0] - diffs[-i - 2][0]) + abs(diffs[-i - 1][1] - diffs[-i - 2][1]) + abs(diffs[-i - 1][2] - diffs[-i - 2][2])) / 3
                    if avg_conv < delta:
                        iters_since_conv = iters_since_conv + 1
    
    MH_sample = initial_state
    MH_results = [0, 0, 0]
    iters_since_conv = 0
    diffs = []
    
    while iters_since_conv != N:
        MH_sample_prior = MH_sample
        MH_sample = MH_sampler(bayes_net, MH_sample)
        MH_count = MH_count + 1
        if MH_sample_prior == MH_sample:
            MH_rejection_count = MH_rejection_count + 1
        if MH_count > Burnout:
            MH_results[MH_sample[4]] = MH_results[MH_sample[4]] + 1
            MH_total = MH_results[0] + MH_results[1] + MH_results[2]
            MH_convergence = [MH_results[0] / MH_total, MH_results[1] / MH_total, MH_results[2] / MH_total]
            diffs.append((MH_convergence))
            iters_since_conv = 0
            if MH_count > N + Burnout:
                for i in range(N):
                    avg_conv = (abs(diffs[-i - 1][0] - diffs[-i - 2][0]) + abs(diffs[-i - 1][1] - diffs[-i - 2][1]) + abs(diffs[-i - 1][2] - diffs[-i - 2][2])) / 3
                    if avg_conv < delta:
                        iters_since_conv = iters_since_conv + 1
                        
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    Bayes_net = get_game_network()
    Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count = compare_sampling(Bayes_net, [0, 0, 0, 0, 0, 0])
    
    if Gibbs_count < MH_count:
        choice = 0
        factor = MH_count / Gibbs_count
    else:
        choice = 1
        factor = Gibbs_count / MH_count
        
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Pranav Shankar Manjunath"
