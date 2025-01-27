import torch as t
import random
import numpy as np
from evotorch import Problem, Solution, SolutionBatch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import trange
from typing import Tuple, Dict, Iterable, Callable, List

# native imports
from steel_scheduling.utilities.policies import AttentionPolicy
from steel_scheduling.simulator import Steel_plant



#######################################################################################################################


class evotorch_wrapper(Problem):
    def __init__(self, cfg: DictConfig, simulator, dimensionality: int, initial_bounds: Tuple
                 ):
        super().__init__(objective_sense="min",
                         solution_length=dimensionality,
                         initial_bounds=initial_bounds,
                         num_actors = 'max',
                         store_solution_stats=True)


        self.cfg = cfg
        self.simulator = simulator
        self.net_dict = self.simulator.net.state_dict()

    def _evaluate(self, solution: Solution) -> t.Tensor:
        x = solution.values

        n = 0
        net_dict = {}

        for key, value in self.net_dict.items():
            net_dict[key] = x[n:n + value.numel()].reshape(value.shape)
            n += value.numel()

        agent = self.simulator.from_params(net_dict, self.cfg)
        loss = agent.fitness()
        solution.set_evals(loss)

        return solution.values


class Simulator:
    def __init__(self, cfg: DictConfig):

        self.cfg = cfg
        self.env = Steel_plant(cfg)
        self.seed = 0

        if self.cfg.policy.name == "AttentionPolicy":

            self.net = AttentionPolicy(cfg)
            self.control_inference = self.attention_policy
            self.create_controls = self.generating_controls

        else:
            raise NotImplementedError(f"{self.cfg.policy.name} not configured")

    @classmethod
    def from_params(cls, params: Dict[str, t.Tensor], cfg: DictConfig):

        simulator = cls(cfg)
        simulator.net.load_state_dict(params)
        return simulator

    def attention_policy(self, state):

        state = t.tensor(state)

        s_ = state.view((1, state.shape[0], state.shape[1]))

        with t.no_grad():
            control_rules = self.net(s_)


        return control_rules

    def generating_controls(self, control_mask, control_rules):

        np.random.seed(self.seed)


        t = self.env.period

        new_control_mask = { }

        ################################# Modifying the control mask  according to decision rules #####################

        ########################################## Stage I , Unit I ####################################################
        ################################ Shortest processing time (SPT) ###############################################

        if control_rules[0] == 0:

            for ranking in self.cfg.EAF.SPT:
                if any(elem in self.cfg.EAF.SPT[ranking] for elem in control_mask[0]):
                    new_control_mask[0] =  [elem for elem in self.cfg.EAF.SPT[ranking] if elem in control_mask[0]]
                    break
                else:
                    new_control_mask[0] = control_mask[0] # element is [0]

            #print("control_mask 0:", new_control_mask[0])

        ################################ Longest  processing time (LPT) ###############################################

        elif control_rules[0] == 1:

            for ranking in self.cfg.EAF.LPT:
                if any(elem in self.cfg.EAF.LPT[ranking] for elem in control_mask[0]):
                    new_control_mask[0] =  [elem for elem in self.cfg.EAF.LPT[ranking] if elem in control_mask[0]]
                    break
                else:
                    new_control_mask[0] = control_mask[0] # element is [0]

            #print("control_mask 0:", new_control_mask[0])

        #############  Shortest remaining machine time not including current processing time (SRM) ####################

        elif control_rules[0] == 2:

            for ranking in self.cfg.EAF.SRM:
                if any(elem in self.cfg.EAF.SRM[ranking] for elem in control_mask[0]):
                    new_control_mask[0] =  [elem for elem in self.cfg.EAF.SRM[ranking] if elem in control_mask[0]]
                    break
                else:
                    new_control_mask[0] = control_mask[0] # element is [0]

            #print("control_mask 0:", new_control_mask[0])

        #############  Longest remaining machine time not including current processing time (LRM) ####################

        elif control_rules[0] == 3:

            for ranking in self.cfg.EAF.LRM:
                if any(elem in self.cfg.EAF.LRM[ranking] for elem in control_mask[0]):
                    new_control_mask[0] =  [elem for elem in self.cfg.EAF.LRM[ranking] if elem in control_mask[0]]
                    break
                else:
                    new_control_mask[0] = control_mask[0] # element is [0]

            #print("control_mask 0:", new_control_mask[0])

        ############################################  Dont do anything  ################################################

        else:

            if 0 in control_mask[0]:
                new_control_mask[0] = [0]
            else:
                new_control_mask[0] = control_mask[0]

            #print("control_mask 0:", new_control_mask[0])

        ########################################## Stage I , Unit II ###################################################
        ################################ Shortest processing time (SPT) ###############################################

        if control_rules[1] == 0:

            for ranking in self.cfg.EAF.SPT:
                if any(elem in self.cfg.EAF.SPT[ranking] for elem in control_mask[1]):
                    new_control_mask[1] =  [elem for elem in self.cfg.EAF.SPT[ranking] if elem in control_mask[1]]
                    break
                else:
                    new_control_mask[1] = control_mask[1] # element is [0]

            #print("control_mask 1:", new_control_mask[1])

        ################################ Longest  processing time (SPT) ###############################################

        elif control_rules[1] == 1:

            for ranking in self.cfg.EAF.LPT:
                if any(elem in self.cfg.EAF.LPT[ranking] for elem in control_mask[1]):
                    new_control_mask[1] =  [elem for elem in self.cfg.EAF.LPT[ranking] if elem in control_mask[1]]
                    break
                else:
                    new_control_mask[1] = control_mask[1] # element is [0]

            #print("control_mask 1:", new_control_mask[1])

        #############  Shortest remaining machine time not including current processing time (SRM) ####################

        elif control_rules[1] == 2:

            for ranking in self.cfg.EAF.SRM:
                if any(elem in self.cfg.EAF.SRM[ranking] for elem in control_mask[1]):
                    new_control_mask[1] =  [elem for elem in self.cfg.EAF.SRM[ranking] if elem in control_mask[1]]
                    break
                else:
                    new_control_mask[1] = control_mask[1] # element is [0]


            #print("control_mask 1:", new_control_mask[1])

        #############  Longest remaining machine time not including current processing time (LRM) ####################

        elif control_rules[1] == 3:

            for ranking in self.cfg.EAF.LRM:
                if any(elem in self.cfg.EAF.LRM[ranking] for elem in control_mask[1]):
                    new_control_mask[1] =  [elem for elem in self.cfg.EAF.LRM[ranking] if elem in control_mask[1]]
                    break
                else:
                    new_control_mask[1] = control_mask[1] # element is [0]


            #print("control_mask 1:", new_control_mask[1])

        ############################################  Dont do anything  ################################################

        else:

            if 0 in control_mask[1]:
                new_control_mask[1] = [0]
            else:
                new_control_mask[1] = control_mask[1]

            #print("control_mask 1:", new_control_mask[1])

        ########################################## Stage II , Unit I ###################################################
        ################################ Shortest processing time (SPT) ###############################################

        if control_rules[2] == 0:

            for ranking in self.cfg.AOD.SPT:
                if any(elem in self.cfg.AOD.SPT[ranking] for elem in control_mask[2]):
                    new_control_mask[2] =  [elem for elem in self.cfg.AOD.SPT[ranking] if elem in control_mask[2]]
                    break
                else:
                    new_control_mask[2] = control_mask[2] # element is [0]


            #print("control_mask 2:", new_control_mask[2])

        ################################ Longest  processing time (SPT) ################################################

        elif control_rules[2] == 1:

            for ranking in self.cfg.AOD.LPT:
                if any(elem in self.cfg.AOD.LPT[ranking] for elem in control_mask[2]):
                    new_control_mask[2] =  [elem for elem in self.cfg.AOD.LPT[ranking] if elem in control_mask[2]]
                    break
                else:
                    new_control_mask[2] = control_mask[2] # element is [0]

            #print("control_mask 2:", new_control_mask[2])

        #############  Shortest remaining machine time not including current processing time (SRM) #####################

        elif control_rules[2] == 2:

            for ranking in self.cfg.AOD.SRM:
                if any(elem in self.cfg.AOD.SRM[ranking] for elem in control_mask[2]):
                    new_control_mask[2] =  [elem for elem in self.cfg.AOD.SRM[ranking] if elem in control_mask[2]]
                    break
                else:
                    new_control_mask[2] = control_mask[2] # element is [0]

            #print("control_mask 2:", new_control_mask[2])

      #############  Longest remaining machine time not including current processing time (LRM) ######################

        elif control_rules[2] == 3:

            for ranking in self.cfg.AOD.LRM:
                if any(elem in self.cfg.AOD.LRM[ranking] for elem in control_mask[2]):
                    new_control_mask[2] =  [elem for elem in self.cfg.AOD.LRM[ranking] if elem in control_mask[2]]
                    break
                else:
                    new_control_mask[2] = control_mask[2] # element is [0]

            #print("control_mask 2:", new_control_mask[2])

        ###############################  Most former task in waiting area ##############################################

        elif control_rules[2] == 4:

            appended_tasks = []
            for task in self.env.stage1_copy.completed_transfer_tasks_policy(t):
                if task.id in control_mask[2]:
                    appended_tasks.append(task)

            if appended_tasks:
                lowest_end_time = min(task.end_time for task in appended_tasks)
                tasks_with_lowest_end_time = [task.id for task in appended_tasks if
                                              task.end_time == lowest_end_time]
                new_control_mask[2] =  tasks_with_lowest_end_time

                # lowest_end_time_task = min(appended_tasks, key=lambda task: task.end_time)
                # control_mask[2] = ([0] if 0 in control_mask[2] else []) + [lowest_end_time_task.id]
            else:
                new_control_mask[2] = control_mask[2]

            #print("control_mask 2:", new_control_mask[2])

        ###############################  Most recent task in waiting area ############################################

        elif control_rules[2] == 5:

            appended_tasks = []
            for task in self.env.stage1_copy.completed_transfer_tasks_policy(t):
                if task.id in control_mask[2]:
                    appended_tasks.append(task)

            if appended_tasks:
                highest_end_time = max(task.end_time for task in appended_tasks)
                tasks_with_highest_end_time = [task.id for task in appended_tasks if
                                               task.end_time == highest_end_time]
                new_control_mask[2] =  tasks_with_highest_end_time

            else:
                new_control_mask[2] = control_mask[2]

            #print("control_mask 2:", new_control_mask[2])

        ############################################  Dont do anything  ################################################

        else:

            if 0 in control_mask[2]:
                new_control_mask[2] = [0]
            else:
                new_control_mask[2] = control_mask[2]

            #print("control_mask 2:", new_control_mask[2])

        ########################################## Stage II, Unit II ###################################################
        ################################ Shortest processing time (SPT) ###############################################

        if control_rules[3] == 0:
            for ranking in self.cfg.AOD.SPT:
                if any(elem in self.cfg.AOD.SPT[ranking] for elem in control_mask[3]):
                    new_control_mask[3] =  [elem for elem in self.cfg.AOD.SPT[ranking] if elem in control_mask[3]]
                    break
                else:
                    new_control_mask[3] = control_mask[3] # element is [0]

           # print("control_mask 3:", new_control_mask[3])

        ################################ Longest  processing time (SPT) ###############################################

        elif control_rules[3] == 1:

            for ranking in self.cfg.AOD.LPT:
                if any(elem in self.cfg.AOD.LPT[ranking] for elem in control_mask[3]):
                    new_control_mask[3] =  [elem for elem in self.cfg.AOD.LPT[ranking] if elem in control_mask[3]]
                    break
                else:
                    new_control_mask[3] = control_mask[3] # element is [0]

            #print("control_mask 3:", new_control_mask[3])

        #############  Shortest remaining machine time not including current processing time (SRM) ####################

        elif control_rules[3] == 2:

            for ranking in self.cfg.AOD.SRM:
                if any(elem in self.cfg.AOD.SRM[ranking] for elem in control_mask[3]):
                    new_control_mask[3] =  [elem for elem in self.cfg.AOD.SRM[ranking] if elem in control_mask[3]]
                    break
                else:
                    new_control_mask[3] = control_mask[3] # element is [0]

            #print("control_mask 3:", new_control_mask[3])

        #############  Longest remaining machine time not including current processing time (LRM) ####################

        elif control_rules[3] == 3:

            for ranking in self.cfg.AOD.LRM:
                if any(elem in self.cfg.AOD.LRM[ranking] for elem in control_mask[3]):
                    new_control_mask[3] =  [elem for elem in self.cfg.AOD.LRM[ranking] if elem in control_mask[3]]
                    break
                else:
                    new_control_mask[3] = control_mask[3] # element is [0]

            #print("control_mask 3:", new_control_mask[3])

        ###############################  Most former task in waiting area ##############################################

        elif control_rules[3] == 4:

            appended_tasks = []
            for task in self.env.stage1_copy.completed_transfer_tasks_policy(t):
                if task.id in control_mask[3]:
                    appended_tasks.append(task)

            if appended_tasks:
                lowest_end_time = min(task.end_time for task in appended_tasks)
                tasks_with_lowest_end_time = [task.id for task in appended_tasks if
                                              task.end_time == lowest_end_time]
                new_control_mask[3] = tasks_with_lowest_end_time

                # lowest_end_time_task = min(appended_tasks, key=lambda task: task.end_time)
                # control_mask[3] = ([0] if 0 in control_mask[3] else []) + [lowest_end_time_task.id]
            else:
                new_control_mask[3] = control_mask[3]

            #print("control_mask 3:", new_control_mask[3])

        ###############################  Most recent task in waiting area ############################################

        elif control_rules[3] == 5:

            appended_tasks = []
            for task in self.env.stage1_copy.completed_transfer_tasks_policy(t):
                if task.id in control_mask[3]:
                    appended_tasks.append(task)

            if appended_tasks:
                highest_end_time = max(task.end_time for task in appended_tasks)
                tasks_with_highest_end_time = [task.id for task in appended_tasks if
                                               task.end_time == highest_end_time]
                new_control_mask[3] =  tasks_with_highest_end_time

            else:
                new_control_mask[3] = control_mask[3]

            #print("control_mask 3:", new_control_mask[3])

        ############################################  Dont do anything  ################################################

        else:

            if 0 in control_mask[3]:
                new_control_mask[3] = [0]
            else:
                new_control_mask[3] = control_mask[3]

            #print("control_mask 3:", new_control_mask[3])

        ########################################## Stage III, Unit I  ##################################################
        ################################ Shortest processing time (SPT) ###############################################

        if control_rules[4] == 0:

            for ranking in self.cfg.LF.SPT:
                if any(elem in self.cfg.LF.SPT[ranking] for elem in control_mask[4]):
                    new_control_mask[4] =  [elem for elem in self.cfg.LF.SPT[ranking] if elem in control_mask[4]]
                    break
                else:
                    new_control_mask[4] = control_mask[4] # element is [0]

            #print("control_mask 4:", new_control_mask[4])

        ################################ Longest  processing time (LPT) ###############################################

        elif control_rules[4] == 1:

            for ranking in self.cfg.LF.LPT:
                if any(elem in self.cfg.LF.LPT[ranking] for elem in control_mask[4]):
                    new_control_mask[4] =  [elem for elem in self.cfg.LF.LPT[ranking] if elem in control_mask[4]]
                    break
                else:
                    new_control_mask[4] = control_mask[4] # element is [0]

            #print("control_mask 4:", new_control_mask[4])

        ###############################  Most former task in waiting area ##############################################

        elif control_rules[4] == 2:

            appended_tasks = []
            for task in self.env.stage2_copy.completed_transfer_tasks_policy(t):
                if task.id in control_mask[4]:
                    appended_tasks.append(task)

            if appended_tasks:
                lowest_end_time = min(task.end_time for task in appended_tasks)
                tasks_with_lowest_end_time = [task.id for task in appended_tasks if
                                              task.end_time == lowest_end_time]
                new_control_mask[4] =  tasks_with_lowest_end_time

            else:
                new_control_mask[4] = control_mask[4]

            #print("control_mask 4:", new_control_mask[4])


        ###############################  Most recent task in waiting area ############################################

        elif control_rules[4] == 3:

            appended_tasks = []
            for task in self.env.stage2_copy.completed_transfer_tasks_policy(t):
                if task.id in control_mask[4]:
                    appended_tasks.append(task)

            if appended_tasks:
                highest_end_time = max(task.end_time for task in appended_tasks)
                tasks_with_highest_end_time = [task.id for task in appended_tasks if
                                               task.end_time == highest_end_time]
                new_control_mask[4] =  tasks_with_highest_end_time

            else:
                new_control_mask[4] = control_mask[4]

            #print("control_mask 4:", new_control_mask[4])

        ############################################  Dont do anything  ################################################

        else:

            if 0 in control_mask[4]:
                new_control_mask[4] = [0]
            else:
                new_control_mask[4] = control_mask[4]

            #print("control_mask 4:", new_control_mask[4])

        ########################################## Stage III, Unit II  #################################################
        ################################ Shortest processing time (SPT) ###############################################

        if control_rules[5] == 0:

            for ranking in self.cfg.LF.SPT:
                if any(elem in self.cfg.LF.SPT[ranking] for elem in control_mask[5]):
                    new_control_mask[5] =  [elem for elem in self.cfg.LF.SPT[ranking] if elem in control_mask[5]]
                    break
                else:
                    new_control_mask[5] = control_mask[5] # element is [0]

            #print("control_mask 5:", new_control_mask[5])

        ################################ Longest  processing time (LPT) ###############################################

        elif control_rules[5] == 1:

            for ranking in self.cfg.LF.LPT:
                if any(elem in self.cfg.LF.LPT[ranking] for elem in control_mask[5]):
                    new_control_mask[5] =  [elem for elem in self.cfg.LF.LPT[ranking] if elem in control_mask[5]]
                    break
                else:
                    new_control_mask[5] = control_mask[5] # element is [0]

            #print("control_mask 5:", new_control_mask[5])

        ###############################  Most former task in waiting area ##############################################

        elif control_rules[5] == 2:

            appended_tasks = []
            for task in self.env.stage2_copy.completed_transfer_tasks_policy(t):
                if task.id in control_mask[5]:
                    appended_tasks.append(task)

            if appended_tasks:
                lowest_end_time = min(task.end_time for task in appended_tasks)
                tasks_with_lowest_end_time = [task.id for task in appended_tasks if
                                              task.end_time == lowest_end_time]
                new_control_mask[5] =  tasks_with_lowest_end_time

            else:
                new_control_mask[5] = control_mask[5]
            #print("control_mask 5:", new_control_mask[5])

        ###############################  Most recent task in waiting area ############################################

        elif control_rules[5] == 3:

            appended_tasks = []
            for task in self.env.stage2_copy.completed_transfer_tasks_policy(t):
                if task.id in control_mask[5]:
                    appended_tasks.append(task)

            if appended_tasks:
                highest_end_time = max(task.end_time for task in appended_tasks)
                tasks_with_highest_end_time = [task.id for task in appended_tasks if
                                               task.end_time == highest_end_time]
                new_control_mask[5] =  tasks_with_highest_end_time

            else:
                new_control_mask[5] = control_mask[5]

            #print("control_mask 5:", new_control_mask[5])

        ############################################  Dont do anything  ################################################

        else:

            if 0 in control_mask[5]:
                new_control_mask[5] = [0]
            else:
                new_control_mask[5] = control_mask[5]

            #print("control_mask 5:", new_control_mask[5])

        ################### Making control decisions ##################################################################

        control = []

        ############################### Stage I,Unit I #################################################################

        if control_rules[0] == 0: # SPT

            same_rank_tasks = [len([elem for elem in new_control_mask[0] if elem in ranking]) for ranking in
                               self.cfg.EAF.SPT.values()]

            if any(num > 1 for num in same_rank_tasks):                                                                 # if there are same rank tasks then sample the control
                control.extend([np.random.choice(new_control_mask[0])])
            else:
                control.extend(new_control_mask[0])

            #print("control 0 :", control[0])

        elif control_rules[0] == 1: # LPT

            same_rank_tasks = [len([elem for elem in new_control_mask[0] if elem in ranking]) for ranking in
                               self.cfg.EAF.LPT.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[0])])
            else:
                control.extend(new_control_mask[0])

            #print("control 0 :", control[0])

        elif control_rules[0] == 2: # SRM

            same_rank_tasks = [len([elem for elem in new_control_mask[0] if elem in ranking]) for ranking in
                               self.cfg.EAF.SRM.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[0])])
            else:
                control.extend(new_control_mask[0])

            #print("control 0 :", control[0])

        elif control_rules[0] == 3: #LRM
            same_rank_tasks = [len([elem for elem in new_control_mask[0] if elem in ranking]) for ranking in
                               self.cfg.EAF.LRM.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[0])])
            else:
                control.extend(new_control_mask[0])
            #print("control 0 :", control[0])


        else: #DN
            if 0 in control_mask[0]:
                control.extend([0])
            else:
                control.extend(new_control_mask[0])
            #print("control 0 :", control[0])

        ############################### Stage I,Unit II ################################################################

        if control_rules[1] == 0: #SPT

            same_rank_tasks = [len([elem for elem in new_control_mask[1] if elem in ranking]) for ranking in
                               self.cfg.EAF.SPT.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[1])])
            else:
                control.extend(new_control_mask[1])
            #print("control 1 :", control[1])



        elif control_rules[1] == 1: #LPT

            same_rank_tasks = [len([elem for elem in new_control_mask[1] if elem in ranking]) for ranking in
                               self.cfg.EAF.LPT.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[1])])
            else:
                control.extend(new_control_mask[1])

            #print("control 1 :", control[1])



        elif control_rules[1] == 2: #SRM

            same_rank_tasks = [len([elem for elem in new_control_mask[1] if elem in ranking]) for ranking in
                               self.cfg.EAF.SRM.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[1])])
            else:
                control.extend(new_control_mask[1])
            #print("control 1 :", control[1])



        elif control_rules[1] == 3: #LRM
            same_rank_tasks = [len([elem for elem in new_control_mask[1] if elem in ranking]) for ranking in
                               self.cfg.EAF.LRM.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[1])])
            else:
                control.extend(new_control_mask[1])
            #print("control 1 :", control[1])

        else: #DN
            if 0 in control_mask[1]:
                control.extend([0])
            else:
                control.extend(new_control_mask[1])
            #print("control 1 :", control[1])

        ############################### Stage II,Unit I ################################################################

        if control_rules[2] == 0: #SPT

            same_rank_tasks = [len([elem for elem in new_control_mask[2] if elem in ranking]) for ranking in
                               self.cfg.AOD.SPT.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[2])])
            else:
                control.extend(new_control_mask[2])
            #print("control 2 :", control[2])



        elif control_rules[2] == 1: #LPT

            same_rank_tasks = [len([elem for elem in new_control_mask[2] if elem in ranking]) for ranking in
                               self.cfg.AOD.LPT.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[2])])
            else:
                control.extend(new_control_mask[2])
            #print("control 2 :", control[2])



        elif control_rules[2] == 2: #SRM

            same_rank_tasks = [len([elem for elem in new_control_mask[2] if elem in ranking]) for ranking in
                               self.cfg.AOD.SRM.values()]

            if any(num > 1 for num in same_rank_tasks):
               control.extend([np.random.choice(new_control_mask[2])])
            else:
                control.extend(new_control_mask[2])
            #print("control 2 :", control[2])



        elif control_rules[2] == 3: #LRM

            same_rank_tasks = [len([elem for elem in new_control_mask[2] if elem in ranking]) for ranking in
                               self.cfg.AOD.LRM.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[2])])
            else:
                control.extend(new_control_mask[2])
            #print("control 2 :", control[2])



        elif control_rules[2] == 4: # MFT

            if len(new_control_mask[2]) > 1:
                control.extend([np.random.choice(new_control_mask[2])])
            else:
                control.extend(new_control_mask[2])
            #print("control 2 :", control[2])



        elif control_rules[2] == 5: # MRT
            if len(new_control_mask[2]) > 1:
                control.extend([np.random.choice(new_control_mask[2])])
            else:
                control.extend(new_control_mask[2])
            #print("control 2 :", control[2])



        else: #DN
            if 0 in control_mask[2]:
                control.extend([0])
            else:
                control.extend(new_control_mask[2])
            #print("control 2 :", control[2])

        ############################### Stage II,Unit II ################################################################

        if control_rules[3] == 0: #SPT

            same_rank_tasks = [len([elem for elem in new_control_mask[3] if elem in ranking]) for ranking in
                               self.cfg.AOD.SPT.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[3])])
            else:
                control.extend(new_control_mask[3])

            #print("control 3 :", control[3])



        elif control_rules[3] == 1: #LPT

            same_rank_tasks = [len([elem for elem in new_control_mask[3] if elem in ranking]) for ranking in
                               self.cfg.AOD.LPT.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[3])])
            else:
                control.extend(new_control_mask[3])

            #print("control 3 :", control[3])



        elif control_rules[3] == 2: #SRM

            same_rank_tasks = [len([elem for elem in new_control_mask[3] if elem in ranking]) for ranking in
                               self.cfg.AOD.SRM.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[3])])
            else:
                control.extend(new_control_mask[3])
            #print("control 3 :", control[3])



        elif control_rules[3] == 3: #LRM
            same_rank_tasks = [len([elem for elem in new_control_mask[3] if elem in ranking]) for ranking in
                               self.cfg.AOD.LRM.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[3])])
            else:
                control.extend(new_control_mask[3])
            #print("control 3 :", control[3])



        elif control_rules[3] == 4: #MFT

            if len(new_control_mask[3]) > 1:
                control.extend([np.random.choice(new_control_mask[3])])
            else:
                control.extend(new_control_mask[3])
            #print("control 3 :", control[3])



        elif control_rules[3] == 5: #MRT
            if len(new_control_mask[3]) > 1:
                control.extend([np.random.choice(new_control_mask[3])])
            else:
                control.extend(new_control_mask[3])
            #print("control 3 :", control[3])


        else: #DN
            if 0 in control_mask[3]:
                control.extend([0])
            else:
                control.extend(new_control_mask[3])
            #print("control 3 :", control[3])

        ############################### Stage III,Unit I ################################################################

        if control_rules[4] == 0: #SPT

            same_rank_tasks = [len([elem for elem in new_control_mask[4] if elem in ranking]) for ranking in
                               self.cfg.LF.SPT.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[4])])
            else:
                control.extend(new_control_mask[4])
            #print("control 4 :", control[4])



        elif control_rules[4] == 1: #LPT

            same_rank_tasks = [len([elem for elem in new_control_mask[4] if elem in ranking]) for ranking in
                               self.cfg.LF.LPT.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[4])])
            else:
                control.extend(new_control_mask[4])

            #print("control 4 :", control[4])


        elif control_rules[4] == 2: #MFT

            if len(new_control_mask[4]) > 1:
                control.extend([np.random.choice(new_control_mask[4])])
            else:
                control.extend(new_control_mask[4])
            #print("control 4 :", control[4])



        elif control_rules[4] == 3: #MRT
            if len(new_control_mask[4]) > 1:
                control.extend([np.random.choice(new_control_mask[4])])
            else:
                control.extend(new_control_mask[4])

            #print("control 4 :", control[4])


        else: # DN
            if 0 in control_mask[4]:
                control.extend([0])
            else:
                control.extend(new_control_mask[4])

            #print("control 4 :", control[4])

        ############################### Stage III,Unit II ##############################################################

        if control_rules[5] == 0: #SPT

            same_rank_tasks = [len([elem for elem in new_control_mask[5] if elem in ranking]) for ranking in
                               self.cfg.LF.SPT.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[5])])
            else:
                control.extend(new_control_mask[5])

            #print("control 5 :", control[5])

        elif control_rules[5] == 1: #LPT

            same_rank_tasks = [len([elem for elem in control_mask[5] if elem in ranking]) for ranking in
                               self.cfg.LF.LPT.values()]

            if any(num > 1 for num in same_rank_tasks):
                control.extend([np.random.choice(new_control_mask[5])])
            else:
                control.extend(new_control_mask[5])

            #print("control 5 :", control[5])

        elif control_rules[5] == 2: #MFT

            if len(new_control_mask[5]) > 1:
                control.extend([np.random.choice(new_control_mask[5])])
            else:
                control.extend(new_control_mask[5])

            #print("control 5 :", control[5])



        elif control_rules[5] == 3: #MRT

            if len(new_control_mask[5]) > 1:
                control.extend([np.random.choice(new_control_mask[5])])
            else:
                control.extend(new_control_mask[5])

            #print("control 5 :", control[5])



        else: #DN
            if 0 in control_mask[5]:
                control.extend([0])
            else:
                control.extend(new_control_mask[5])
            #print("control 5 :", control[5])

       # print(control)

        ############################# no two controls in same stage should be same ####################################

        if control[0] == control[1] and control[0] == 0:
            pass
        elif control[0] == control[1]:
            available_choices = [x for x in control_mask[1] if x != control[0]]
            control[1] = np.random.choice(available_choices)
        else:
            pass

        if control[2] == control[3] and control[2] == 0:
            pass
        elif control[2] == control[3]:
            available_choices = [x for x in control_mask[3] if x != control[2]]
            control[3] = np.random.choice(available_choices)
        else:
            pass


        if control[4] == control[5] and control[4] == 0:
            pass
        elif control[4] == control[5]:
            available_choices = [x for x in control_mask[5] if x != control[4]]
            control[5] = np.random.choice(available_choices)
        else:
            pass


        return control


    def sample_episode(self, render=False):

        state, control_mask = self.env.reset()

        total_electricity_cost = 0


        for i in range(int(self.env.periods)):


            control_rules =  self.control_inference(state)
            control = self.create_controls(control_mask,control_rules)
            state, current_electricity_cost, control_mask, done, info = self.env.step(control)
            total_electricity_cost += current_electricity_cost

            if i == 95:
                if self.env.all_heats_finished != 0:
                    if len(self.env.incomplete_tasks) == 1:
                        total_electricity_cost = total_electricity_cost + 100000
                    elif len(self.env.incomplete_tasks) == 2:
                        total_electricity_cost = total_electricity_cost + 125000
                    else:
                        total_electricity_cost = total_electricity_cost + 150000


            if done:
                break

        is_spinning_constraints_violated = self.env.spinning_reserve_constraint()
        if 1 in is_spinning_constraints_violated:
            # add penalty
            total_electricity_cost = total_electricity_cost + 150000

        if render:
            self.env.render()


        return t.tensor(total_electricity_cost)

    def fitness(self, hist_id="rewards", global_step=None, **kwargs) -> t.Tensor:
        samples = t.Tensor(
            [
                self.sample_episode(**kwargs)
                for _ in trange(self.cfg.policy.samples, desc="Sampling episodes")

            ]
        )

        fitness = t.mean(samples)

        return fitness
