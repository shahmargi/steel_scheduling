"""
Testing of SNES on production scheduling
author: Margi Shah and Max Mowbray
"""
import os
import hydra
import torch as t
import pickle

from omegaconf import DictConfig, OmegaConf
from evotorch.algorithms import SNES, XNES, PGPE ,CEM
from evotorch.logging import StdOutLogger, PandasLogger, PicklingLogger
from steel_scheduling.utilities.optimizer_utils import evotorch_wrapper, Simulator


@hydra.main(config_path='./config', config_name="default", version_base=None)
def foo(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")
    # %% <Load and Pre-Process Dataset>

    # getting hydra output directory
    orig_cwd = hydra.utils.get_original_cwd()

    # Read file

    # define optimizer
    # Create a SearchAlgorithm instance to optimise the Problem instance
    if cfg.evotorch.algorithm == 'SNES':
        algorithm_optimiser = SNES
        algorithm_kwargs = cfg.snes
    elif cfg.evotorch.algorithm == 'XNES':
        algorithm_optimiser = XNES
        algorithm_kwargs = cfg.xnes
    elif cfg.evotorch.algorithm == 'PGPE':
        algorithm_optimiser = PGPE
        algorithm_kwargs = cfg.pgpe
    else:
        raise NotImplementedError(f'{cfg.evotorch.algorithm} optimiser not implemented')

    simulator = Simulator(cfg)
    dimensionality = simulator.net.count_params()


    evotorch_problem = evotorch_wrapper(
        cfg=cfg,
        simulator=simulator,
        dimensionality=dimensionality,
        initial_bounds=(-5, 5)
        )

    
    ############################### Using SNES default parameters ######################################################

    searcher = algorithm_optimiser(evotorch_problem,
                                   center_init=algorithm_kwargs.center_init,
                                   radius_init=algorithm_kwargs.radius_init,
                                   popsize=algorithm_kwargs.popsize,
                                   center_learning_rate=algorithm_kwargs.center_learning_rate,
                                   stdev_max_change=algorithm_kwargs.stdev_max_change
                                   )

    

    
    def further_population_info():
        population_evals = searcher.population.evals[:, 0]
        return {"pop_evals_stdev": float(t.std(population_evals))}

    searcher.after_step_hook.append(further_population_info)

    # Create loggers as desired
    stdout_logger = StdOutLogger(searcher)                                                                              # Status printed to the stdout

    pandas_logger = PandasLogger(searcher)                                                                              # Status stored in a Pandas dataframe
    pickler = PicklingLogger(searcher, interval=5)

    best_evals_list=[]
    center_evals_list=[]

    for generation in range(1, 1 + cfg.dfo_args_general.max_generations):
        searcher.step()

        # bestsolution.evals append
        best_solution = searcher.status["best"]
        best_evals_list.append(best_solution.evals)

        # center solution.evals append
        center_solution = searcher.status["center"]
        center_batch = evotorch_problem.generate_batch(1)
        center_batch[0].set_values(searcher.status["center"])
        evotorch_problem.evaluate(center_batch)
        center_evals_list.append(center_batch[0].evals[0])


  ######################################################################################################################

    progress = pandas_logger.to_dataframe()                                                                             # Process the information accumulated by the loggers.
    fig___ = progress.best_eval.plot(figsize=(20, 16), fontsize=26).get_figure()                                         # Display a graph of the evolutionary progress by using the pandas data frame
    fig___.savefig('training_progress.svg')
    
    # Save data to a text file
    progress.to_csv('training_progress.txt', sep='\t', index=False)

    # for best solution
    # with open("savedparameters.pickle", "wb") as f:
    #     pickle.dump(searcher.status["best"], f)


    # for center solution
#     with open("savedparameters.pickle", "wb") as f:
#         pickle.dump(searcher.status["center"], f)
    
    # for saving both the center and the best solutions
    with open("savedparameters.pickle", "wb") as f:
        pickle.dump(
            {
                "best": searcher.status["best"],
                "center": searcher.status["center"],
            },
            f,
        )


    best_solution = evotorch_problem.status["best"]                                                                      # Best solution found
    t.save(best_solution, 'optimum.pt')

    # Save the best evaluations list to a text file
    with open('best_evaluations.txt', 'w') as file:
        for evals in best_evals_list:
            file.write(f"{evals}\n")

    # Save the center evaluations list to a text file
    with open('center_evaluations.txt', 'w') as file:
        for evals in center_evals_list:
            file.write(f"{evals}\n")


    ################################################ Plotting gantt chart ##############################################
    x = best_solution.values

    n = 0

    simulator_net_dict = evotorch_problem.net_dict
    net_dict = {}

    for key, value in simulator_net_dict.items():
        net_dict[key] = x[n:n + value.numel()].reshape(value.shape)
        n += value.numel()

    agent = evotorch_problem.simulator.from_params(net_dict, cfg=evotorch_problem.cfg)
    loss = agent.fitness()
    agent.env.render()
    print("In policy optimiser,gantt chart,loss :", loss)
    violation_check = agent.env.spinning_reserve_constraint()  # To check if spinning reserve constraint is not violated
    Electricity_cost = sum(agent.env.total_cost_electricity)  # Electricity cost
    SP_profit = sum(agent.env.total_cost_spinning_reserve)  # Spinning reserve profit

    with open("loss_values.txt", "a") as file:
        # Write the loss value to the file
        file.write("In policy optimiser, gantt chart, loss: " + str(loss) + "\n")
        file.write("violation_check: " + str(violation_check) + "\n")  #
        file.write("Electricity_cost: " + str(Electricity_cost) + "\n")
        file.write("SP_profit: " + str(SP_profit) + "\n")
    
    consumption = agent.env.maxaux
    schedule = [agent.env.stage1_copy, agent.env.stage2_copy, agent.env.stage3_copy]
    cost = agent.env.total_costs
    
    with open("electricity_consumption.txt", "a") as file:
        file.write("The electricity consumption list is: " + str(consumption) + "\n")

    with open("schedule.txt", "a") as file:
        file.write("The schedule is: " + str(schedule) + "\n")

    with open("electricity_cost.txt", "a") as file:
        file.write("The electricity cost list is: " + str(cost) + "\n")
    
    

###################################### Running policy_optimiser ########################################################
if __name__ == '__main__':
    foo()


