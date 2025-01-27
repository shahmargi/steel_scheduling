
############################################ consumption plot ##########################################################
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import rcParams
# import csv
# #
# # Set font fallback to Calibri
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Calibri']
#
# # Electricity consumption data
# electricity_consumption = [
#     34.876508939, 37.982300209, 38.944353894, 40.01561147, 42.108414404, 33.99497869266667,
#     40.639866551, 42.813400907, 41.285361381, 40.557199613, 39.511205623, 29.44758399966667,
#     33.92671352666666, 32.525399416666666, 33.53598533, 34.05348987666666, 33.74909591,
#     20.41286779033333, 35.047683322666664, 34.850481332, 35.54075620966667, 35.05096860766666,
#     33.53068813, 21.933322296666674, 24.49691845, 25.771872186666663, 28.639696766666667,
#     31.38491316, 30.47015229, 14.566817773333343, 26.16265981, 26.11498497, 23.01894769,
#     21.51008506, 16.64445067, 0, 7.021443490000003, 5.759053629999997, 3.0, 3.5464523999999997,
#     4.833316359999998, 0, 14.232160140000001, 20.22979168, 25.21973208, 28.600595939999998,
#     32.67623389, 5.78373146999999, 13.459112209, 14.960640249, 16.26495780866667, 14.937796058,
#     14.417116589, 0, 10.01508939, 11.13198441, 10.48817043, 7.213649633333333, 9.19603474, 0,
#     10.94538532, 9.52614221, 9.43973157, 9.008196236666668, 11.40904345, 0, 12.963984957666668,
#     10.87728569, 13.292428564, 13.653045892, 13.018985891, 0, 30.92153517, 31.0038965,
#     30.447282639999997, 31.582089676666662, 32.89089798, 3.7474999566666565, 31.47774156,
#     33.36443743, 35.70091173, 36.199239375666664, 37.173508888, 9.479099814666657, 0, 0, 0, 0,
#     0.155325218, 0, 0, 0, 0, 0, 0.1524304316666667, 0.5
# ]
#
# prices = [33.28, 33.28, 33.28, 33.28, 30.0, 30.0, 30.0, 30.0, 29.15, 29.15, 29.15, 29.15, 28.49, 28.49, 28.49, 28.49, 34.66, 34.66, 34.66, 34.66, 50.01, 50.01, 50.01, 50.01, 71.52, 71.52, 71.52, 71.52, 77.94, 77.94, 77.94, 77.94, 81.97, 81.97, 81.97, 81.97, 87.9, 87.9, 87.9, 87.9, 92.76, 92.76, 92.76, 92.76, 94.98, 94.98, 94.98, 94.98, 92.31, 92.31, 92.31, 92.31, 90.03, 90.03, 90.03, 90.03, 90.09, 90.09, 90.09, 90.09, 87.44, 87.44, 87.44, 87.44, 85.37, 85.37, 85.37, 85.37, 79.97, 79.97, 79.97, 79.97, 79.92, 79.92, 79.92, 79.92, 77.83, 77.83, 77.83, 77.83, 76.28, 76.28, 76.28, 76.28, 65.06, 65.06, 65.06, 65.06, 53.07, 53.07, 53.07, 53.07, 34.16, 34.16, 34.16, 34.16]
#
#
# # fig, ax = plt.subplots(figsize=(14, 8))
#
# fig, ax = plt.subplots(figsize=(12, 6))
#
#
# bars = ax.bar(range(len(electricity_consumption)), electricity_consumption, color='green')
#
#
# # ax.set_xlabel('Time slots', fontsize=22, fontname='Calibri')
# # ax.set_ylabel('Electricity consumption (MWh)', fontsize=27, fontname='Calibri')
#
# ax.set_xlabel('Time slots', fontsize=30)
# ax.set_ylabel('Electricity\nconsumption (MWh)', fontsize=30)
#
#
# ax.grid(axis='y', linestyle='--', alpha=0.7)
# tick_positions = list(range(0, len(electricity_consumption), 8))
# if len(electricity_consumption) - 1 not in tick_positions:
#     tick_positions.append(len(electricity_consumption) - 1)
# ax.set_xticks(tick_positions)
# ax.tick_params(axis='both', which='major', labelsize=28)
#
#
#
#
# ax2 = ax.twinx()
# ax2.plot(prices, color='black', marker='o', linestyle='-', label='Prices', linewidth=5)
#
# # ax2.set_ylabel('Electricity prices(£/MWh)', fontsize=27, color='black', fontname='Calibri')
# ax2.set_ylabel('Electricity prices(£/MWh)', fontsize=28, color='black')
# ax2.tick_params(axis='both', which='major', labelsize=28)
#
#
#
# ax.spines['top'].set_visible(False)
# ax2.spines['top'].set_visible(False)
#
# midpoint = 0.5
# distance_between_legends = 0.1
#
#
# left_legend_position = midpoint - 0.4
#
#
# right_legend_position = midpoint + 0.2
#
#
# legend1 = ax.legend(['Electricity consumption'], loc='upper left', fontsize=28, prop={'family': 'Calibri'}, bbox_to_anchor=(left_legend_position, -0.15), frameon=False)
# legend2 = ax2.legend(['Electricity price'], loc='upper left', fontsize=28, prop={'family': 'Calibri'}, bbox_to_anchor=(right_legend_position, -0.15), frameon=False)
#
# for text in legend1.get_texts():
#     text.set_fontsize(28)
#
# for text in legend2.get_texts():
#     text.set_fontsize(28)
#
# plt.tight_layout()
# # plt.savefig('consumption_plot.png', dpi=500)  # Save the plot with higher resolution (300 DPI)
# plt.savefig('consumption_plot.svg', format='svg', bbox_inches='tight')
# plt.show()
# plt.close()
#
#
#
#
# ############################################ training plot ##########################################################
#
# with open('training_progress.txt', 'r') as file:
#
#     reader = csv.DictReader(file, delimiter='\t')
#
#     best_eval_list = []
#
#     for row in reader:
#         best_eval_list.append(float(row['best_eval']))
#
#
# iterations = list(range(1, len(best_eval_list) + 1))
#
#
# plt.figure(figsize=(12, 6))
#
#
# plt.plot(iterations, best_eval_list, linestyle='-', color='#FF1493', linewidth=6)
#
#
# # plt.xlabel('Iterations', fontsize=27, fontname='Calibri')
# # plt.ylabel('Electricity cost (£)', fontsize=27, fontname='Calibri')
#
# plt.xlabel('Iterations', fontsize=27)
# plt.ylabel('Electricity cost (£)', fontsize=27)
#
#
# # plt.xticks(fontsize=27, fontname='Calibri')
# # plt.yticks(fontsize=27, fontname='Calibri')
#
# plt.xticks(fontsize=27)
# plt.yticks(fontsize=27)
#
#
# plt.tight_layout()
# #plt.savefig('MILP_DRL.png', dpi=500)  # Save the plot with higher resolution (300 DPI)
# plt.savefig('MILP_DRL_training_plot.svg', format='svg', bbox_inches='tight')
# plt.show()
# plt.close()
#
# ############################################ electricity price plot  ###################################################
#Data
# energy_price = [33.28, 30.00, 29.15, 28.49, 34.66, 50.01,
#                 71.52, 77.94, 81.97, 87.90, 92.76, 94.98,
#                 92.31, 90.03, 90.09, 87.44, 85.37, 79.97,
#                 79.92, 77.83, 76.28, 65.06, 53.07, 34.16]
#
# sp_price = [3.28, 3.00, 2.15, 2.49, 3.66, 5.01,
#             7.52, 7.94, 8.97, 8.90, 9.76, 9.98,
#             9.31, 9.03, 9.09, 8.44, 8.37, 7.97,
#             7.92, 7.83, 7.28, 6.06, 5.07, 3.16]
#
# # Plotting
# plt.figure(figsize=(12, 6))
# plt.plot(energy_price, label='Energy price', marker='o', linestyle='-', color='b')
# plt.plot(sp_price, label='Reserve price', marker='x', linestyle='--', color='r')
#
# # Title and labels
# plt.xlabel('Hour', fontsize=22)
# plt.ylabel('Price (£/MWh)', fontsize=22)
# plt.xticks(range(len(energy_price)), fontsize=21)  # Ensure all x-axis ticks are displayed
# plt.yticks(fontsize=22)
#
# # Grid and legend
# plt.legend(fontsize=22, loc='upper left')
#
# # Display plot
# plt.tight_layout()
# plt.savefig('price.png', dpi=500)
# plt.show()
# plt.close()
#
# ############################################ generation  plot  #########################################################
#
# generation = [7.623491061, 4.517699791, 3.555646106, 2.48438853, 0.391585596, 1.421687974, 1.860133449,
#                 0.686599093,
#                            2.214638619, 2.942800387, 3.988794377, 6.969082667, 9.23995314, 10.64126725, 10.46401467,
#                            10.11317679,
#                            10.25090409, 9.420465543, 8.118983344, 8.649518668, 8.125910457, 8.615698059,
#                            10.46931187,
#                            14.98334437,
#                            19.50308155, 17.39479448, 14.5269699, 12.61508684, 14.02984771, 15.76651556, 17.33734019,
#                            17.38501503,
#                            19.98105231, 21.98991494, 27.85554933, 35.53486477, 36.97855651, 37.74094637, 40,
#                            39.9535476,
#                            39.66668364, 37.53965263, 29.76783986, 23.27020832, 17.78026792, 14.89940406,
#                            11.82376611,
#                            10.21626853,
#                            9.290887791, 7.289359751, 5.651708858, 6.812203942, 8.832883411, 10.47705394,
#                            12.23491061,
#                            11.11801559,
#                            11.76182957, 14.3696837, 12.55396526, 11.41751133, 11.80461468, 13.22385779, 12.81026843,
#                            12.90847043,
#                            10.34095655, 9.97952427, 8.952681709, 10.87271431, 8.957571436, 8.596954108, 8.731014109,
#                            9.608720012,
#                            11.57846483, 12.4961035, 13.05271736, 11.58457699, 10.10910202, 10.91916671, 11.02225844,
#                            10.63556257,
#                            8.29908827, 7.467427291, 6.326491112, 5.687566852, 4.413793103, 6.339530383, 5.93286813,
#                            2.206081597,
#                            1.344674782, 2.710130902, 2.766362757, 2.951764886, 3.475373096, 1.560230225,
#                            0.514236235, 0]
#
# # Plotting
# plt.figure(figsize=(12, 6))
# plt.plot(generation , label='Wind', marker='o', linestyle='-', color='g')
#
#
# # Title and labels
# plt.xlabel('Time slots', fontsize=22)
# plt.ylabel('Wind generation(MW)', fontsize=22)
# # plt.xticks(range(len(generation)), fontsize=21)  # Ensure all x-axis ticks are displayed
# plt.yticks(fontsize=22)
# plt.xticks(fontsize=22)
#
# # Grid and legend
# plt.legend(fontsize=22, loc='upper left')
#
# # Display plot
# plt.tight_layout()
# plt.savefig('generation.png', dpi=500)
# plt.show()
# plt.close()
#
# ############################################ gantt  plot  #########################################################
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np
#
# # Define your data structure here. For this example, let's create a simplified version.
# # Replace this with your actual data structure.
# class ProductionTask:
#     def __init__(self, id, start_time, end_time):
#         self.id = id
#         self.start_time = start_time
#         self.end_time = end_time
#
# class ProductionEquipment:
#     def __init__(self, production_tasks):
#         self.production_tasks = production_tasks
#
# class Stage:
#     def __init__(self, production_equipment):
#         self.production_equipment = production_equipment
#
# class GanttChartGenerator:
#     def __init__(self, stage1, stage2, stage3, periods):
#         self.stage1_copy = stage1
#         self.stage2_copy = stage2
#         self.stage3_copy = stage3
#         self.periods = periods
#
#     def generate_gantt_chart(self):
#         # Create a dictionary to store the colors for each task ID
#         task_colors = {}
#
#         EAF1 = self.stage1_copy.production_equipment[0]
#         EAF2 = self.stage1_copy.production_equipment[1]
#         AOD1 = self.stage2_copy.production_equipment[0]
#         AOD2 = self.stage2_copy.production_equipment[1]
#         LF1 = self.stage3_copy.production_equipment[0]
#         LF2 = self.stage3_copy.production_equipment[1]
#
#         # Assign colors to each unique task ID
#         task_ids = range(1, 25)
#         colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gray', 'aquamarine',
#                   'skyblue', 'navy', 'orange', 'purple', 'lime', 'pink', 'teal', 'gold',
#                   'silver', 'olive', 'maroon', 'salmon', 'peru', 'plum', 'darkgreen']
#
#         for i, task_id in enumerate(task_ids):
#             color = colors[i % len(colors)]
#             task_colors[task_id] = color
#
#         # Create a figure and axis
#         fig, ax = plt.subplots(figsize=(14, 6))
#
#         ax.set_xlim(0, int(self.periods))
#         ax.set_xticks(np.arange(0, int(self.periods) + 1, 4))
#
#         # Set the y-axis labels
#         equipment_labels = ['EAF1', 'EAF2', 'AOD1', 'AOD2', 'LF1', 'LF2']
#         ax.set_yticks(range(len(equipment_labels)))
#         ax.set_yticklabels(equipment_labels, fontsize=30)  # Set font size for y-axis labels
#         ax.set_ylim(-0.5, len(equipment_labels) - 0.5)
#
#         ax.set_xticklabels(range(0, int(self.periods) + 1, 4), fontsize=24, rotation=45, ha='right')  # Set font size, rotation, and alignment for x-axis labels
#
#         # Plot the tasks for each equipment
#         legend_patches = []  # List to store legend patches
#         processed_task_ids = set()
#
#         for i, equipment in enumerate([EAF1, EAF2, AOD1, AOD2, LF1, LF2]):
#             if equipment.production_tasks is not None:
#                 for task in equipment.production_tasks:
#                     color = task_colors.get(task.id, 'gray')  # Default to 'gray' if color is not found
#                     start_time = task.start_time
#                     end_time = task.end_time
#
#                     # Plot a horizontal bar for each task
#                     ax.barh(i, end_time - start_time, left=start_time, height=0.2, color=color)
#
#                     # Add legend patch for the task if it doesn't exist in the list
#                     if task.id not in processed_task_ids:
#                         legend_patch = mpatches.Patch(color=color, label=f'Heat {task.id}')
#                         legend_patches.append(legend_patch)
#                         processed_task_ids.add(task.id)
#
#         ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)  # Decrease font size of legend
#         fig.subplots_adjust(bottom=0.2)
#         fig.tight_layout()  # Adjusts the subplot parameters to fit the legend properly
#
#         # Set the chart title and labels
#         plt.xlabel('Time slots', fontsize=30)  # Set font size for x-axis label
#
#         # Rotate the y-axis labels for better readability
#         plt.tick_params(axis='y', rotation=0)
#
#         # Remove grid lines
#         ax.grid(False)
#
#         plt.savefig('rl_gantt_chart.svg', format='svg', bbox_inches='tight')
#         plt.show()
#         plt.close()
#
# # Example data
# eaf1_tasks =  [
#               ProductionTask(id=7, start_time=0, end_time=5),
#               ProductionTask(id=8, start_time=6, end_time=11),
#               ProductionTask(id=24, start_time=12, end_time=17),
#               ProductionTask(id=14, start_time=18, end_time=23),
#               ProductionTask(id=15, start_time=24, end_time=29),
#               ProductionTask(id=16, start_time=30, end_time=35),
#               ProductionTask(id=17, start_time=36, end_time=41),
#               ProductionTask(id=21, start_time=42, end_time=47),
#               ProductionTask(id=22, start_time=48, end_time=53),
#               ProductionTask(id=23, start_time=54, end_time=59),
#               ProductionTask(id=1, start_time=60, end_time=65),
#               ProductionTask(id=2, start_time=66, end_time=71),
#               ProductionTask(id=3, start_time=72, end_time=77),
#               ProductionTask(id=4, start_time=78, end_time=83),
#               ]
#
#
#
# eaf2_tasks = [
#               ProductionTask(id=12, start_time=0, end_time=5),
#               ProductionTask(id=9,  start_time=6, end_time=11),
#               ProductionTask(id=11, start_time=12, end_time=17),
#               ProductionTask(id=10, start_time=18, end_time=23),
#               ProductionTask(id=5, start_time=24, end_time=29),
#               ProductionTask(id=6, start_time=30, end_time=35),
#               ProductionTask(id=13, start_time=36, end_time=41),
#               ProductionTask(id=19, start_time=42, end_time=47),
#               ProductionTask(id=20, start_time=72, end_time=77),
#               ProductionTask(id=18, start_time=78, end_time=83),
#               ]
#
#
#
# aod1_tasks = [
#               ProductionTask(id=7,  start_time=7, end_time=12),
#               ProductionTask(id=8, start_time=6, end_time=13),
#               ProductionTask(id=11, start_time=19, end_time=25),
#               ProductionTask(id=10, start_time=26, end_time=32),
#               ProductionTask(id=15, start_time=33, end_time=38),
#               ProductionTask(id=16, start_time=39, end_time=44),
#               ProductionTask(id=13, start_time=45, end_time=50),
#               ProductionTask(id=19, start_time=51, end_time=57),
#               ProductionTask(id=22, start_time=58, end_time=63),
#               ProductionTask(id=1, start_time=67, end_time=71),
#               ProductionTask(id=2, start_time=73, end_time=77),
#               ProductionTask(id=20, start_time=79, end_time=85),
#               ProductionTask(id=4, start_time=86, end_time=90),
#               ]
#
#
#
# aod2_tasks = [
#               ProductionTask(id=12,  start_time=7, end_time=13),
#               ProductionTask(id=9, start_time=14, end_time=20),
#               ProductionTask(id=24, start_time=21, end_time=26),
#               ProductionTask(id=14, start_time=27, end_time=32),
#               ProductionTask(id=5, start_time=33, end_time=38),
#               ProductionTask(id=6, start_time=39, end_time=44),
#               ProductionTask(id=17, start_time=45, end_time=50),
#               ProductionTask(id=21, start_time=52, end_time=57),
#               ProductionTask(id=23, start_time=61, end_time=66),
#               ProductionTask(id=3, start_time=79, end_time=83),
#               ProductionTask(id=18, start_time=85, end_time=91),
#              ]
#
#
# lf1_tasks = [
#               ProductionTask(id=7,  start_time=14, end_time=15),
#               ProductionTask(id=8, start_time=20, end_time=21),
#               ProductionTask(id=9, start_time=22, end_time=24),
#               ProductionTask(id=11, start_time=27, end_time=29),
#               ProductionTask(id=10, start_time=34, end_time=36),
#               ProductionTask(id=15, start_time=40, end_time=41),
#               ProductionTask(id=16, start_time=46, end_time=47),
#               ProductionTask(id=13, start_time=52, end_time=53),
#               ProductionTask(id=19, start_time=59, end_time=61),
#               ProductionTask(id=22, start_time=65, end_time=66),
#               ProductionTask(id=23, start_time=68, end_time=69),
#               ProductionTask(id=1, start_time=73, end_time=75),
#               ProductionTask(id=2, start_time=79, end_time=81),
#               ProductionTask(id=3, start_time=85, end_time=87),
#               ProductionTask(id=4, start_time=92, end_time=94),
#               ]
#
#
# lf2_tasks = [
#               ProductionTask(id=12,  start_time=15, end_time=17),
#               ProductionTask(id=24, start_time=28, end_time=29),
#               ProductionTask(id=14, start_time=34, end_time=35),
#               ProductionTask(id=5, start_time=40, end_time=42),
#               ProductionTask(id=6, start_time=46, end_time=48),
#               ProductionTask(id=17, start_time=52, end_time=53),
#               ProductionTask(id=21, start_time=59, end_time=60),
#               ProductionTask(id=20, start_time=87, end_time=88),
#               ProductionTask(id=18, start_time=93, end_time=95),
#              ]
#
#
# # Create production equipment instances
# EAF1 = ProductionEquipment(eaf1_tasks)
# EAF2 = ProductionEquipment(eaf2_tasks)
# AOD1 = ProductionEquipment(aod1_tasks)
# AOD2 = ProductionEquipment(aod2_tasks)
# LF1 = ProductionEquipment(lf1_tasks)
# LF2 = ProductionEquipment(lf2_tasks)
#
# # Create stage instances
# stage1 = Stage([EAF1, EAF2])
# stage2 = Stage([AOD1, AOD2])
# stage3 = Stage([LF1, LF2])
#
# # Example periods
# periods = 96  # Adjust this according to your data
#
# generator = GanttChartGenerator(stage1, stage2, stage3, periods)
#
# # Call the generate_gantt_chart method
# generator.generate_gantt_chart()
#
#
#
# ########################################### mip_gantt chart ############################################################
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np
#
# # Define your data structure here. For this example, let's create a simplified version.
# # Replace this with your actual data structure.
# class ProductionTask:
#     def __init__(self, id, start_time, end_time):
#         self.id = id
#         self.start_time = start_time
#         self.end_time = end_time
#
# class ProductionEquipment:
#     def __init__(self, production_tasks):
#         self.production_tasks = production_tasks
#
# class Stage:
#     def __init__(self, production_equipment):
#         self.production_equipment = production_equipment
#
# class GanttChartGenerator:
#     def __init__(self, stage1, stage2, stage3, periods):
#         self.stage1_copy = stage1
#         self.stage2_copy = stage2
#         self.stage3_copy = stage3
#         self.periods = periods
#
#     def generate_gantt_chart(self):
#         # Create a dictionary to store the colors for each task ID
#         task_colors = {}
#
#         EAF1 = self.stage1_copy.production_equipment[0]
#         EAF2 = self.stage1_copy.production_equipment[1]
#         AOD1 = self.stage2_copy.production_equipment[0]
#         AOD2 = self.stage2_copy.production_equipment[1]
#         LF1 = self.stage3_copy.production_equipment[0]
#         LF2 = self.stage3_copy.production_equipment[1]
#
#         # Assign colors to each unique task ID
#         task_ids = range(1, 25)
#         colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gray', 'aquamarine',
#                   'skyblue', 'navy', 'orange', 'purple', 'lime', 'pink', 'teal', 'gold',
#                   'silver', 'olive', 'maroon', 'salmon', 'peru', 'plum', 'darkgreen']
#
#         for i, task_id in enumerate(task_ids):
#             color = colors[i % len(colors)]
#             task_colors[task_id] = color
#
#         # Create a figure and axis
#         fig, ax = plt.subplots(figsize=(14, 6))
#
#         ax.set_xlim(0, int(self.periods))
#         ax.set_xticks(np.arange(0, int(self.periods) + 1, 4))
#
#         # Set the y-axis labels
#         equipment_labels = ['EAF1', 'EAF2', 'AOD1', 'AOD2', 'LF1', 'LF2']
#         ax.set_yticks(range(len(equipment_labels)))
#         ax.set_yticklabels(equipment_labels, fontsize=30)  # Set font size for y-axis labels
#         ax.set_ylim(-0.5, len(equipment_labels) - 0.5)
#
#         ax.set_xticklabels(range(0, int(self.periods) + 1, 4), fontsize=24, rotation=45, ha='right')  # Set font size, rotation, and alignment for x-axis labels
#
#         # Plot the tasks for each equipment
#         legend_patches = []  # List to store legend patches
#         processed_task_ids = set()
#
#         for i, equipment in enumerate([EAF1, EAF2, AOD1, AOD2, LF1, LF2]):
#             if equipment.production_tasks is not None:
#                 for task in equipment.production_tasks:
#                     color = task_colors.get(task.id, 'gray')  # Default to 'gray' if color is not found
#                     start_time = task.start_time
#                     end_time = task.end_time
#
#                     # Plot a horizontal bar for each task
#                     ax.barh(i, end_time - start_time, left=start_time, height=0.2, color=color)
#
#                     # Add legend patch for the task if it doesn't exist in the list
#                     if task.id not in processed_task_ids:
#                         legend_patch = mpatches.Patch(color=color, label=f'Heat {task.id}')
#                         legend_patches.append(legend_patch)
#                         processed_task_ids.add(task.id)
#
#         ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)  # Decrease font size of legend
#         fig.subplots_adjust(bottom=0.2)
#         fig.tight_layout()  # Adjusts the subplot parameters to fit the legend properly
#
#         # Set the chart title and labels
#         plt.xlabel('Time slots', fontsize=30)  # Set font size for x-axis label
#
#         # Rotate the y-axis labels for better readability
#         plt.tick_params(axis='y', rotation=0)
#
#         # Remove grid lines
#         ax.grid(False)
#
#         # plt.savefig('mip_gantt_chart.svg', bbox_inches='tight',dpi=500)
#         plt.savefig('mip_gantt_chart.svg', format='svg', bbox_inches='tight')
#         plt.show()
#         plt.close()
#
# # Example data
# eaf1_tasks =  [
#               ProductionTask(id=9, start_time=0, end_time=5),
#               ProductionTask(id=10, start_time=6, end_time=11),
#               ProductionTask(id=5, start_time=12, end_time=17),
#               ProductionTask(id=13, start_time=18, end_time=23),
#               ProductionTask(id=1, start_time=24, end_time=29),
#               ProductionTask(id=14, start_time=32, end_time=38),
#               ProductionTask(id=3, start_time=38, end_time=43),
#               ProductionTask(id=19, start_time=48, end_time=53),
#               ProductionTask(id=6, start_time=54, end_time=59),
#               ProductionTask(id=23, start_time=60, end_time=65),
#               ProductionTask(id=18, start_time=67, end_time=72),
#               ProductionTask(id=2, start_time=73, end_time=78),
#               ProductionTask(id=17, start_time=79, end_time=84),
#               ]
#
#
#
# eaf2_tasks = [
#               ProductionTask(id=12, start_time=0, end_time=5),
#               ProductionTask(id=11,  start_time=6, end_time=11),
#               ProductionTask(id=16, start_time=12, end_time=17),
#               ProductionTask(id=15, start_time=18, end_time=23),
#               ProductionTask(id=21, start_time=24, end_time=29),
#               ProductionTask(id=8, start_time=30, end_time=35),
#               ProductionTask(id=7, start_time=36, end_time=41),
#               ProductionTask(id=4, start_time=42, end_time=47),
#               ProductionTask(id=22, start_time=66, end_time=71),
#               ProductionTask(id=24, start_time=72, end_time=77),
#               ProductionTask(id=20, start_time=78, end_time=83),
#               ]
#
#
#
# aod1_tasks = [
#               ProductionTask(id=9,  start_time=7, end_time=13),
#               ProductionTask(id=10, start_time=14, end_time=20),
#               ProductionTask(id=5, start_time=21, end_time=26),
#               ProductionTask(id=13, start_time=27, end_time=32),
#               ProductionTask(id=1, start_time=33, end_time=37),
#               ProductionTask(id=8, start_time=38, end_time=43),
#               ProductionTask(id=7, start_time=46, end_time=51),
#               ProductionTask(id=3, start_time=53, end_time=57),
#               ProductionTask(id=6, start_time=62, end_time=67),
#               ProductionTask(id=23, start_time=68, end_time=73),
#               ProductionTask(id=18, start_time=74, end_time=80),
#               ProductionTask(id=2, start_time=81, end_time=85),
#               ProductionTask(id=17, start_time=86, end_time=91),
#               ]
#
#
#
# aod2_tasks = [
#               ProductionTask(id=12,  start_time=7, end_time=13),
#               ProductionTask(id=11, start_time=14, end_time=20),
#               ProductionTask(id=16, start_time=21, end_time=26),
#               ProductionTask(id=15, start_time=27, end_time=32),
#               ProductionTask(id=21, start_time=37, end_time=42),
#               ProductionTask(id=14, start_time=46, end_time=51),
#               ProductionTask(id=4, start_time=52, end_time=56),
#               ProductionTask(id=19, start_time=65, end_time=71),
#               ProductionTask(id=22, start_time=73, end_time=78),
#               ProductionTask(id=24, start_time=79, end_time=84),
#               ProductionTask(id=20, start_time=85, end_time=91),
#              ]
#
#
# lf1_tasks = [
#               ProductionTask(id=9,  start_time=15, end_time=17),
#               ProductionTask(id=10, start_time=22, end_time=24),
#               ProductionTask(id=5, start_time=29, end_time=31),
#               ProductionTask(id=13, start_time=37, end_time=38),
#               ProductionTask(id=1, start_time=41, end_time=43),
#               ProductionTask(id=8, start_time=47, end_time=48),
#               ProductionTask(id=7, start_time=53, end_time=54),
#               ProductionTask(id=3, start_time=64, end_time=66),
#               ProductionTask(id=6, start_time=84, end_time=86),
#               ProductionTask(id=2, start_time=87, end_time=89),
#               ProductionTask(id=18, start_time=90, end_time=92),
#               ProductionTask(id=17, start_time=93, end_time=94),
#               ]
#
#
# lf2_tasks = [
#               ProductionTask(id=12,  start_time=15, end_time=17),
#               ProductionTask(id=11, start_time=22, end_time=24),
#               ProductionTask(id=16, start_time=37, end_time=38),
#               ProductionTask(id=15, start_time=41, end_time=42),
#               ProductionTask(id=21, start_time=47, end_time=48),
#               ProductionTask(id=14, start_time=53, end_time=54),
#               ProductionTask(id=4, start_time=65, end_time=67),
#               ProductionTask(id=23, start_time=84, end_time=85),
#               ProductionTask(id=24, start_time=86, end_time=87),
#               ProductionTask(id=19, start_time=88, end_time=90),
#               ProductionTask(id=22, start_time=91, end_time=92),
#               ProductionTask(id=20, start_time=93, end_time=94),
#              ]
#
#
# # Create production equipment instances
# EAF1 = ProductionEquipment(eaf1_tasks)
# EAF2 = ProductionEquipment(eaf2_tasks)
# AOD1 = ProductionEquipment(aod1_tasks)
# AOD2 = ProductionEquipment(aod2_tasks)
# LF1 = ProductionEquipment(lf1_tasks)
# LF2 = ProductionEquipment(lf2_tasks)
#
# # Create stage instances
# stage1 = Stage([EAF1, EAF2])
# stage2 = Stage([AOD1, AOD2])
# stage3 = Stage([LF1, LF2])
#
# # Example periods
# periods = 96  # Adjust this according to your data
#
# generator = GanttChartGenerator(stage1, stage2, stage3, periods)
#
# # Call the generate_gantt_chart method
# generator.generate_gantt_chart()


############################################## mip spinning reserve ####################################################
# import matplotlib.pyplot as plt
# import numpy as np
#
# mip_reserve = [
#     68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
#     68, 68, 68, 68, 68, 68, 68, 68, 68, 34, 34, 34, 34, 68, 68, 68, 68, 68, 68, 68,
#     68, 68, 68, 68, 68, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
#     34, 34, 34, 34, 34, 34, 34, 34, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
#     68, 68, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# ]
#
# # Plotting
# plt.figure(figsize=(12, 6))
# y_values = np.arange(len(mip_reserve))
# plt.step(y_values, mip_reserve, label='STOR provision', color='b')
#
# # Title and labels
# plt.xlabel('Time slot [15 min]', fontsize=22)
# plt.ylabel('STOR (MW)', fontsize=22)
#
# # Set x-axis ticks
# x_ticks = np.arange(0, len(mip_reserve), 10)  # Adjust the step size as needed
# plt.xticks(x_ticks, fontsize=16)
# plt.yticks(fontsize=16)
#
# # Grid and legend
# plt.legend(fontsize=18, loc='lower left')
#
# # Display plot
# plt.tight_layout()
# plt.savefig('mip_reserve.png', dpi=500)
# plt.show()
# plt.close()
#
# ############################################## RL spinning reserve  ####################################################
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# rl_reserve= [68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
# 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
# 68, 68, 68, 68, 68, 68, 68, 68, 68, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
# 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 68, 68, 68, 68, 68, 68, 68,
# 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
# # Plotting
# plt.figure(figsize=(12, 6))
# y_values = np.arange(len(rl_reserve))
# plt.step(y_values, rl_reserve, label='STOR provision', color='b')
#
# # Title and labels
# plt.xlabel('Time slot [15 min]', fontsize=22)
# plt.ylabel('STOR (MW)', fontsize=22)
#
# # Set x-axis ticks
# x_ticks = np.arange(0, len(rl_reserve), 10)  # Adjust the step size as needed
# plt.xticks(x_ticks, fontsize=16)
# plt.yticks(fontsize=16)
#
# # Grid and legend
# plt.legend(fontsize=18, loc='lower left')
#
# # Display plot
# plt.tight_layout()
# plt.savefig('rl_reserve.png', dpi=500)
# plt.show()
# plt.close()
########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np

mip_reserve = [
    68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
    68, 68, 68, 68, 68, 68, 68, 68, 68, 34, 34, 34, 34, 68, 68, 68, 68, 68, 68, 68,
    68, 68, 68, 68, 68, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
    68, 68, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

rl_reserve = [
    68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
    68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
    68, 68, 68, 68, 68, 68, 68, 68, 68, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 68, 68, 68, 68, 68, 68, 68, 68, 68,
    68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

# Truncate rl_reserve to 95 elements
rl_reserve = rl_reserve[:95]

# Plotting
plt.figure(figsize=(12, 6))
y_values = np.arange(len(mip_reserve))

# MIP reserve plot
plt.step(y_values, mip_reserve, label='MILP Spinning reserve provision', color='b', linewidth=5)

# RL reserve plot
plt.step(y_values, rl_reserve, label='RL Spinning reserve provision', color='r', linestyle='--', linewidth=5)

# Title and labels
plt.xlabel('Time slot [15 min]', fontsize=28)
plt.ylabel('Spinning reserve (MW)', fontsize=28) #22

# Set x-axis ticks
x_ticks = np.arange(0, len(mip_reserve), 10)  # Adjust the step size as needed
plt.xticks(x_ticks, fontsize=27) #16
plt.yticks(fontsize=27) #16

# Grid and legend
plt.legend(fontsize=26, loc='lower left') #18

# Display plot
plt.tight_layout()
plt.savefig('combined_reserve.svg', format='svg', bbox_inches='tight')
plt.show()
plt.close()

########################### sub plots for input data ###################################################################

# import matplotlib.pyplot as plt
#
# # Data for energy price
# energy_price = [33.28, 30.00, 29.15, 28.49, 34.66, 50.01,
#                 71.52, 77.94, 81.97, 87.90, 92.76, 94.98,
#                 92.31, 90.03, 90.09, 87.44, 85.37, 79.97,
#                 79.92, 77.83, 76.28, 65.06, 53.07, 34.16]
#
# # Extend energy_price to match the number of time slots
# extended_energy_price = []
# for price in energy_price:
#     extended_energy_price.extend([price] * int(60 / 15))
#
# # Data for wind generation
# generation = [7.623491061, 4.517699791, 3.555646106, 2.48438853, 0.391585596, 1.421687974, 1.860133449,
#                 0.686599093, 2.214638619, 2.942800387, 3.988794377, 6.969082667, 9.23995314, 10.64126725,
#                 10.46401467, 10.11317679, 10.25090409, 9.420465543, 8.118983344, 8.649518668, 8.125910457,
#                 8.615698059, 10.46931187, 14.98334437, 19.50308155, 17.39479448, 14.5269699, 12.61508684,
#                 14.02984771, 15.76651556, 17.33734019, 17.38501503, 19.98105231, 21.98991494, 27.85554933,
#                 35.53486477, 36.97855651, 37.74094637, 40, 39.9535476, 39.66668364, 37.53965263, 29.76783986,
#                 23.27020832, 17.78026792, 14.89940406, 11.82376611, 10.21626853, 9.290887791, 7.289359751,
#                 5.651708858, 6.812203942, 8.832883411, 10.47705394, 12.23491061, 11.11801559, 11.76182957,
#                 14.3696837, 12.55396526, 11.41751133, 11.80461468, 13.22385779, 12.81026843, 12.90847043,
#                 10.34095655, 9.97952427, 8.952681709, 10.87271431, 8.957571436, 8.596954108, 8.731014109,
#                 9.608720012, 11.57846483, 12.4961035, 13.05271736, 11.58457699, 10.10910202, 10.91916671,
#                 11.02225844, 10.63556257, 8.29908827, 7.467427291, 6.326491112, 5.687566852, 4.413793103,
#                 6.339530383, 5.93286813, 2.206081597, 1.344674782, 2.710130902, 2.766362757, 2.951764886,
#                 3.475373096, 1.560230225, 0.514236235, 0]
#
# # Plotting
# fig, ax1 = plt.subplots(figsize=(12, 6))
#
# # Plot energy price on the left y-axis
# ax1.plot(extended_energy_price, label='Electricity Price', marker='o', linestyle='-', color='b')
# ax1.set_xlabel('Time slots', fontsize=22)
# ax1.set_ylabel('Price (£/MWh)', fontsize=22)
# ax1.tick_params(axis='x', labelsize=22)
# ax1.tick_params(axis='y', labelsize=22)
#
# # Create a second y-axis for the wind generation
# ax2 = ax1.twinx()
# ax2.plot(generation, label='Wind Generation', marker='o', linestyle='-', color='g')
# ax2.set_ylabel('Wind Generation (MW)', fontsize=22)
# ax2.tick_params(axis='y', labelsize=22)
#
# # Title and legend
# fig.tight_layout()
# fig.legend(loc='upper right', fontsize=18, bbox_to_anchor=(0.92, 0.97))
#
# # Save and display the plot
# plt.savefig('combined_plot.png', dpi=500)
# plt.show()
# plt.close()

#######################################################################################################################
# import matplotlib.pyplot as plt
#
# # Data for energy price
# energy_price = [33.28, 30.00, 29.15, 28.49, 34.66, 50.01,
#                 71.52, 77.94, 81.97, 87.90, 92.76, 94.98,
#                 92.31, 90.03, 90.09, 87.44, 85.37, 79.97,
#                 79.92, 77.83, 76.28, 65.06, 53.07, 34.16]
#
# # Extend energy_price to match the number of time slots
# extended_energy_price = []
# for price in energy_price:
#     extended_energy_price.extend([price] * int(60 / 15))
#
# # Data for sp_price
# sp_price = [3.28, 3.00, 2.15, 2.49, 3.66, 5.01,
#             7.52, 7.94, 8.97, 8.90, 9.76, 9.98,
#             9.31, 9.03, 9.09, 8.44, 8.37, 7.97,
#             7.92, 7.83, 7.28, 6.06, 5.07, 3.16]
#
# # Extend sp_price to match the number of time slots
# extended_sp_price = []
# for price in sp_price:
#     extended_sp_price.extend([price] * int(60 / 15))
#
# # Data for wind generation
# generation = [7.623491061, 4.517699791, 3.555646106, 2.48438853, 0.391585596, 1.421687974, 1.860133449,
#                 0.686599093, 2.214638619, 2.942800387, 3.988794377, 6.969082667, 9.23995314, 10.64126725,
#                 10.46401467, 10.11317679, 10.25090409, 9.420465543, 8.118983344, 8.649518668, 8.125910457,
#                 8.615698059, 10.46931187, 14.98334437, 19.50308155, 17.39479448, 14.5269699, 12.61508684,
#                 14.02984771, 15.76651556, 17.33734019, 17.38501503, 19.98105231, 21.98991494, 27.85554933,
#                 35.53486477, 36.97855651, 37.74094637, 40, 39.9535476, 39.66668364, 37.53965263, 29.76783986,
#                 23.27020832, 17.78026792, 14.89940406, 11.82376611, 10.21626853, 9.290887791, 7.289359751,
#                 5.651708858, 6.812203942, 8.832883411, 10.47705394, 12.23491061, 11.11801559, 11.76182957,
#                 14.3696837, 12.55396526, 11.41751133, 11.80461468, 13.22385779, 12.81026843, 12.90847043,
#                 10.34095655, 9.97952427, 8.952681709, 10.87271431, 8.957571436, 8.596954108, 8.731014109,
#                 9.608720012, 11.57846483, 12.4961035, 13.05271736, 11.58457699, 10.10910202, 10.91916671,
#                 11.02225844, 10.63556257, 8.29908827, 7.467427291, 6.326491112, 5.687566852, 4.413793103,
#                 6.339530383, 5.93286813, 2.206081597, 1.344674782, 2.710130902, 2.766362757, 2.951764886,
#                 3.475373096, 1.560230225, 0.514236235, 0]
#
# # Plotting
# fig, ax1 = plt.subplots(figsize=(12, 6))
#
# # Plot energy price on the left y-axis
# ax1.plot(extended_energy_price, label='Energy\nprice', marker='o', markersize=6, linestyle='-', color='b')
# ax1.plot(extended_sp_price, label='Reserve\nprice', marker='x', markersize=6, linestyle='--', color='r')
# ax1.set_xlabel('Time slots', fontsize=34)
# ax1.set_ylabel('Price (£/MWh)', fontsize=34)
# ax1.tick_params(axis='x', labelsize=28)
# ax1.tick_params(axis='y', labelsize=28)
#
# # Create a second y-axis for the wind generation
# ax2 = ax1.twinx()
# ax2.plot(generation, label='Wind', marker='o', markersize=6, linestyle='-', color='g')
# ax2.set_ylabel('Wind generation (MW)', fontsize=34)
# ax2.tick_params(axis='y', labelsize=28)
#
# # Title and legend
# fig.tight_layout()
# # fig.legend(loc='upper right', fontsize=16, bbox_to_anchor=(0.92, 0.97))
# # Adjusting legend position
# fig.legend(loc='upper left', fontsize=22, bbox_to_anchor=(0.1, 0.98))
#
#
#
# # Save and display the plot
# #plt.savefig('combined_plot_with_sp_price.png', dpi=500)
# plt.savefig('combined_plot_with_sp_price.svg', format='svg', bbox_inches='tight')
# #plt.savefig('combined_plot_with_sp_price.jpg', format='jpg', dpi=900, bbox_inches='tight')
#
#
# plt.show()
# plt.close()




