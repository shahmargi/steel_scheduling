# TODO --- define an attention neural network in pytorch --- take as input state, a mask and a feasible set of
#  controls --- use the feasible set of controls and a rounding policy to select feasible controls using the
#  predictions from the policy


import torch
import torch.nn.functional as F
import math
import random
from torch import nn
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table




################################################### Attention Policy ###################################################


class AttentionPolicy(nn.Module):
    def __init__(self, cfg: DictConfig):

        super(AttentionPolicy, self).__init__()

        self.cfg =cfg

        number_of_inputs_stage = cfg.policy.state_size                                                                   # 5 (No. of columns/features within state space)
        number_of_inputs_price = cfg.policy.price_state_size                                                             # 2 (No. of columns)

        self.mean_electricity_price = cfg.policy.state.electricity_price.mean  # mean and std of electricity prices
        self.std_electricity_price = cfg.policy.state.electricity_price.std

        self.mean_sp_price = cfg.policy.state.sp_price.mean
        self.std_sp_price = cfg.policy.state.sp_price.std

        self.mean_generation = cfg.policy.state.generation.mean  # mean and std of generation
        self.std_generation = cfg.policy.state.generation.std

        self.mean_f_id = cfg.policy.state.state_space.mean_f_id                                                         # mean and standard deviation of feature id
        self.std_f_id = cfg.policy.state.state_space.std_f_id

        self.mean_f_pt = cfg.policy.state.state_space.mean_f_pt                                                         # mean and standard deviation of feature pt
        self.std_f_pt = cfg.policy.state.state_space.std_f_pt

        self.mean_f_st = cfg.policy.state.state_space.mean_f_st                                                         # mean and standard deviation of feature st
        self.std_f_st = cfg.policy.state.state_space.std_f_st

        self.mean_f_et = cfg.policy.state.state_space.mean_f_et                                                         # mean and standard deviation of feature et
        self.std_f_et = cfg.policy.state.state_space.std_f_et

        self.mean_f_pe = cfg.policy.state.state_space.mean_f_pe                                                         # mean and standard deviation of feature pe
        self.std_f_pe = cfg.policy.state.state_space.std_f_pe

        self.attention = Attention(cfg,number_of_inputs_stage = number_of_inputs_stage)                                  # 5
        self.price_layer = nn.Linear(number_of_inputs_price, number_of_inputs_price)                                     # (2,2)

        self.ffn = PositionwiseFeedForward(number_of_inputs = cfg.neural_network.ffn_input,hidden=cfg.neural_network.ffn_hidden) # (11, 3)

        self.linear1 = nn.Linear(cfg.neural_network.linear1_in_features,cfg.neural_network.linear1_out_features)        # (3, 3)


        self.linear_1_1 = nn.Linear(cfg.neural_network.linear_in_features, cfg.neural_network.heuristic_rules_1)        # (12,4)
        self.linear_1_2 = nn.Linear(cfg.neural_network.linear_in_features, cfg.neural_network.heuristic_rules_1)

        self.linear_2_1 = nn.Linear(cfg.neural_network.linear_in_features, cfg.neural_network.heuristic_rules_2)        # (12,6)
        self.linear_2_2 = nn.Linear(cfg.neural_network.linear_in_features, cfg.neural_network.heuristic_rules_2)

        self.linear_3_1 = nn.Linear(cfg.neural_network.linear_in_features, cfg.neural_network.heuristic_rules_3)        # (12,4)
        self.linear_3_2 = nn.Linear(cfg.neural_network.linear_in_features, cfg.neural_network.heuristic_rules_3)



    def forward(self, state):
        # state - ( 1 x 4 x 17)

        mean_vector = torch.tensor([self.mean_f_id,
                                    self.mean_f_pt,
                                    self.mean_f_st,
                                    self.mean_f_et,
                                    self.mean_f_pe])

        std_vector = torch.tensor([self.std_f_id,
                                   self.std_f_pt,
                                   self.std_f_st,
                                   self.std_f_et,
                                   self.std_f_pe])

        stage1_matrix = state[:, :, :5]
        # Stage I masking
        stage1_first_column = stage1_matrix[:, :, 0].squeeze()
        stage1_mask_matrix = torch.zeros(4, 4)
        stage1_mask_matrix[stage1_first_column > 0, :] = 0
        stage1_mask_matrix[stage1_first_column < -1e-6, :] = -9999
        # normalising stage1_matrix
        stage1_matrix = (stage1_matrix - mean_vector) / std_vector

        stage2_matrix = state[:, :, 5:10]
        # Stage II masking
        stage2_first_column = stage2_matrix[:, :, 0].squeeze()
        stage2_mask_matrix = torch.zeros(4, 4)
        stage2_mask_matrix[stage2_first_column > 0, :] = 0
        stage2_mask_matrix[stage2_first_column < -1e-6, :] = -9999
        # normalising stage2_matrix
        stage2_matrix = (stage2_matrix - mean_vector) / std_vector

        stage3_matrix = state[:, :, 10:15]
        # Stage III masking
        stage3_first_column = stage3_matrix[:, :, 0].squeeze()
        stage3_mask_matrix = torch.zeros(4, 4)
        stage3_mask_matrix[stage3_first_column > 0, :] = 0
        stage3_mask_matrix[stage3_first_column < -1e-6, :] = -9999
        # normalising stage3_matrix
        stage3_matrix = (stage3_matrix - mean_vector) / std_vector

        price_matrix = state[:, :, 15:16]
        price_matrix = (price_matrix - self.mean_electricity_price) / self.std_electricity_price                                                          # normalising matrix

        sp_price_matrix = state[:, :, 16:17]
        sp_price_matrix = (sp_price_matrix - self.mean_sp_price) / self.std_sp_price

        generation_matrix = state[:, :, 17:18]
        generation_matrix = (generation_matrix - self.mean_generation) / self.std_generation

        price_generation_matrix = torch.cat((price_matrix,sp_price_matrix, generation_matrix), dim=2)             # concatenating price and generation matrix


        x1, x2, x3 = self.attention(stage1_matrix, stage2_matrix, stage3_matrix, stage1_mask_matrix, stage2_mask_matrix,
                                    stage3_mask_matrix)

        x4 = self.price_layer(price_generation_matrix.float())                                                          # Using .float() on a matrix or tensor typically converts all the elements to floating-point numbers.in machine learning, neural networks often work with floating-point numbers because they involve computations with weights, activations, and gradients, which are typically represented as floating-point values.

        x = torch.cat((x1, x2, x3, x4), dim=2)                                                                  # x.size = ( 1 x 4 x 11)

        x = self.ffn(x)                                                                                                 # x.size = ( 1 x 4 x 3)

        x = self.linear1(x)                                                                                             # x.size = ( 1 x 4 x 3)

        x = x.flatten()                                                                                                 # Flatten the matrix to make it  12

        # stage I,unit 1 #
        u1 = self.linear_1_1(x)                                                                                         # 4

        # stage I,unit 2 #
        u2 = self.linear_1_2(x)                                                                                         # 4


        # stage II,unit 1 #
        u3 = self.linear_2_1(x)                                                                                         # 6


        # stage II,unit 2 #
        u4 = self.linear_2_2(x)                                                                                         # 6


        # stage III,unit 1 #
        u5 = self.linear_3_1(x)                                                                                         # 6


        # stage III,unit 2 #
        u6 = self.linear_3_2(x)                                                                                         # 6



        # Applying softmax #
        u1_softmax = torch.nn.functional.softmax(u1, dim=0)

        u2_softmax = torch.nn.functional.softmax(u2, dim=0)

        u3_softmax = torch.nn.functional.softmax(u3, dim=0)

        u4_softmax = torch.nn.functional.softmax(u4, dim=0)

        u5_softmax = torch.nn.functional.softmax(u5, dim=0)

        u6_softmax = torch.nn.functional.softmax(u6, dim=0)



        ######### which decision rule to select depends on the highest probablity index ###############################

        highest_prob_index_u1 = torch.argmax(u1_softmax)
        highest_prob_index_u2 = torch.argmax(u2_softmax)
        highest_prob_index_u3 = torch.argmax(u3_softmax)
        highest_prob_index_u4 = torch.argmax(u4_softmax)
        highest_prob_index_u5 = torch.argmax(u5_softmax)
        highest_prob_index_u6 = torch.argmax(u6_softmax)

        

        control_rules =[
                        highest_prob_index_u1,
                        highest_prob_index_u2,
                        highest_prob_index_u3,
                        highest_prob_index_u4,
                        highest_prob_index_u5,
                        highest_prob_index_u6
                        ]


        return control_rules




 #######################################################################################################################

    def count_params(self, print_table=True):

        """Count the number of parameters in a state dict."""

        state_dict = self.state_dict()

        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Modules", justify="left", style="cyan")
        table.add_column("Parameters", justify="right", style="magenta")
        table.add_column("shape", justify="right", style="green")

        total_params = 0
        for name, parameter in state_dict.items():
            params = parameter.numel()
            table.add_row(name, str(params), str(parameter.shape))
            total_params += params
        table.title = f"Total Trainable Params: {total_params}"

        if print_table:
            console = Console()
            console.print(table)
        return total_params


###################################################  I: Attention Operation ############################################
class Attention(nn.Module):                                                                                              # Class for Attention operation

    def __init__(self,cfg: DictConfig,number_of_inputs_stage):
        super().__init__()
        self.number_of_inputs_stage = number_of_inputs_stage                                                            # 5
        self.qkv1_layer = nn.Linear(number_of_inputs_stage, cfg.neural_network.qkv_out_features)                        # (5,9)
        self.qkv2_layer = nn.Linear(number_of_inputs_stage, cfg.neural_network.qkv_out_features)                        # (5,9)
        self.qkv3_layer = nn.Linear(number_of_inputs_stage, cfg.neural_network.qkv_out_features)                        # (5,9)
        self.linear1_layer = nn.Linear(cfg.neural_network.attention_linear_in_features, cfg.neural_network.attention_linear_out_features)   # (3,3)
        self.linear2_layer = nn.Linear(cfg.neural_network.attention_linear_in_features, cfg.neural_network.attention_linear_out_features)   # (3,3)
        self.linear3_layer = nn.Linear(cfg.neural_network.attention_linear_in_features, cfg.neural_network.attention_linear_out_features)   # (3,3)

    def forward(self, stage1_matrix, stage2_matrix, stage3_matrix, stage1_mask_matrix, stage2_mask_matrix,
                stage3_mask_matrix):


        qkv1 = self.qkv1_layer(stage1_matrix.float())                                                                     # stage1_matrix goes into qkv layer of size (5,9)
                                                                                                                         # Output qkv1  ( 1 x 4 x 9)

        qkv2 = self.qkv2_layer(stage2_matrix.float())                                                                     # stage2_matrix goes into qkv layer of size (5,9)
                                                                                                                         # Output qkv2  ( 1 x 4 x 9)

        qkv3 = self.qkv3_layer(stage3_matrix.float())                                                                     # stage3_matrix goes into qkv layer of size (5,9)
                                                                                                                         # Output qkv3  ( 1 x 4 x 9)

        q1, k1, v1 = qkv1.chunk(3, dim=-1)                                                                               # Output qkv1 ( 1 x 4 x 9) gets chunked to q1( 1 x 4 x 3) ,k1( 1 x 4 x 3), v1( 1 x 4 x 3)
        values1, attention1 = scaled_dot_product1(q1, k1, v1, stage1_mask_matrix)                                       # values1( 1 x 4 x 3) ,attention1( 1 x 4 x 4)
        out1 = self.linear1_layer(values1)                                                                               # Passing values ( 1 x 4 x 3) to linear layer of size (3,3)
                                                                                                                        # Output out1  ( 1 x 4 x 3)

        q2, k2, v2 = qkv2.chunk(3, dim=-1)
        values2, attention2 = scaled_dot_product2(q2, k2, v2, stage2_mask_matrix)
        out2 = self.linear2_layer(values2)

        q3, k3, v3 = qkv3.chunk(3, dim=-1)
        values3, attention3 = scaled_dot_product3(q3, k3, v3, stage3_mask_matrix)
        out3 = self.linear3_layer(values3)

        return out1, out2, out3


###################################################  stage1 ############################################################
def scaled_dot_product1(q1, k1, v1, stage1_mask_matrix):
    d_k1 = q1.size()[-1]        # 3
    scaled1 = torch.matmul(q1, k1.transpose(1, 2)) / math.sqrt(d_k1)                                                    # Output scaled1  (1 x 4 x 4)

    if stage1_mask_matrix is not None:
        scaled1 += stage1_mask_matrix                                                                                   # Adding Mask matrix to scaled1 matrix
    attention1 = F.softmax(scaled1, dim=-1)                                                                             # Doing SoftMax and resulting attention1 is of size (1 x 4 x 4)
    values1 = torch.matmul(attention1, v1)                                                                              # values 1 of size  (1 x 4 x 3)

    return values1, attention1


###################################################  stage2 ############################################################
def scaled_dot_product2(q2, k2, v2, stage2_mask_matrix):
    d_k2 = q2.size()[-1]
    scaled2 = torch.matmul(q2, k2.transpose(1, 2)) / math.sqrt(d_k2)

    if stage2_mask_matrix is not None:
        scaled2 += stage2_mask_matrix
    attention2 = F.softmax(scaled2, dim=-1)
    values2 = torch.matmul(attention2, v2)

    return values2, attention2


###################################################  stage3 ############################################################
def scaled_dot_product3(q3, k3, v3, stage3_mask_matrix):
    d_k3 = q3.size()[-1]
    scaled3 = torch.matmul(q3, k3.transpose(1, 2)) / math.sqrt(d_k3)

    if stage3_mask_matrix is not None:
        scaled3 += stage3_mask_matrix
    attention3 = F.softmax(scaled3, dim=-1)
    values3 = torch.matmul(attention3, v3)

    return values3, attention3


###################################################  II: FFNN Operation ################################################
class PositionwiseFeedForward(nn.Module):
    def __init__(self, number_of_inputs, hidden):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs, hidden)                                                               # (11,3)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.linear1(x)                                                                                             # x = (1 * 4 * 3)

        x = self.relu(x)

        return x
