#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import time
import numpy as np
import math
import os

# gene[f][c] f:function type, c:connection (nodeID)
class Individual(object):

    def __init__(self, net_info, init):
        self.net_info = net_info
        self.gene = np.zeros((self.net_info.node_num + self.net_info.out_num, self.net_info.max_in_num + 1)).astype(int)
        self.is_active = np.empty(self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.is_pool = np.empty(self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.eval = None
        if init:
            print('init with specific architectures')
            self.init_gene_with_conv() # In the case of starting only convolution
        else:
            self.init_gene()           # generate initial individual randomly

    def init_gene_with_conv(self):
        # initial architecture
        arch = ['S_ConvBlock_64_3']

        input_layer_num = int(self.net_info.input_num / self.net_info.rows) + 1
        output_layer_num = int(self.net_info.out_num / self.net_info.rows) + 1
        layer_ids = [((self.net_info.cols - 1 - input_layer_num - output_layer_num) + i) // (len(arch)) for i in range(len(arch))]
        prev_id = 0 # i.e. input layer
        current_layer = input_layer_num
        block_ids = []  # *do not connect with these ids

        # building convolution net
        for i, idx in enumerate(layer_ids):

            current_layer += idx
            n = current_layer * self.net_info.rows + np.random.randint(self.net_info.rows)
            block_ids.append(n)
            self.gene[n][0] = self.net_info.func_type.index(arch[i])
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0

            self.gene[n][1] = prev_id
            for j in range(1, self.net_info.max_in_num):
                self.gene[n][j + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)

            prev_id = n + self.net_info.input_num

        # output layer
        n = self.net_info.node_num
        type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
        self.gene[n][0] = np.random.randint(type_num)
        col = np.min((int(n / self.net_info.rows), self.net_info.cols))
        max_connect_id = col * self.net_info.rows + self.net_info.input_num
        min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
            if col - self.net_info.level_back >= 0 else 0

        self.gene[n][1] = prev_id
        for i in range(1, self.net_info.max_in_num):
            self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)
        block_ids.append(n)

        # intermediate node
        for n in range(self.net_info.node_num + self.net_info.out_num):

            if n in block_ids:
                continue

            # type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            self.gene[n][0] = np.random.randint(type_num)
            # connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            for i in range(self.net_info.max_in_num):
                self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)

        self.check_active()

    def init_gene(self):
        # intermediate node
        for n in range(self.net_info.node_num + self.net_info.out_num):
            # type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            self.gene[n][0] = np.random.randint(type_num)
            # connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            for i in range(self.net_info.max_in_num):
                self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)

        self.check_active()

    def __check_course_to_out(self, n):
        if not self.is_active[n]:
            self.is_active[n] = True
            t = self.gene[n][0]
            if n >= self.net_info.node_num:    # output node
                in_num = self.net_info.out_in_num[t]
            else:    # intermediate node
                in_num = self.net_info.func_in_num[t]

            for i in range(in_num):
                if self.gene[n][i+1] >= self.net_info.input_num:
                    self.__check_course_to_out(self.gene[n][i+1] - self.net_info.input_num)

    def check_active(self):
        # clear
        self.is_active[:] = False
        # start from output nodes
        for n in range(self.net_info.out_num):
            self.__check_course_to_out(self.net_info.node_num + n)

    def check_pool(self):
        is_pool = True
        pool_num = 0
        for n in range(self.net_info.node_num + self.net_info.out_num):
            if self.is_active[n]:
                if self.gene[n][0] > 19:
                    is_pool = False
                    pool_num += 1
        return is_pool, pool_num

    def __mutate(self, current, min_int, max_int):
        mutated_gene = current
        while current == mutated_gene:
            mutated_gene = min_int + np.random.randint(max_int - min_int)
        return mutated_gene

    def mutation(self, mutation_rate=0.05):
        active_check = False

        for n in range(self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
                if self.is_active[n]:
                    active_check = True
            # mutation for connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            in_num = self.net_info.func_in_num[t] if n < self.net_info.node_num else self.net_info.out_in_num[t]
            for i in range(self.net_info.max_in_num):
                if np.random.rand() < mutation_rate and max_connect_id - min_connect_id > 1:
                    self.gene[n][i+1] = self.__mutate(self.gene[n][i+1], min_connect_id, max_connect_id)
                    if self.is_active[n] and i < in_num:
                        active_check = True

        self.check_active()
        return active_check

    def neutral_mutation(self, mutation_rate=0.05):
        for n in range(self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if not self.is_active[n] and np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
            # mutation for connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            in_num = self.net_info.func_in_num[t] if n < self.net_info.node_num else self.net_info.out_in_num[t]
            for i in range(self.net_info.max_in_num):
                if (not self.is_active[n] or i >= in_num) and np.random.rand() < mutation_rate \
                        and max_connect_id - min_connect_id > 1:
                    self.gene[n][i+1] = self.__mutate(self.gene[n][i+1], min_connect_id, max_connect_id)

        self.check_active()
        return False

    def count_active_node(self):
        return self.is_active.sum()

    def copy(self, source):
        self.net_info = source.net_info
        self.gene = source.gene.copy()
        self.is_active = source.is_active.copy()
        self.eval = source.eval

    def active_net_list(self):
        net_list = [["input", 0, 0]]
        active_cnt = np.arange(self.net_info.input_num + self.net_info.node_num + self.net_info.out_num)
        active_cnt[self.net_info.input_num:] = np.cumsum(self.is_active)

        for n, is_a in enumerate(self.is_active):
            if is_a:
                t = self.gene[n][0]
                if n < self.net_info.node_num:    # intermediate node
                    type_str = self.net_info.func_type[t]
                else:    # output node
                    type_str = self.net_info.out_type[t]

                connections = [active_cnt[self.gene[n][i+1]] for i in range(self.net_info.max_in_num)]
                net_list.append([type_str] + connections)
        return net_list


# CGP with (1 + \lambda)-ES
class CGP(object):
    def __init__(self, net_info, eval_func, population=1, lam=4, imgSize=32, init=False):
        self.pop_size = population
        self.lam = lam
        self.pop = [Individual(net_info, init) for _ in range(1 + self.lam) for _ in range(population)]
        self.eval_func = eval_func
        self.num_gen = 0
        self.num_eval = 0
        self.max_pool_num = int(math.log2(imgSize) - 2)
        self.init = init

    def _evaluation(self, pop, eval_flag):
        # create network list
        net_lists = []
        active_index = np.where(eval_flag)[0]
        for i in active_index:
            net_lists.append(pop[i].active_net_list())

        # evaluation
        fp = self.eval_func(net_lists)
        for i, j in enumerate(active_index):
            pop[j].eval = fp[i]
        evaluations = np.zeros(len(pop))
        for i in range(len(pop)):
            evaluations[i] = pop[i].eval

        self.num_eval += len(net_lists)
        return evaluations

    def _log_data(self, net_info_type='active_only', start_time=0, pop_num=0):
        log_list = [self.num_gen, self.num_eval, time.time()-start_time, self.pop[pop_num].eval, self.pop[pop_num].count_active_node()]
        if net_info_type == 'active_only':
            log_list.append(self.pop[pop_num].active_net_list())
        elif net_info_type == 'full':
            log_list += self.pop[pop_num].gene.flatten().tolist()
        else:
            pass
        return log_list

    def _log_data_children(self, net_info_type='active_only', start_time=0, pop=None):
        log_list = [self.num_gen, self.num_eval, time.time()-start_time, pop.eval, pop.count_active_node()]
        if net_info_type == 'active_only':
            log_list.append(pop.active_net_list())
        elif net_info_type == 'full':
            log_list += pop.gene.flatten().tolist()
        else:
            pass
        return log_list

    # TODO: change with self.pop_size 'but, currently UNUSED'
    def load_log(self, log_data):
        self.num_gen = log_data[0]
        self.num_eval = log_data[1]
        net_info = self.pop[0].net_info
        self.pop[0].eval = log_data[3]
        self.pop[0].gene = np.array(log_data[5:]).reshape((net_info.node_num + net_info.out_num, net_info.max_in_num + 1))
        self.pop[0].check_active()

    # Evolution CGP:
    #   At each iteration:
    #     - Generate lambda individuals in which at least one active node changes (i.e., forced mutation)
    #     - Mutate the best individual with neutral mutation (unchanging the active nodes)
    #         if the best individual is not updated.
    def modified_evolution(self, max_gen=250, mutation_rate=0.05, log_folder='./log_folder', neutral_mutation_mode='normal'):
        # variable settings
        start_time = time.time()
        eval_flag = np.empty(self.pop_size * self.lam)
        active_num = [self.pop[i].count_active_node() for i in range(self.pop_size)]
        pool_num = []
        for i in range(self.pop_size):
            _, pool_num_i = self.pop[i].check_pool()
            pool_num.append(pool_num_i)
        if self.init:
            pass
        # else: # in the case of not using an init indiviudal
            # TODO: change with self.pop_size 'but, currently UNUSED'
            # while active_num < self.pop[0].net_info.min_active_num or active_num > self.pop[0].net_info.max_active_num or pool_num > self.max_pool_num:
            #     self.pop[0].mutation(1.0)
            #     active_num = self.pop[0].count_active_node()
            #     _, pool_num= self.pop[0].check_pool()
        self._evaluation([self.pop[i] for i in range(self.pop_size)], np.array([True for _ in range(self.pop_size)]))
        print(self._log_data(net_info_type='active_only', start_time=start_time))

        best_is_parent = 0 # for strong neutral mutation

        # Create path for log files
        if not os.path.isdir(log_folder):
            os.mkdir(log_folder)
        child_path = os.path.join(log_folder, 'child.txt')
        arch_child_path = os.path.join(log_folder, 'arch_child.txt')
        log_cgp_path = os.path.join(log_folder, 'log_cgp.txt')
        log_arch_path = os.path.join(log_folder, 'arch.txt')

        # condition for termination
        while self.num_gen < max_gen:
            self.num_gen += 1
            # reproduction
            for k in range(self.pop_size):  # for every parent in population
                # self.pop = [parent0, parent1, .. , parent(self.pop_size-1), children0-0, children0-1, .. children0-(self.lam-1), children1-0, ...]
                # starting children index for current parent.
                start_index = self.pop_size + k * self.lam
                for i in range(self.lam):  # create self.lam child each
                    _i = start_index + i
                    eval_flag[k * self.lam + i] = False
                    self.pop[_i].copy(self.pop[k])  # copy a parent
                    active_num[k] = self.pop[_i].count_active_node()  # number of active nodes for parent k.
                    _, pool_num_k= self.pop[_i].check_pool()
                    pool_num[k] = pool_num_k
                    # mutation (forced mutation)
                    while not eval_flag[k * self.lam + i] or active_num[k] < self.pop[_i].net_info.min_active_num or active_num[k] > self.pop[_i].net_info.max_active_num or pool_num[k] > self.max_pool_num:
                        self.pop[_i].copy(self.pop[k])  # copy a parent
                        eval_flag[k * self.lam + i] = self.pop[_i].mutation(mutation_rate)  # mutation
                        active_num[k] = self.pop[_i].count_active_node()
                        _, pool_num_k= self.pop[_i].check_pool()
                        pool_num[k] = pool_num_k
            # main log file for child and its arch
            child = open(child_path, 'a')
            arch_child = open(arch_child_path, 'a')
            writer_child = csv.writer(child, lineterminator='\n')
            writer_arch_child = csv.writer(arch_child, lineterminator='\n')
            for c in range(len(self.pop)):  # writes all including parents
                writer_child.writerow(self._log_data_children(net_info_type='full', start_time=start_time, pop=self.pop[c]))
                writer_arch_child.writerow(self._log_data_children(net_info_type='active_only', start_time=start_time, pop=self.pop[c]))
            child.close()
            arch_child.close()

            # evaluation and selection
            # replace the parent by the best individual.
            child_evaluations = self._evaluation(self.pop[self.pop_size:], eval_flag=eval_flag)
            parent_evaluations = [self.pop[i].eval for i in range(self.pop_size)]
            combined_evaluations = parent_evaluations + child_evaluations
            # choose bests from parent and children, and iterate.
            # if it is survived parent, neutral mutation. if children, pass.
            best_args = np.argpartition(combined_evaluations, -self.pop_size)[-self.pop_size:]
            best_args.sort()  # bring survived parents to the front
            for index, arg in np.ndenumerate(best_args):
                index = index[0]  # index is tuple
                if arg < self.pop_size:  # neutral mutation
                    self.pop[index].copy(self.pop[arg])
                    self.pop[index].neutral_mutation(mutation_rate)
                else:  # replace with children
                    self.pop[index].copy(self.pop[arg])
            # TODO: if need strong neutral mutation

            # log for each generation
            print(self._log_data(net_info_type='active_only', start_time=start_time))
            log_cgp = open(log_cgp_path, 'a')
            log_arch = open(log_arch_path, 'a')
            writer_cgp = csv.writer(log_cgp, lineterminator='\n')
            writer_arch = csv.writer(log_arch, lineterminator='\n')
            for i in range(self.pop_size):
                writer_cgp.writerow(self._log_data(net_info_type='full', start_time=start_time, pop_num=i))
                writer_arch.writerow(self._log_data(net_info_type='active_only', start_time=start_time, pop_num=i))
            log_cgp.close()
            log_arch.close()
