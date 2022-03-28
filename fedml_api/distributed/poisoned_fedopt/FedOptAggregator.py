import copy
import logging
import random
import time

import numpy as np
import torch
import wandb
# from FedML.fedml_core.robustness.robust_aggregation import vectorize_weight

from .optrepo import OptRepo
from .utils import transform_list_to_tensor
from training.utils.poison_utils import *
from fedml_core.robustness.robust_aggregation import RobustAggregator, is_weight_param



class FedOptAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer, poi_args):
        self.trainer = model_trainer

        self.args = args
        self.poi_args = poi_args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        self.poison_flag = dict()
        self.poison_results = dict()
        self.opt = self._instantiate_opt()
        self.robust_aggregator = RobustAggregator(args)
        self.adversary_rounds = []
        if poi_args.use and poi_args.adv_sampling == "fixed":
            freq = poi_args.adv_sampling_freq
            self.adversary_rounds = [freq*i-1 for i in range(1, args.comm_round//freq +1)]
            print(self.adversary_rounds)

        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def _instantiate_opt(self):
        return OptRepo.name2cls(self.args.server_optimizer)(
            filter(lambda p: p.requires_grad, self.get_model_params()), lr=self.args.server_lr, momentum=self.args.server_momentum,
        )
        
    def get_model_params(self):
        # return model parameters in type of generator
        return self.trainer.model.parameters()

    def get_global_model_params(self):
        # return model parameters in type of ordered_dict
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num, num_poison, poison_result):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True
        self.poison_flag[index] = num_poison
        self.poison_results[index] = poison_result

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0
        num_poisons_per_round = sum([v for k, v in self.poison_flag.items()])

        # Colluding adversaries
        if self.poi_args.collude:
            word_embedding_key = return_word_embedding_key(self.model_dict[0])
            poisoned_idx = [idx for idx in range(self.worker_num) if self.poison_results[idx] == 1]
            # More than two
            if num_poisons_per_round > 1:
                boss_idx = random.sample(poisoned_idx, 1)[0]
                for idx in poisoned_idx:
                    self.model_dict[idx][word_embedding_key] = self.model_dict[boss_idx][word_embedding_key]
                    self.poison_results[idx] = self.poison_results[boss_idx]


        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])

            # conduct the defense here:
            local_sample_number, local_model_params = self.sample_num_dict[idx], self.model_dict[idx]
            clipped_local_state_dict = local_model_params
            if self.robust_aggregator.defense_type in ("norm_diff_clipping", "weak_dp"):
                # get global parameters in cpu
                global_model = copy.deepcopy(self.trainer.model).to(device=torch.device("cpu"))
                clipped_local_state_dict = self.robust_aggregator.norm_diff_clipping(
                    local_model_params,
                    global_model.state_dict())

            model_list.append((local_sample_number, clipped_local_state_dict))
            training_num += local_sample_number

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))
        logging.info("################aggregate: %d" % len(model_list))

        # Median aggregation
        if self.robust_aggregator.defense_type == "median_agg":
            (num0, averaged_params) = model_list[0]
            vectorized_params = []
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                vectors = self.vectorize_weight(local_model_params)
                vectorized_params.append(vectors.unsqueeze(-1))
            vectorized_params = torch.cat(vectorized_params, dim=-1)
            vec_median_params = torch.median(vectorized_params, dim=-1).values

            index = 0
            for k, params in averaged_params.items():
                median_params = vec_median_params[index:index+params.numel()].view(params.size())
                index += params.numel()
                averaged_params[k] = median_params


        # Mean aggregation method
        else:
            (num0, averaged_params) = model_list[0]
            for k in averaged_params.keys():
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    w = local_sample_number / training_num
                    local_layer_update = local_model_params[k]

                    if self.robust_aggregator.defense_type == "weak_dp":
                        if is_weight_param(k):
                            local_model_params[k] = self.robust_aggregator.add_noise(
                                local_layer_update)

                    if i == 0:
                        averaged_params[k] = local_model_params[k] * w
                    else:
                        averaged_params[k] += local_model_params[k] * w

        # server optimizer
        # save optimizer state
        self.opt.zero_grad()
        opt_state = self.opt.state_dict()
        # set new aggregated grad
        self.set_model_global_grads(averaged_params)
        self.opt = self._instantiate_opt()
        # load optimizer state
        self.opt.load_state_dict(opt_state)
        self.opt.step()

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        poisoned_result = [v for k,v in self.poison_results.items()]
        return self.get_global_model_params(), num_poisons_per_round, poisoned_result

    def set_model_global_grads(self, new_state):
        new_model = copy.deepcopy(self.trainer.model)
        new_model.load_state_dict(new_state)
        with torch.no_grad():
            for parameter, new_parameter in zip(
                    self.trainer.model.parameters(), new_model.parameters()
            ):
                parameter.grad = parameter.data - new_parameter.data
                # because we go to the opposite direction of the gradient
        model_state_dict = self.trainer.model.state_dict()
        new_model_state_dict = new_model.state_dict()
        for k in dict(self.trainer.model.named_parameters()).keys():
            new_model_state_dict[k] = model_state_dict[k]
        # self.trainer.model.load_state_dict(new_model_state_dict)
        self.set_global_model_params(new_model_state_dict)

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(1, client_num_in_total), num_clients, replace=False)
            if round_idx in self.adversary_rounds:
                # process_id=1 should have client_idx=0
                client_indexes[0] = 0
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num  = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        if self.trainer.test_on_the_server(self.train_data_local_dict, self.test_data_local_dict, self.device, round_idx, self.poi_args, self.args):
            return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################local_test_on_all_clients : {}".format(round_idx))
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []

            test_num_samples = []
            test_tot_corrects = []
            test_losses = []
            for client_idx in range(self.args.client_num_in_total):
                # train data
                metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
                train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))
                
                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)
            test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
                'test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)

    def vectorize_weight(self, state_dict):
        weight_list = []
        for (k, v) in state_dict.items():
            weight_list.append(v.flatten())
        return torch.cat(weight_list)
