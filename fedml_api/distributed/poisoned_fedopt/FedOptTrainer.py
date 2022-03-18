from .utils import transform_tensor_to_list


class FedOptTrainer(object):

    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, train_data_num, device,
                 args, model_trainer):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num
        # self.train_local = self.train_data_local_dict[client_index]
        # self.local_sample_number = self.train_data_local_num_dict[client_index]

        self.device = device
        self.args = args

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index, poi_args=None):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        if poi_args and poi_args.use:
            self.poi_train_local = poi_args.train_data_local_dict.get(client_index, None)
            self.poi_test_local = poi_args.test_data_local_dict.get(client_index, None)


    def train(self, round_idx=None, poi_args=None):
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args, poi_args)

        weights = self.trainer.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number

    def poison_model(self, poi_args, round_idx=None):
        self.args.round_idx = round_idx
        poi_data = (self.poi_train_local, self.poi_test_local)
        result = self.trainer.poison_model(poi_data, self.device, poi_args)
        weights = self.trainer.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number, result
