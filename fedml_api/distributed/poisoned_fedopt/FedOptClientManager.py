import logging
import os
import sys
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message

from .message_define import MyMessage
from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process
from training.utils.poison_utils import is_poi_client


class FedOptClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI", poi_args=None):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.client_idx = None
        self.poi_args = poi_args
        self.model_states = []
        if poi_args and poi_args.use:
            self.poisoned_client_idxs = poi_args.poisoned_client_idxs

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.client_idx = client_index

        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index), self.poi_args)
        self.round_idx = 0
        self.__train()

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.client_idx = client_index

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        poi_client_flag = is_poi_client(self.poi_args, self.client_idx, self.poisoned_client_idxs)
        if poi_client_flag and self.poi_args.ensemble:
            self.model_states.append(model_params)
            while len(self.model_states) > self.poi_args.num_ensemble:
                self.model_states.pop(0)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index), self.poi_args)
        self.round_idx += 1
        self.__train()
        if self.round_idx == self.num_rounds - 1:
            post_complete_message_to_sweep_process(self.args)
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num, num_poison_per_round, poi_result):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params("num_poison", num_poison_per_round)
        message.add_params("poison_result", poi_result)
        self.send_message(message)

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        poi_client_flag = is_poi_client(self.poi_args, self.client_idx, self.poisoned_client_idxs)
        if self.poi_args.use and self.poi_args.ensemble and poi_client_flag:
            self.poi_args.model_states = self.model_states
            #global_model = copy.deepcopy(self.trainer.trainer.model)
            #self.poi_args.global_model = global_model

        # Data poisoning
        if poi_client_flag and self.poi_args.data_poison:
            weights, local_sample_num, poi_result = self.trainer.poison_model(self.poi_args, self.round_idx)
            num_poison_per_round = 1
        else:
            weights, local_sample_num = self.trainer.train(self.round_idx, self.poi_args)
            num_poison_per_round = 0

            # Model Poisoning is done after training
            poi_result = None
            if poi_client_flag:
                weights, local_sample_num, poi_result = self.trainer.poison_model(self.poi_args, self.round_idx)
                num_poison_per_round = 1
        self.send_model_to_server(0, weights, local_sample_num, num_poison_per_round, poi_result)
