import time
import logging
import numpy as np
from typing import List
import socket
from mindspore_serving.config.config import ServingConfig
from mindspore_serving.models.build_inputs import build_inputs


class BaseInputsOfInfer:
    """
    定义通用的推理输入操作
    BaseInputsOfInfer interface.
    """

    def get_inputs(self, model, **kwargs):
        pass

    @staticmethod
    def get_lite_tensor_list(inputs, model):
        # 将输入转换为 Lite 模型所需的张量（tensor）列表。
        # 它遍历输入列表，将非 None 的项加入到一个新的列表中，并将这些数据设置到模型的输入张量中。
        input_list = []
        for item in inputs:
            if item is None:
                continue
            input_list.append(item)
        lite_inputs = model.get_inputs()
        for input_np, tensor in zip(input_list, lite_inputs):
            tensor.set_data_from_numpy(input_np)
        return lite_inputs


class CommonInputsOfInfer(BaseInputsOfInfer):
    """
    # 用于处理常见的大语言模型（LLM）输入数据，主要特点是支持多批次（multibatch）操作。
    common infer inputs of llm models.
    """

    def __init__(self):
        pass

    # pylint: disable=W0221
    def get_inputs(self, input_ids=None, current_index=None, valid_length=None,
                   init_reset=None, is_first_iteration=True, **kwargs):
        if not is_first_iteration:#模型在多次迭代中逐步处理输入
            inputs_tmp = []
            for i in range(len(current_index)):#根据批次编号来计算每个批次在全局数据中的偏移量
                current_index_tmp = int(current_index[i]) - i * input_ids.shape[1]  # multibatch
                # use numpy to slice array to avoid complie ascend slice op
                inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
            input_ids = np.array(inputs_tmp, dtype=np.int32)

        inputs = [input_ids, current_index, init_reset, valid_length]
        return inputs


class CommonInputsOfInferDyn(BaseInputsOfInfer):
    """
    增加了一些额外的输入参数，如 mask、freq_cos 和 freq_sin，这些可能是与模型结构相关的额外输入。
    common infer inputs of llm models.
    """

    def __init__(self):
        pass

    # pylint: disable=W0221
    def get_inputs(self, input_ids=None, current_index=None, valid_length=None,
                   init_reset=None, is_first_iteration=True, InputExtraList=[], **kwargs):
        mask = InputExtraList[0]
        freq_cos = InputExtraList[1]
        freq_sin = InputExtraList[2]
        if not is_first_iteration:
            inputs_tmp = []
            for i in range(len(current_index)):
                current_index_tmp = int(current_index[i]) - i * input_ids.shape[1]  # multibatch
                # use numpy to slice array to avoid complie ascend slice op

                inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
            input_ids = np.array(inputs_tmp, dtype=np.int32)
        if is_first_iteration:
            # mask, freq_cos, fre_sin 
            inputs = [input_ids, current_index, init_reset, valid_length, mask, freq_cos, freq_sin]
        else:
            inputs = [input_ids, current_index, init_reset, valid_length]
        return inputs


class CustomInputsOfInfer(BaseInputsOfInfer):
    """
    支持用户自定义输入的类。
    common infer inputs of llm models.
    """

    def __init__(self):
        self.get_input_from_config = get_inputs_custom

    # pylint: disable=W0221
    def get_inputs(self, **kwargs):
        return self.get_input_from_config(**kwargs)

        # logging.debug("inputs after get_inputs:{}".format(inputs))
        # lite_inputs = BaseInputsOfInfer.get_lite_tensor_list(inputs, model)
        # return lite_inputs
        inputs_custom = self.get_input_from_config(**kwargs)
        if inputs_custom is None:
            logging.error('custom inputs definited by customer is None,please check it in server config!')
        return inputs_custom


class InputOfInfer:
    """
    管理不同模型推理输入的类
    Input of llm model.
    """
    MAPPING = {
        "bloom": CommonInputsOfInfer,
        "llama": CommonInputsOfInfer,
        "glm2": CommonInputsOfInfer,
        "common": CommonInputsOfInfer,
        "llama_dyn": CommonInputsOfInferDyn,
        "wizard_coder": CommonInputsOfInferDyn,
        "internlm": CommonInputsOfInfer,
        "baichuan2": CommonInputsOfInfer,
        "custom": CustomInputsOfInfer
    }

    @classmethod
    def get_inputs(cls, model_name: str, **kwargs):
        """
        Get input tensor list of mslite.

        Args:
            model_name: str, model name.

        Returns:
            tensor list of mslite.
        """
        # name = ""
        # if Baseconfig['input_function'] == 'custom':
        #     model_name = "custom"
        #     logging.debug('model name {}'.format(model_name))
        # if model_name not in InputOfInfer.MAPPING:
        #     for k in InputOfInfer.MAPPING:
        #         if model_name.startswith(k):
        #             name = k
        #             break
        #     if not name:
        #         logging.warning("Model name not in support maps.Common input format will be used to do inference.")
        #         name = "common"
        # else:
        #     name = model_name
        return InputOfInfer.MAPPING['common']().get_inputs(**kwargs)


class CommonWarp:
    """
    封装通用的大语言模型输入，提供了对输入张量的预处理。
    common infer inputs of llm models.
    """

    def __init__(self):
        pass

    # pylint: disable=W0221
    def get_warp_inputs(self, lite_inputs=None, **kwargs):
        init = 0
        init_reset = [init for _ in range(Baseconfig.prefill_batch_size)]

        lite_inputs[2] = np.array(init_reset).reshape(Baseconfig.prefill_batch_size, 1).astype(np.int32)

        first_group = np.concatenate((lite_inputs[0], lite_inputs[1].reshape(Baseconfig.prefill_batch_size, 1),
                                      lite_inputs[2], lite_inputs[3].reshape(Baseconfig.prefill_batch_size, 1)), axis=1)
        second_group = []
        return first_group, second_group


class CommonWarpDyn:
    """
    处理的是动态输入类型。
    common infer inputs of llm models.
    """

    def __init__(self):
        pass

    # pylint: disable=W0221
    def get_warp_inputs(self, lite_inputs=None, **kwargs):
        init = 0
        init_reset = [init for _ in range(Baseconfig.prefill_batch_size)]
        lite_inputs[2] = np.array(init_reset).reshape(Baseconfig.prefill_batch_size, 1).astype(np.int32)

        first_group = np.concatenate((lite_inputs[0], lite_inputs[1].reshape(Baseconfig.prefill_batch_size, 1),
                                      lite_inputs[2], lite_inputs[3].reshape(Baseconfig.prefill_batch_size, 1)), axis=1)

        second_group = []
        for i in range(4, len(lite_inputs)):
            second_group.append(lite_inputs[i])
        return first_group, second_group


class WarpInputOfInfer:
    """
    用于管理不同模型的输入封装操作。
    Input of llm model.
    """
    MAPPING = {
        "bloom": CommonWarp,
        "llama": CommonWarp,
        "glm2": CommonWarp,
        "common": CommonWarp,
        "llama_dyn": CommonWarpDyn,
        "wizard_coder": CommonWarpDyn,
        "internlm": CommonWarp,
        "baichuan2": CommonWarp,
    }

    @classmethod
    def get_warp_inputs(cls, model_name: str, **kwargs):
        """
        Get warpping input tensor list of mslite.

        Args:
            model_name: str, model name.

        Returns:
            tensor list of mslite.
        """
        name = ""
        if model_name not in InputOfInfer.MAPPING:
            for k in InputOfInfer.MAPPING:
                if model_name.startswith(k):
                    name = k
                    break
            if not name:
                logging.warning("Model name not in support maps.Common input format will be used to do inference.")
                name = "common"
        else:
            name = model_name
        return WarpInputOfInfer.MAPPING[name]().get_warp_inputs(**kwargs)


class Singleton(object):
    # 单例模式的实现
    def __init__(self, cls):
        self._cls = cls
        self.uniqueInstance = None

    def __call__(self):
        if self.uniqueInstance is None:
            self.uniqueInstance = self._cls()
        return self.uniqueInstance


"""
全局定义一个DisModel, 保存和agents的通信管道
"""


@Singleton
class DisModel:
    # 用来处理与外部代理服务器的交互及模型推理相关的操作
    # 通过 socket 与外部代理服务器交互，同时管理输入和推理过程中的各种状态
    def __init__(self):
        self.agent_stubs = []#用于存储与多个代理服务器的连接。
        self.model_name = None#用于存储模型名称。
        self.config = None#用于存储配置信息。

    def init(self, config, shm_names: List[str] = None):
        # 用于初始化模型，接收 config 配置信息和 shm_names（共享内存名称列表）。
        self.config = config
        # 从 config 提取 agent_ip 和 agent_ports，用于连接代理服务器，同时获取 model_name（模型名称）。
        agent_ip = config.serving_config.agent_ip
        agent_ports = config.serving_config.agent_ports
        model_name = config.model_config.model_name
        logging.debug(f"agent_ports is {agent_ports}")
        # 循环遍历代理端口，使用 socket 创建 TCP 连接，与代理服务器进行通信。
        for port in agent_ports:
            logging.debug("port ip is {}".format(port))
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # socket是1对1的，设置超时机制，防止多个serving连接同一个LLM
            client.settimeout(5)
            client.connect((agent_ip, port))
            send_str = '#' + ",".join(str(element) for element in shm_names)
            client.sendall(send_str.encode())
            data = client.recv(6, socket.MSG_WAITALL).decode()
            logging.debug(data)
            # 如果代理返回 "failed" 字符串，说明代理已连接其他客户端，此时关闭所有连接并抛出错误。
            if data == "failed":
                client.close()
                for agent in self.agent_stubs:
                    agent.close()
                raise ConnectionError("there exists another connected serving now, stop the previous serving at first")
            # 否则，将成功连接的代理存储在 agent_stubs 列表中。
            self.agent_stubs.append(client)
            client.settimeout(None)
            # send shm_names
        self.model_name = model_name

    @staticmethod
    def reset_agent_status(config):
        # 用于重置代理服务器的状态。通过向每个代理端口发送 "r" 请求，代理响应 "succes" 时表示重置成功。
        logging.debug("waiting to reset agents status")
        agent_ip = config.serving_config.agent_ip
        agent_ports = config.serving_config.agent_ports
        for port in agent_ports:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # socket是1对1的，设置超时机制，防止多个serving连接同一个LLM
            client.settimeout(5)
            client.connect((agent_ip, port))
            client.sendall("r".encode())
            data = client.recv(6, socket.MSG_WAITALL).decode()
            logging.debug(data)
            if data == "succes":
                logging.debug("reset")
        logging.debug("reset all agents!")

    def stop(self):
        # 用于停止与代理的连接。
        logging.debug("waiting worker to exit")
        # 通过向代理发送 "e" 来请求停止，并等待代理返回 "free" 来确认是否关闭连接。
        for item in self.agent_stubs:
            cnt = 0
            while True or cnt < 1000:
                item.sendall("e".encode())
                data = item.recv(4096).decode()
                logging.debug(data)
                if data == "free":
                    logging.debug("close socket")
                    item.close()
                    break
                cnt += 1
            if cnt >= 1000:
                logging.debug("agent is running now, failed to stop serving, try to stop later")
        logging.debug("exit!")

    def get_predict_inputs(self, input_ids, current_index=None,
                           valid_length=None, init_reset=None, is_first_iteration=True, **kwargs):
        # 获取预测时的模型输入
        """Get inputs of llm model for mslite."""
        return InputOfInfer.get_inputs(self.model_name, input_ids=input_ids, current_index=current_index,
                                       valid_length=valid_length, init_reset=init_reset,
                                       is_first_iteration=is_first_iteration, **kwargs)

    def get_model_inputs(self, input_ids, current_index=None,
                         valid_length=None, is_first_iteration=True, **kwargs) -> np.array:
        # 基于是否是第一次迭代，调整 init_reset 的值，然后调用 get_predict_inputs 获取模型输入。
        if is_first_iteration:
            init_reset = np.array([False])
            lite_inputs = self.get_predict_inputs(input_ids, current_index,
                                                  valid_length, init_reset, is_first_iteration, **kwargs)

        else:
            init_reset = np.array([True])
            lite_inputs = self.get_predict_inputs(input_ids, current_index,
                                                  valid_length, init_reset, is_first_iteration, **kwargs)

        return lite_inputs

    def get_warp_inputs(self, lite_inputs=None, **kwargs):
        """处理封装输入 Get inputs of llm model for mslite."""
        return WarpInputOfInfer.get_warp_inputs(self.model_name, lite_inputs=lite_inputs, **kwargs)

    @staticmethod
    def get_gen_parms_np(batch_size, dtype=np.float16, **kwargs):
        # 构造并返回一个多参数矩阵 parms_np，该矩阵用于生成模型的推理参数。
        do_sample_list = kwargs.pop("do_sample_list")
        top_k_list = kwargs.pop("top_k_list")
        top_p_list = kwargs.pop("top_p_list"),
        temperature_list = kwargs.pop("temperature_list"),
        repetition_penalty = kwargs.pop("repetition_penalty")
        decode_index_list = kwargs.pop("decode_index_list")

        do_sample_np = np.array(do_sample_list).reshape(batch_size, 1).astype(dtype)
        top_p_np = np.array(top_p_list).reshape(batch_size, 1).astype(dtype)
        top_k_np = np.array(top_k_list).reshape(batch_size, 1).astype(dtype)
        temperature_np = np.array(temperature_list).reshape(batch_size, 1).astype(dtype)
        repetition_np = np.array(repetition_penalty).reshape(batch_size, 1).astype(dtype)
        decode_index_np = np.array(decode_index_list).reshape(batch_size, 1).astype(dtype)
        parms_np = np.concatenate((do_sample_np, top_p_np, top_k_np, temperature_np, repetition_np, decode_index_np),
                                  axis=-1)
        return parms_np

    def _assemble_pa_inputs(self, is_first_iteration, batch_valid_length: np.array, cache_engine_list, seq_length,
                            valid_batch_flag):
        # 用于标识是否是第一次迭代,处理全量输入和增量输入的封装和映射。
        if is_first_iteration:
            return self._assemble_pa_full_inputs(batch_valid_length, cache_engine_list, seq_length, valid_batch_flag)
        else:
            return self._assemble_pa_inc_inputs(batch_valid_length, cache_engine_list, seq_length, valid_batch_flag)

    def _assemble_pa_full_inputs(self, batch_valid_length: np.array, cache_engine_list, seq_length, valid_batch_flag):
        """
        batch_valid_length: 每个序列有效的长度，表示批次中每个输入的有效token数量。
        cache_engine_list: 缓存引擎的列表，管理模型的缓存机制。
        seq_length: 序列的总长度。
        valid_batch_flag: 标识哪些批次中的序列是有效的。
        """
        # 获取块大小，这是模型缓存管理中每个缓存块的尺寸。
        block_size = cache_engine_list[0].block_size
        # 计算每个序列能够分配的最大块数量，等于序列总长度除以块大小。
        max_num_blocks_per_seq = seq_length // block_size

        # 有效批次的大小，等于 valid_batch_flag 的长度。
        bs = len(valid_batch_flag)
        # 用于存储块表和插槽映射的列表。
        block_tables = []
        slot_mapping = []
        # 循环处理每个有效批次
        for i in range(bs):
            # 如果对应的 valid_batch_flag[i] 为真，调用 prepare_cache 准备缓存，
            # 传入 batch_valid_length[i]（即该批次序列的有效长度）。
            if valid_batch_flag[i]:
                cache_engine_list[i].prepare_cache(batch_valid_length[i])
            # 预留出首个块，给冗余写用，全量需要这个 TODO:后续优化ReshapeAndCache逻辑，跳过冗余位置
            # 即缓存块的映射。
            block_table = cache_engine_list[i].block_table
            # padded_table = block_table + [ -1 for _ in range(max_num_blocks_per_seq - len(cache_engine_list[i].block_table) + 1)]
            # padded_table 通过在 block_table 后面填充 -1 来保证其长度达到最大块数量。
            padded_table = block_table + [-1 for _ in
                                          range(max_num_blocks_per_seq - len(cache_engine_list[i].block_table))]
            block_tables.append(padded_table)

            # 计算每个有效 token 在缓存中的位置，将其转换为 block_table 中的插槽映射。
            slots = [block_table[k // block_size] * block_size + k % block_size for k in range(batch_valid_length[i])]
            # 对于序列中未使用的部分，使用 null_slot_idx 进行填充。
            null_slot_idx = 0
            slots = slots + [null_slot_idx for _ in range(seq_length - batch_valid_length[i])]
            slot_mapping = slot_mapping + slots
        block_tables = np.array(block_tables, dtype=np.int32)
        slot_mapping = np.array(slot_mapping, dtype=np.int32)#这个是什么东西？
        return block_tables, slot_mapping

    def _assemble_pa_inc_inputs(self, batch_valid_length: np.array, cache_engine_list, seq_length, valid_batch_flag):
        block_size = cache_engine_list[0].block_size
        max_num_blocks_per_seq = seq_length // block_size
        bs = len(valid_batch_flag)
        block_tables = []
        slot_mapping = []
        for i in range(bs):
            if valid_batch_flag[i]:
                # 调用 prepare_cache(1) 准备增量缓存，即为每个序列增加一个 token。
                cache_engine_list[i].prepare_cache(1)  # 增量推理时，每个序列新增一个token。
                valid_length = cache_engine_list[i].num_token  # - block_size
            else:
                valid_length = 1
            block_table = cache_engine_list[i].block_table
            padded_table = block_table + [-1 for _ in
                                          range(max_num_blocks_per_seq - len(cache_engine_list[i].block_table))]
            block_tables.append(padded_table)
            # 增量推理只增加一个 token，因此 curent_idx = valid_length - 1，并将该索引位置的 slot 记录下来。
            curent_idx = valid_length - 1

            slots = [block_table[curent_idx // block_size] * block_size + curent_idx % block_size]
            slot_mapping = slot_mapping + slots
        block_tables = np.array(block_tables, dtype=np.int32)
        slot_mapping = np.array(slot_mapping, dtype=np.int32)
        return block_tables, slot_mapping

    def call(self, shms: List, input_ids, current_index,
             valid_length, init_reset, is_first_iteration, valid_batch_flag, extra_inputs=None,
             current_batch_size=None, **kwargs):
        """kvcache infer"""
        # 整个推理过程的核心
        time_start = time.time()
        logging.debug("is prefill {}".format(is_first_iteration))
        decode_index_list = kwargs.get("decode_index_list")
        # 加入pa
        # 如果模型配置中启用了页注意力机制，提取 cache_engine_list 和 seq_length
        if self.config.model_config.page_attention:
            cache_engine_list = kwargs.get("cache_engine_list")
            seq_length = kwargs.get("seq_length")
        # 如果是第一次迭代，调用 get_model_inputs 获取推理输入。
        if is_first_iteration:
            lite_inputs = self.get_model_inputs(input_ids, current_index, valid_length,
                                                is_first_iteration, extra_inputs=extra_inputs, **kwargs)
            # 前4个array拼接成一个
            # init_reset 变成[batch_size, 1]
            # first_group, second_group = self.get_warp_inputs(lite_inputs=lite_inputs, **kwargs)
            
            # 初始化 init_reset 并填充 lite_inputs。
            init = 0
            prefill_bs = len(input_ids)
            init_reset = [init for _ in range(prefill_bs)]
            lite_inputs[2] = np.array(init_reset).reshape(prefill_bs, 1).astype(np.int32)

            first_group = np.concatenate((lite_inputs[0], lite_inputs[1].reshape(prefill_bs, 1),
                                          lite_inputs[2], lite_inputs[3].reshape(prefill_bs, 1)), axis=1)
            shape_list = []
            # 将输入的几个部分（ input_ids,current_index,init_reset,valid_length ）拼接成一个大的 NumPy 数组 first_group，并将其写入共享内存 shms[0]。
            first = np.ndarray(first_group.shape, dtype=first_group.dtype, buffer=shms[0].buf)
            first[:] = first_group[:]
            shape_list.append(first_group.shape)

            # 如果是prefill的话，需要将另外三个array也写到共享内存中
            # 处理 second_group （额外的输入数据）并将其写入共享内存。
            second_group = []
            for i in range(4, len(lite_inputs)):
                second_group.append(lite_inputs[i])
            logging.debug("second_group {}".format(second_group))
            if len(second_group) != 0:
                for j in range(len(second_group)):
                    logging.debug("second_group index {}".format(j))
                    item = np.ndarray(second_group[j].shape, dtype=second_group[j].dtype, buffer=shms[1 + j].buf)
                    item[:] = second_group[j][:]
                    shape_list.append(second_group[j].shape)
            # mem_index = len(second_group)
            # 调用 get_gen_parms_np 获取推理参数，并将其写入共享内存。
            params_np_dtype = np.float16
            params_np = self.get_gen_parms_np(prefill_bs, params_np_dtype, **kwargs)
            # gen_index = max(3, mem_index)

            gen_params = np.ndarray(params_np.shape, dtype=params_np_dtype, buffer=shms[4].buf)
            gen_params[:] = params_np[:]

            shape_list.append(params_np.shape)

            shape_strs = []
            for shape in shape_list:
                shape_str = " ".join(str(element) for element in shape)
                shape_strs.append(shape_str)
            shapes_str = "*" + ",".join(element for element in shape_strs)
        else:# decoding
            # 构建形状字符串 shapes_str，包含批次标志信息。
            logging.debug("valid_batch_flag in decode is {}".format(valid_batch_flag))
            batch_flag_str = " ".join(str(element) for element in valid_batch_flag)
            shapes_str = "a" + '_' + str(current_batch_size) + '_' + batch_flag_str

        # 加入pa
        if self.config.model_config.page_attention:
            # 如果启用 page_attention，调用 _assemble_pa_inputs 获取 block_tables 和 slot_mapping，并将其写入共享内存 shms。
            block_tables, slot_mapping = self._assemble_pa_inputs(is_first_iteration, valid_length, cache_engine_list,
                                                                  seq_length, valid_batch_flag)
            block_tables_np = np.array(block_tables, dtype=np.int32)
            block_tables_shm = np.ndarray(block_tables_np.shape, dtype=block_tables_np.dtype, buffer=shms[7].buf)
            block_tables_shm[:] = block_tables_np[:]
            slot_mapping_np = np.array(slot_mapping, dtype=np.int32)
            slot_mapping_shm = np.ndarray(slot_mapping_np.shape, dtype=slot_mapping_np.dtype, buffer=shms[8].buf)
            slot_mapping_shm[:] = slot_mapping_np[:]

            shape_strs = []
            for shape in [block_tables_np.shape, slot_mapping_np.shape]:
                shape_str = " ".join(str(element) for element in shape)
                shape_strs.append(shape_str)
            if is_first_iteration:
                shapes_str += "," + ",".join(element for element in shape_strs)
            else:
                shapes_str += "_" + "_".join(element for element in shape_strs)
        logging.debug("get input lite is {} ".format((time.time() - time_start) * 1000))
        logging.debug("server decode batch size is {} ".format(current_batch_size))
        shapes_str = shapes_str.encode()

        # 将形状数据通过 socket 发送给代理服务器。
        for item in self.agent_stubs:
            item.sendall(shapes_str)
        # 接收来自代理的推理结果并解析。
        recv_data = self.agent_stubs[0].recv(1, socket.MSG_WAITALL).decode()
        logging.debug("recv_data is {} ".format(recv_data))
        
        result = []
        if recv_data == "2":
            for _ in decode_index_list:
                # result.append(int(Baseconfig.end_token))
                result.append((int(-1),0))
            logging.debug("--------------------predict failed, abandon current prompt, please try again----------------")
            logging.error("predict failed, abandon current prompt, please try again")
            return result, 1
        for decode_index in decode_index_list:
            tmp = np.ndarray((decode_index + 1,), dtype=np.int32, buffer=shms[5].buf)
            tmp_logprob = np.ndarray((decode_index + 1,), dtype=np.float64, buffer=shms[6].buf)
            result.append((int(tmp[decode_index:decode_index + 1]), float(tmp_logprob[decode_index:decode_index + 1])))

        logging.info("--------------------callV3 result value is {} ".format(result))
        logging.info("model.call time is {} ".format((time.time() - time_start) * 1000))
        return result, 1
