import time
import logging
from typing import List
import numpy as np
from multiprocessing import shared_memory #通过多进程共享内存模块来在多个进程间共享内存。
from .model_init_multimodel import DisModel
from mindspore_serving.serving_utils.entry import EntryMetaData, EntryStatus
from mindspore_serving.config.config import ServingConfig
from mindspore_serving.models.build_inputs import build_inputs


# class worker
class Worker:
    def __init__(self, config: ServingConfig) -> None:
        self.model = DisModel()#实例化模型对象 DisModel，该对象负责加载和执行实际的推理模型。
        self.config = config#保存传入的配置对象 ServingConfig，用于管理模型的配置信息。
        self.shms = []#初始化两个空列表，用于存储共享内存对象和共享内存名称。
        self.shm_names = []
        self.valid_length = None# 记录当前推理任务的有效长度、序列长度、批次大小和当前索引。
        self.seq_length = None
        self.batch_size = 1
        self.current_index = 0
        self.vocab_size = config.model_config.vocab_size #模型的词汇表大小，从配置中读取
        self.seq_length_list = config.model_config.seq_length #模型的序列长度列表，从配置中读取。
        self.extra_func = build_inputs(config.extra_inputs, module_type='extra_inputs')#构建额外输入的处理函数。
        # 0 : input_ids, current_index, valid_length, init_reset
        # 1 : mask=mask,
        # 2 : freq_cos
        # 3 : freq_sin
        # 4 : gen_params, top_k top_p ...
        # 5 : predict output
        # 6 : logprob
		# 7 : block table # 128个slot组成一个block
        # 8 : slot mapping #

        # 根据是否启用了 page_attention 决定创建多少个共享内存区域。若启用，则创建 9 个，否则创建 7 个。
        shm_name_num = 9 if self.config.model_config.page_attention else 7  # 根据模型分配
        for i in range(shm_name_num):
            # 每个共享内存块分配 1GB 内存，创建后存储在 shms 列表中。
            tmp = shared_memory.SharedMemory(create=True, size=1024 * 1024 * 1024)
            self.shms.append(tmp)
            # 保存共享内存的名称，以便后续使用。
            self.shm_names.append(tmp.name)

    def _init_worker(self) -> None:
        # 初始化工作者中的模型，并通过共享内存与模型进行通信。
        try:
            self.model.init(self.config, self.shm_names)
        except ConnectionError:
            self.model.reset_agent_status(self.config)
            self.model.init(self.config, self.shm_names)

    @staticmethod
    def _get_seq_length_dynmic_dinning(seq_list, seq_length):
        # 根据输入序列的长度动态调整目标序列长度。
        # 遍历 seq_list（即可能的序列长度列表），找到大于等于当前输入 seq_length 的最小长度并返回。
        for data in seq_list:
            if seq_length < data:
                return data
        return seq_list[-1]

    @staticmethod
    def _padding(origin_inputs, seq_length, default_padding_values):
        #  对输入 tokens 进行填充，使其达到固定的序列长度。
        pad_ids = []
        # 对origin_inputs batch中的每个prompt token list进行padding
        for item in origin_inputs:
            # 对于每个 origin_inputs 中的 item，计算其与目标序列长度 seq_length 的差值 pad_length。
            pad_length = seq_length - len(item)
            if pad_length < 0:
                logging.error('input sequence length is over max in serving system!')
            # 使用 np.pad() 将序列填充到目标长度，使用指定的 default_padding_values 进行填充。
            pad_item = np.pad(item, (0, pad_length), 'constant', constant_values=default_padding_values)
            pad_ids.append(pad_item)

        logging.debug("prefill _padding result list is {}".format(pad_ids))
        # 返回填充后的序列数组。
        return np.array(pad_ids)

    @staticmethod
    def _get_valid_length(origin_inputs, default_padding_values):
        # 计算输入 tokens 中的有效长度，即去除填充部分后实际的序列长度。
        batch_size, _ = origin_inputs.shape
        valid_length_each_example = []
        for i in range(batch_size):
            # As the nonzero returns the index and we need length
            valid_length_each_example.append(np.max(np.argwhere(origin_inputs[i] != default_padding_values)) + 1)
        valid_length = np.array(valid_length_each_example, dtype=np.int32)
        return valid_length, batch_size

    # pa
    def _get_seq_length(self, input_ids, is_prefill):
        # 根据输入 input_ids 和是否为预填充操作 is_prefill，确定序列长度。
        max_length = 0
        # 如果当前不是预填充，并且启用了 page_attention，则直接返回解码序列长度 decode_seq_length。
        if not is_prefill:
            if self.config.model_config.page_attention:
                return self.config.pa_config.decode_seq_length
        # 如果 item 是列表，更新 max_length 为列表中prompt的最大长度。
        for item in input_ids:
            if isinstance(item, list):
                max_length = max(max_length, len(item))
            else:
                max_length = max(max_length, 1)
        # 确定序列长度的具体值。如果配置了动态序列 seq_type == 'dyn'，则使用 max_length 作为序列长度。这里是static
        if self.config.model_config.seq_type == 'dyn':
            seq_length = max_length
        # 否则，调用 _get_seq_length_dynmic_dinning 根据 max_length 动态确定序列长度。
        elif len(self.config.model_config.seq_length) > 1:
            seq_length = self._get_seq_length_dynmic_dinning(self.seq_length_list, max_length)
        else:
            if len(self.config.model_config.seq_length) == 0 and self.config.model_config.seq_type != 'dyn':
                logging.error('seq length is None ! using default 2048')
                seq_length = 2048
            else:
                seq_length = self.config.model_config.seq_length[0]
        return seq_length

    def _predict(self,
                 input_ids: List[List[int]],
                 is_prefill: bool,
                 valid_batch_flag: List[int],
                 current_batch_size=None,
                 **generate_parms) -> List:
        # 执行模型的推理操作，并处理输入数据，如计算序列长度、填充等。
        time_start = time.time()
        # Init outputs with original inputs
        seq_length = self._get_seq_length(input_ids, is_prefill)
        logging.info("decode_seq_length: %s", seq_length)
        generate_parms["seq_length"] = seq_length
        # 处理预填充（prefill）阶段的输入，包括填充、计算有效长度以及当前索引。
        if is_prefill:
            default_padding_values = 0
            if self.config.model_config.pad_token_id:
                default_padding_values = self.config.model_config.pad_token_id
            # 如果是预填充阶段，将 input_ids 使用 _padding() 方法进行填充，填充值为 pad_token_id。
            input_ids = self._padding(input_ids, seq_length, default_padding_values)
            # 计算每个序列的有效长度。 这个过程是不是有点和上面的多余，本来就是已知的，非要加上padding再算？
            self.valid_length, self.batch_size = self._get_valid_length(input_ids, default_padding_values)
            # 计算当前索引位置，基于有效长度减 1，并以批次大小为单位更新 self.current_index。
            current_index_ = [self.valid_length[i] - 1 + i * seq_length for i in range(self.batch_size)]
            self.current_index = np.array(current_index_, np.int32)
        # If target length exceeds seq_length, use seq_length instead
        # A list of the frequency of each token
        # For first graph, not_init should be false
        # 确定 init 标志，该标志用于控制模型是否在首次执行推理时进行初始化。
        init_true = True
        init = init_true and not is_prefill

        logging.info("pre-process time is {} ".format((time.time() - time_start) * 1000))
        # 获取额外的输入，例如注意力掩码或频率向量。
        mask_time = time.time()
        extra_input_list = self.extra_func.get_extra_inputs(input_ids, self.current_index, init, is_prefill,
                                                            self.valid_length,
                                                            zactivate_len=self.config.model_config.zactivate_len)
        if extra_input_list is None:
            logging.error('extra inputs by customer is None,please check it in server config!')
        logging.info("mask time is {} ".format((time.time() - mask_time) * 1000))
        # Call a single inference with input size of (bs, seq_length)
        # 调用模型的推理函数，进行实际的推理计算。传入共享内存 shms、输入 input_ids、当前索引、有效长度、初始化标志 init、是否是预填充等参数。
        call = time.time()
        # 返回推理结果 result 和共享内存对象 shm。
        result, shm = self.model.call(self.shms, np.array(input_ids, np.int32), self.current_index,
                                      self.valid_length, init, is_prefill, valid_batch_flag,
                                      extra_inputs=extra_input_list, current_batch_size=current_batch_size,
                                      **generate_parms)
        if is_prefill:
            logging.info("PrefillTime {} ".format((time.time() - call) * 1000))
        else:
            logging.info("DecodeTime {} ".format((time.time() - call) * 1000))
        return result

    @staticmethod
    def get_generate_parms(page_attention, entry_metadata_list):
        # 从 entry_metadata_list 中提取生成任务的参数，组织成列表。
        do_sample_list = []
        top_k_list = []
        top_p_list = []
        temperature_list = []
        repetition_penalty = []
        decode_index_list = []
        cache_engine_list = []
        for item in entry_metadata_list:
            entry_data = item.get_entry_data()
            do_sample_list.append(entry_data.do_sample)
            top_k_list.append(entry_data.top_k)
            top_p_list.append(entry_data.top_p)
            temperature_list.append(entry_data.temperature)
            repetition_penalty.append(entry_data.repetition_penalty)
            decode_index_list.append(entry_data.decode_index)
            if page_attention:
                cache_engine_list.append(item.cache_engine)
        parms = {
            "do_sample_list": do_sample_list,
            "top_k_list": top_k_list,
            "top_p_list": top_p_list,
            "temperature_list": temperature_list,
            "repetition_penalty": repetition_penalty,
            "decode_index_list": decode_index_list,
        }
        if page_attention:
            parms["cache_engine_list"] = cache_engine_list
        return parms

    def predict(self, current_batch_size, entry_metadata_list: List[EntryMetaData]):
        # 执行模型推理，根据元数据列表构建输入并调用 _predict 方法。
        if_prefill = entry_metadata_list[0].is_prompt
        inputs_ids = []  # length is batch size
        valid_batch_flag = []
        # 遍历列表，将list中的有效数据写入到inputids和根据任务状态设置batch flag
        for item in entry_metadata_list:
            entry_data = item.get_entry_data()
            token_ids = entry_data.get_all_tokens()
            if if_prefill:
                # 这是一个二维列表，将prompt tokenid加到列表中
                inputs_ids.append(token_ids)
            else:
                inputs_ids.append(token_ids[-1])
            if entry_data.get_status() == EntryStatus.RUNNING:
                valid_batch_flag.append(1)
            else:
                valid_batch_flag.append(0)
        # 将准备推理的list中的任务的推理请求参数和cache engine 加入到list中
        generate_parms = self.get_generate_parms(self.config.model_config.page_attention, entry_metadata_list)
        current_batch_size_dyn = current_batch_size
        # 将请求的tokenid 的二维list，是否prefill，有效batch的flag，当前decoding的batch数，还有batch任务的请求参数进行推理
        outputs = self._predict(inputs_ids, if_prefill, valid_batch_flag, current_batch_size=current_batch_size_dyn,
                                **generate_parms)

        return outputs

    def stop(self):
        self.model.stop()
        for shm in self.shms:
            shm.close()
