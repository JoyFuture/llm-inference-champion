import logging
from typing import List

from mindspore_serving.serving_utils.constant import *
from mindspore_serving.serving_utils.entry import EntryMetaData, EntryStatus


class CompletionOutput:
    """
    用于存储生成模型在某一步中生成的输出信息，包括token的索引、生成的文本、该token的对数概率，以及是否为特殊token。
       index : current index of output token
       text: text of current iteration step generation
       token_ids: all tokens of one request so for.
    """

    def __init__(
            self,
            index: int,
            special: bool,
            logprob: float,
            text: str
    ) -> None:
        self.index = index# 当前生成的token的索引
        self.text = text# 当前生成的文本
        self.logprob = logprob# 生成这个token的对数概率（log probability）
        self.special = special# 是否为特殊token（例如结束符或其他控制token）


class ResponseOutput:
    """output put in ascynio queue used by stream return"""
    """用于异步队列返回的输出"""
    def __init__(
            self,
            request_id: str,
            prompt: str,
            prompt_token_ids: List[int],
            outputs: List[CompletionOutput],
            finished: bool,
            finish_reason: str,
            output_tokens_len: int
    ) -> None:
        self.request_id = request_id# 请求的唯一标识
        self.prompt = prompt# 请求的输入（prompt）
        self.prompt_token_ids = prompt_token_ids# 输入的token id列表
        self.outputs = outputs# 生成的CompletionOutput对象列表
        self.finished = finished# 表示生成是否完成
        self.finish_reason = finish_reason# 生成完成的原因
        self.output_tokens_len = output_tokens_len# 生成的输出token的长度

    @classmethod
    def generate_result(cls,# 指代当前类，即 ResponseOutput 类本身。使用 cls 可以在类方法内部创建类的实例。
                        output_token: int,
                        output_token_logprob: float,# 生成这个token的对数概率，代表模型对该token的信心值。
                        entry_meta_data: EntryMetaData,# 包含生成过程元数据的对象，包括请求的ID、生成状态等
                        output_str: str,# 生成的文本，模型在某次迭代中生成的实际内容
                        eos_id,
                        reason: str = None):
        # 根据生成的token和状态创建一个ResponseOutput实例。
        # 可以通过类本身直接调用，而不需要实例化对象。
        # 这个方法的目的是根据模型的生成状态动态创建并返回 ResponseOutput 实例。
        output_tokens_len = entry_meta_data.get_entry_data().get_output_len()
        request_id = entry_meta_data.request_id
        status = entry_meta_data.get_entry_data().get_status()
        finished_reason = None
        finished = False

        # 判断生成是否完成,如果当前状态不是RUNNING或WAITING，且不是等待批次中的状态，则根据状态获取完成原因。
        if status != EntryStatus.RUNNING or status != EntryStatus.WAITING:
            if status != EntryStatus.WAITING_BATCH:
                finished_reason = EntryStatus.get_finished_reason(status)
        # 如果生成的token为空，意味着生成结束，设置状态为FINISHED_STOPPED。
        if output_token == INPUT_EMPTY_TOKEN[0]:
            finished = True
            completion_out = CompletionOutput(0, text=reason, logprob=0.0, special=False)
            entry_meta_data.get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            return cls(request_id,
                       entry_meta_data.get_prompt(),
                       entry_meta_data.get_entry_data().get_prompt_token(),
                       [completion_out],
                       finished,
                       "prompt_token_ids_empty",
                       0)
        # 输入超出模型的处理范围，生成任务被终止。
        if output_token == INPUT_OUT_OF_TOKEN[0]:
            print(f'>>>>>>request {request_id} input is too large, out of input length of model')
            finished = True
            completion_out = CompletionOutput(0, text=reason, logprob=0.0, special=False)
            entry_meta_data.get_entry_data().set_status(EntryStatus.FINISHED_STOPPED)
            return cls(request_id,
                       entry_meta_data.get_prompt(),
                       entry_meta_data.get_entry_data().get_prompt_token(),
                       [completion_out],
                       finished,
                       "prompt_out_of_range",
                       0)

        if status == EntryStatus.PADDING_INVAILED:
            finished = False

        elif output_token == PREDICT_FAILED_CODE:
            entry_meta_data.get_entry_data().prefix_index = 0
            entry_meta_data.get_entry_data().read_index = 0
            output_str = RETURN_REASON_PREDICT_FAILED
            finished = True
            finished_reason = "predict_failed"
        # 如果生成的token是结束符，表示生成完成。
        elif output_token == eos_id:
            entry_meta_data.get_entry_data().prefix_index = 0
            entry_meta_data.get_entry_data().read_index = 0
            finished = True
            finished_reason = "eos"

        elif entry_meta_data.entry_data.get_output_len() == entry_meta_data.entry_data.max_token_len:
            entry_meta_data.get_entry_data().prefix_index = 0
            entry_meta_data.get_entry_data().read_index = 0
            logging.debug("stop inference because of iteration is max_len, request is {}".format(request_id))
            finished = True
            finished_reason = "length"

        is_special = True#判断是否为特殊token
        if output_str != "":
            is_special = False

        completion_out = CompletionOutput(output_token, text=output_str, logprob=output_token_logprob, special=is_special)

        return cls(request_id,
                   entry_meta_data.get_prompt(),
                   entry_meta_data.get_entry_data().get_prompt_token(),
                   [completion_out],
                   finished,
                   finished_reason,
                   output_tokens_len)


class Counter:
    # 跟踪生成过程中的某些计数
    def __init__(self, start=0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


class ModelInfo:
    # 存储模型的配置信息，包括模型的ID、最大输入长度、批处理的最大token数、并发处理的最大请求数等。
    def __init__(
            self,
             docker_label: str,
             max_batch_total_tokens: int,
             max_concurrent_requests: int,
             max_input_length: int,
             max_total_tokens: int,
             model_dtype: str,
             model_id: str
    ) -> None:
        self.docker_label = docker_label
        self.max_batch_total_tokens = max_batch_total_tokens
        self.max_concurrent_requests = max_concurrent_requests
        self.max_input_length = max_input_length
        self.max_total_tokens = max_total_tokens
        self.max_batch_total_tokens = max_batch_total_tokens
        self.model_dtype = model_dtype
        self.model_id = model_id