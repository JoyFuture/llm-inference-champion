from tabnanny import check
from typing import List, Optional, Tuple
import copy
import time
import random
import logging

from mindspore_serving.serving_utils.entry import EntryData, EntryMetaData, EntryStatus
from .utils import Counter, ResponseOutput

from mindspore_serving.schedule.schedule import Schedule
from mindspore_serving.worker.worker import Worker
from mindspore_serving.config.config import ServingConfig
from mindspore_serving.models.build_tokenizer import build_tokenizer
from mindspore_serving.schedule.cache_engine import ServingBlockMemPool
from mindspore_serving.serving_utils.constant import *
from mindformers.mindformer_book import MindFormerBook

Eps = 30


class Master:
    def __init__(self,
                 config: ServingConfig):
        self.config = config# 接收并存储传入的 ServingConfig 对象，包含模型和服务的配置信息。
        self.tokenizer = None# 初始化时，分词器设为 None，后面会调用 _init_tokenizer() 方法进行初始化。
        self.counter = Counter()# 初始化一个计数器 Counter，用于生成任务的计数，可能用于生成唯一 ID。
        self.worker = Worker(config)# 创建 Worker 对象，它使用传入的配置来处理实际的模型推理任务。
        self.scheduler = Schedule(config)# 创建调度器 Schedule，负责管理推理任务的调度和分配。

        self.is_running = False#设置一个标志位 is_running，表示当前系统是否正在运行。
        self._init_workers()#调用私有方法 _init_workers()，用于初始化工作进程。
        self._counter_of_token = 0#初始化一个用于统计生成的 token 数量的计数器。
        self._init_tokenizer()#调用私有方法 _init_tokenizer()，用于初始化分词器。
        self.decode_cache = {}#初始化一个解码缓存，存储生成过程中的中间解码结果。
        if self.config.model_config.page_attention:
            self._init_mem_pool()  # PA初始化内存池。

    # PA

    def _init_mem_pool(self):
        # 初始化内存池
        ServingBlockMemPool.init(self.config.pa_config.num_blocks, self.config.pa_config.block_size)

    def _init_tokenizer(self):
        # 调用 build_tokenizer() 函数并传入配置来构建分词器，将生成的分词器实例赋值给 self.tokenizer。
        self.tokenizer = build_tokenizer(self.config)
        if self.tokenizer is None:
            logging.error('load tokenizer failed!')
        logging.debug(f'self.tokenizer is {self.tokenizer}')

    def _init_workers(self):
        # 初始化工作者（Worker）用于处理推理任务。
        self.worker._init_worker()

    def _schedule(self) -> Tuple[List[EntryMetaData], int]:
        # 调用调度器 self.scheduler.schedule()，返回包含 EntryMetaData 列表和批处理大小的元组。
        return self.scheduler.schedule()

    def get_number_of_total_tokens(self):
        # 返回当前生成的总 token 数量
        return self._counter_of_token

    def _detokenizer(self, tokens: List[int]) -> List[str]:
        # 返回解码后的字符串列表
        """
           tokens is results of post-sampling module.
           output: texts list of batch
        """
        texts = []
        for token in tokens:
            token_input = [token]
            text = self.tokenizer.decode(token_input, skip_special_tokens=True)
            logging.debug(f'tokenizer decode result is {text}, token id is {token}')
            texts.append(text)
        return texts

    def _llama_detokenizer(self, outputs):
        # 用于处理 LLAMA 模型生成的 tokens 输出，并将其解码为文本。
        str_outputs = []
        batch_size = len(outputs)
        before_batch_size = len(self.decode_cache.keys())
        # 如果当前批处理大小比之前的缓存小，缓存将被调整
        if batch_size > before_batch_size:
            for i in range(before_batch_size, batch_size):
                self.decode_cache[i] = []
        else:
            while len(self.decode_cache.keys()) > batch_size:
                self.decode_cache.popitem()
        # 解码过程
        for i in range(batch_size):
            self.decode_cache[i].append(outputs[i])
            new_text = self.tokenizer.decode(self.decode_cache[i], skip_special_tokens=True)
            if not new_text.endswith("�"):
                begin_token = self.tokenizer._convert_id_to_token(self.decode_cache[i][0])
                if begin_token == '<0x0A>':
                    begin_token = '\n'
                elif '\u2581' in begin_token:
                    begin_token = ' '
                else:
                    begin_token = ''

                str_outputs.append(begin_token + new_text)
                self.decode_cache[i] = []
            else:
                str_outputs.append('')
        return str_outputs

    def _llama_detokenizer_function(self, index, entry_metadata_list, skip_special_tokens=True):
        # : 针对特定 LLAMA 任务的批次索引，将生成的 tokens 转换为文本，并在每一批次生成的基础上更新解码状态。

        # 获取当前生成任务的 prefix_index 和 read_index，它们分别标记输出的起始和读取位置。
        prefix_index = entry_metadata_list[index].get_entry_data().prefix_index
        read_index = entry_metadata_list[index].get_entry_data().read_index
        if entry_metadata_list[index].get_entry_data().get_status() != EntryStatus.RUNNING:
            return ""
        all_outputs_ids = entry_metadata_list[index].get_entry_data().get_output_token()
        # 解码 tokens 列表。
        prefix_text = self.tokenizer.decode(all_outputs_ids[prefix_index: read_index],
                                            skip_special_tokens=skip_special_tokens)

        new_text = self.tokenizer.decode(all_outputs_ids[prefix_index:], skip_special_tokens=skip_special_tokens)

        # 如果新解码的文本比之前的前缀文本长且没有乱码，则更新索引，并返回新生成的文本。
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            new_text = new_text[len(prefix_text):]
            entry_metadata_list[index].get_entry_data().prefix_index = read_index
            entry_metadata_list[index].get_entry_data().read_index = len(all_outputs_ids)
            return new_text
        else:
            return ""

    def _llama_detokenizer_v2(self,
                              outputs,
                              entry_metadata_list,
                              index_list=None,
                              skip_special_tokens=True):
        #   根据传入的 index_list 或 outputs，批量处理生成任务的文本解码。
        str_outputs = []
        # prompt
        if index_list is not None:
            for index in index_list:
                str_outputs.append(self._llama_detokenizer_function(index, entry_metadata_list, skip_special_tokens))
            return str_outputs
        # decode
        for index, output in enumerate(outputs):#解码过程有必要全过一次吗？29949，或者什么办法过滤一下。
            str_outputs.append(self._llama_detokenizer_function(index, entry_metadata_list, skip_special_tokens))
        return str_outputs

    def _check_error_code(self, output_token):
        # 检查生成的 output_token 是否包含错误码。
        error_code_list = [-1, -202, -203]
        if output_token in error_code_list:
            return True
        return False

    def _postprocess(self,
                     outputs: List[tuple],
                     entry_metadata_list: List[EntryMetaData],
                     index_list: List[int] = None,
                     skip_inference=False) -> List[ResponseOutput]:
        # 处理推理的输出 outputs，并根据推理的结果生成相应的 ResponseOutput 对象。
        end_token = self.config.model_config.end_token  # for debug

        output_tokens = []
        output_logprob = []
        for output_tup in outputs:
            output_tokens.append(output_tup[0])
            output_logprob.append(output_tup[1])

        # 调用调度器 scheduler 的 upate_entries_after_one_step 方法，更新生成任务的状态。
        self.scheduler.upate_entries_after_one_step(output_tokens, end_token, index_list)
        # 检查当前模型是否为 LLAMA 或 wizard_coder 系列，并确保生成的 token 没有错误。
        str_outputs = [''] * len(output_tokens)
        # 调用 _llama_detokenizer_v2() 进行解码。
        if (self.config.model_config.model_name.startswith(
                'llama') or self.config.model_config.model_name == 'wizard_coder') and not self._check_error_code(
            output_tokens[0]):
            # str_outputs = self._llama_detokenizer(outputs)
            str_outputs = self._llama_detokenizer_v2(output_tokens, entry_metadata_list,
                                                     index_list, skip_special_tokens=True)

        elif self.config.model_config.model_name in (
                'internlm_7b', 'baichuan2pa', 'gpt2') and not self._check_error_code(output_tokens[0]):
            str_outputs = self._detokenizer(output_tokens)
        # 更新总生成 token 的计数器 _counter_of_token。
        self._counter_of_token += len(output_tokens)
        logging.debug("target is {}, str_outputs is {}".format(outputs, str_outputs))
        logging.debug("current total token numbers is {}".format(self._counter_of_token))
        # generating output
        results: List[ResponseOutput] = []
        # prompt result
        # 针对存在的 index_list 处理生成结果。
        if index_list is not None:
            # idx: index_list and outputs data index, index: batch list index.
            # 遍历 index_list，对每个索引 index 进行处理。
            for idx, index in enumerate(index_list):
                # 如果生成的任务状态为 PADDING_INVAILED，跳过该条目。
                if entry_metadata_list[index].entry_data.status == EntryStatus.PADDING_INVAILED:
                    logging.debug(f'generate a invalid token, index in batch is {index}')
                    continue
                # 表示输入超出范围，生成相应的 ResponseOutput，并记录错误。
                if output_tokens[0] == INPUT_OUT_OF_TOKEN[0]:
                    logging.debug(f'input out of range, index in batch is {index}')
                    results.append(ResponseOutput.generate_result(output_tokens[idx],
                                                                  0,
                                                                  entry_metadata_list[index],
                                                                  str_outputs[idx],
                                                                  end_token, reason='Error202: prompt out of range'))
                    return results

                # 表示提示词为空，生成相应的错误响应。
                if output_tokens[0] == INPUT_EMPTY_TOKEN[0]:
                    logging.debug(f'prompt token empty, index in batch is {index}')
                    results.append(ResponseOutput.generate_result(output_tokens[idx],
                                                                  0,
                                                                  entry_metadata_list[index],
                                                                  str_outputs[idx],
                                                                  end_token, reason='Error203: prompt token empty'))
                    return results

                # 使用生成的 token 和 logprob 创建 ResponseOutput 对象，并将其加入 results 列表。
                results.append(ResponseOutput.generate_result(output_tokens[idx],
                                                              output_logprob[idx],
                                                              entry_metadata_list[index],
                                                              str_outputs[idx],
                                                              end_token))
        # encode result
        # 如果 index_list 为空，按批次索引直接处理每个生成的 output_token。
        else:
            for index, output_token in enumerate(output_tokens):
                output_token_logprob = output_logprob[index]
                if entry_metadata_list[index].entry_data.status == EntryStatus.PADDING_INVAILED:
                    logging.debug(f'generate a invalid token, index in batch is {index}')
                    continue
                results.append(ResponseOutput.generate_result(output_token,
                                                              output_token_logprob,
                                                              entry_metadata_list[index],
                                                              str_outputs[index],
                                                              end_token))
        return results

    def get_current_batch(self):
        return self.scheduler.get_dyn_batch()

    def get_current_requestes_nums(self):
        return self.scheduler.get_queue_len()

    def abort_request(self,
                      request_id: str) -> None:
        logging.debug(f"Abort request called with request_id: {request_id} of type {type(request_id)}")
        self.scheduler.abort_entry(request_id)

    def add_requests_to_schedule_pool(self,
                                      request_id: str,
                                      prompt: Optional[str],
                                      do_sample,
                                      top_k,
                                      top_p,
                                      temperature,
                                      repetition_penalty,
                                      max_token_len):
        # 将一个新的请求添加到调度池中，该请求包含生成模型所需的参数和输入提示词。
        time_tokenizer = time.time()
        prompt_token_ids = None
        logging.debug("request id add_requests_to_schedule_pool {}".format(request_id))
        # 加入baichuan
        # 直接调用 self.tokenizer.encode(prompt) 进行分词。
        if self.config.model_config.model_name in (
                'baichuan2pa', 'wizard_coder') or self.config.model_config.model_name.startswith('llama'):
            prompt_token_ids = self.tokenizer.encode(prompt)
        elif self.config.model_config.model_name == 'internlm_7b':
            prompt_token_ids = self.tokenizer(prompt)['input_ids'][1:]
        elif self.config.model_config.model_name in MindFormerBook.get_tokenizer_support_list():
            prompt_token_ids = self.tokenizer(prompt)['input_ids']
        else:
            print('incorrect model_name')
            logging.debug('incorrect model_name')

        logging.info('tokenizer result prompt_token_ids is {}'.format(prompt_token_ids))
        logging.info('tokenizer time is {}'.format((time.time() - time_tokenizer) * 1000))

        # if prompt_token_ids is not None and
        # Create the sequences.
        entry_id = next(self.counter)
        entry_data = EntryData(prompt_tokens=prompt_token_ids,
                               max_token_len=max_token_len,
                               do_sample=do_sample,
                               tok_k=top_k,
                               top_p=top_p,
                               temperature=temperature,
                               repetition_penalty=repetition_penalty)
        block_size = 0
        if self.config.model_config.page_attention:
            block_size = self.config.pa_config.block_size
        entry_meta_data = EntryMetaData(page_attention=self.config.model_config.page_attention,
                                        request_id=request_id,
                                        is_prompt=True,
                                        entry_data=entry_data,
                                        entry_id=entry_id,
                                        prompt=prompt,
                                        block_size=block_size)

        logging.debug("add request to schedule queue {}".format(entry_meta_data.request_id))
        # 调用调度器的 add_entrys 方法，将生成任务添加到调度队列中。
        self.scheduler.add_entrys(entry_meta_data)

    def step(self) -> List[ResponseOutput]:
        # 执行推理任务，返回模型的输出结果。
        # do inference
        # 获取任务元数据列表 entry_metadata_list 和批处理大小 batch_size。
        entry_metadata_list, batch_size = self._schedule()
        # Execute the model.
        # output: model infer out(token):
        # output is batch_size * n_src_vocab
        # 模拟模型推理，生成 tokens 输出。
        output = self._mock_run_workers_async(batch_size)
        # 处理生成的 tokens，将其转换为 ResponseOutput 对象，并返回给调用方。
        return self._postprocess(output, entry_metadata_list)

    def _mock_run_workers_async(self, batch_size: int):
        # 该方法是模拟推理操作，在没有实际模型推理时，生成随机的 token 作为输出。
        outputs = []
        for i in range(batch_size):
            output = random.randint(0, 32000)
            outputs.append(output)
        time.sleep(0.15)
        return outputs


class AsyncMaster(Master):
    # Master 类的异步版本，支持异步处理推理任务。
    async def step_async(self) -> List[ResponseOutput]:
        # 调用调度器的 _schedule() 方法获取任务元数据列表 (entries_metadata_list) 和当前批次的大小 (current_batch_size)。
        entries_metadata_list, current_batch_size = self._schedule()
        valid_entry_len = 0
        # 检查每个生成任务的状态，统计处于 RUNNING 或 INPUT_OUTOFRANGE 状态的任务数量。
        for metadata in entries_metadata_list:
            # 推理还在进行或者提示词长度超出范围，这两个状态下的任务计入 valid_entry_len
            if metadata.entry_data.get_status() == EntryStatus.RUNNING or \
                    metadata.entry_data.get_status() == EntryStatus.INPUT_OUTOFRANGE:
                valid_entry_len += 1#那直接可以直接break嘛，没必要算出来多少个可以执行
        # 如果当前批次中没有有效的推理任务（即 valid_entry_len == 0），则返回 None，表示没有可处理的任务。
        if valid_entry_len == 0:
            return None
        # 异步调用 _run_workers_async 执行推理任务，并返回模型的推理结果。
        output = await self._run_workers_async(current_batch_size, entry_metadata_list=entries_metadata_list)
        return output

    def _get_prompt_batch_list(self, entry_metadata_list):
        # PA取一个data进行prefill，如果不是pa，那就能处理多个batch
        # if self.config.model_config.page_attention:
        #     return self._check_prompt_predict_data_pa(entry_metadata_list)
        input_entry_metadata_list, index_list = self._check_prompt_predict_data(entry_metadata_list)
        prompt_data_count = len(input_entry_metadata_list)

        if prompt_data_count == 0:
            return input_entry_metadata_list, index_list
        logging.debug("_get_prompt_batch_list prompt index_list {}, input_entry_metadata_list {}"
                      .format(index_list, input_entry_metadata_list))

        prefill_batch_size_list = self.config.model_config.prefill_batch_size
        # 如果批次大小列表是 None 或为空列表，代码会返回第一个输入元数据和对应的索引。此时仅处理一条数据。
        if prefill_batch_size_list is None or len(prefill_batch_size_list) == 0:
            return [input_entry_metadata_list[0]], [index_list[0]]
        else:  # pure dyn
            # dyn_bach_size 是批次大小列表中的第一个元素，它定义了可以处理的最大批次大小。
            # 如果 prompt_data_count 大于 dyn_bach_size，
            # 则截取 input_entry_metadata_list 和 index_list 中前 dyn_bach_size 条数据。
            dyn_bach_size = prefill_batch_size_list[0]
            if prompt_data_count > dyn_bach_size:
                input_entry_metadata_list = input_entry_metadata_list[:dyn_bach_size]
                index_list = index_list[:dyn_bach_size]

        return input_entry_metadata_list, index_list

    @staticmethod
    def get_last_prompt_entry(entry_metadata_list):
        # 从 entry_metadata_list 中找到最后一个提示词任务。
        for i in range(len(entry_metadata_list) - 1, -1, -1):
            # 从列表末尾开始向前遍历，找到第一个 is_prompt 为 True 的任务元数据并返回。
            entry_meta_data = entry_metadata_list[i]
            if entry_meta_data.is_prompt:
                return entry_meta_data

    @staticmethod
    def _get_prefill_padding_entry(index, entry_meta_data):
        # 创建一个新的无效填充任务，并将其状态设置为 PADDING_INVAILED。
        copy_entry_meta_data = copy.deepcopy(entry_meta_data)
        copy_entry_meta_data.get_entry_data().set_status(EntryStatus.PADDING_INVAILED)
        copy_entry_meta_data.get_entry_data().set_decode_index(index)
        logging.debug(f'add invalid request into prefill batch list, batch size index is {index}')
        return copy_entry_meta_data

    @staticmethod
    def _check_prompt_out_of_range_index_list(entry_metadata_list):
        # 检查每个任务的状态，返回提示词超出范围 (INPUT_OUTOFRANGE) 的任务索引列表。
        """check prompt out of range index list"""
        out_of_range_index_list = []
        # for item in entries_metadata_list:
        for index, item in enumerate(entry_metadata_list):
            if not item.is_prompt or item.entry_data.status != EntryStatus.INPUT_OUTOFRANGE:
                continue

            out_of_range_index_list.append(index)
        return out_of_range_index_list

    @staticmethod
    def _check_prompt_predict_data_pa(entry_metadata_list):
        # 处理 page_attention 模式下的提示词任务，检查任务列表中的每个项目，返回有效的提示词任务及其索引。
        input_entry_metadata_list = []
        index_list = []
        for index, item in enumerate(entry_metadata_list):
            # 遍历 entry_metadata_list，如果任务不是提示词或其状态是 INPUT_OUTOFRANGE，则跳过。
            if not item.is_prompt or item.entry_data.status == EntryStatus.INPUT_OUTOFRANGE:
                continue
            # 否则，将该任务和对应的索引作为单个列表返回，处理一个提示词任务后即结束循环（只处理一个任务）。
            input_entry_metadata_list = [item]
            index_list = [index]
            break
        return input_entry_metadata_list, index_list

    @staticmethod
    def _check_prompt_predict_data(entry_metadata_list):
        # 检查提示词任务列表并过滤有效的提示词数据，返回有效任务的元数据和索引。
        input_entry_metadata_list = []
        index_list = []
        for index, item in enumerate(entry_metadata_list):
            # 如果任务不是提示词或其状态为 INPUT_OUTOFRANGE，则跳过。
            if not item.is_prompt or item.entry_data.status == EntryStatus.INPUT_OUTOFRANGE:
                continue

            input_entry_metadata_list.append(item)
            index_list.append(index)
        return input_entry_metadata_list, index_list

    @staticmethod
    def _check_prompt_token_empty(entry_metadata_list, pad_token_id):
        # 检查提示词 tokens 是否为空或者是否包含填充符号 pad_token_id，设置为对应空提示任务并返回对应的空任务索引列表。
        # 但此时tokenid list里面是个数字2，并不符合这个判空的函数
        empty_list = []
        for index, item in enumerate(entry_metadata_list):
            # 获取entrydata中的tokenid list看是否为None 或者 tokenid list的长度是否为0 如果是空的，那么那就
            if item.get_entry_data().get_prompt_token() == None or item.get_entry_data().get_prompt_len() == 0:
                item.get_entry_data().set_status(EntryStatus.EMPTY_PROMPT_TOKEN)
                empty_list.append(index)
            if pad_token_id in item.get_entry_data().get_prompt_token():
                item.get_entry_data().set_status(EntryStatus.EMPTY_PROMPT_TOKEN)
                empty_list.append(index)
        return empty_list

    async def _run_workers_async(self, current_batch_size, entry_metadata_list):
        # 异步推理方法，用于处理当前批次的推理任务。
        e_t_e_time = time.time()

        # 检查提示词是否为空。
        # 如果发现提示词为空的任务，跳过推理步骤，直接调用 _postprocess() 方法处理空的输入，并返回空 token 的结果。
        prompt_token_empty_list = self._check_prompt_token_empty(entry_metadata_list,
                                                                 self.config.model_config.pad_token_id)
        logging.debug("prompt token empty list index_list {}".format(prompt_token_empty_list))
        if len(prompt_token_empty_list) > 0:
            return self._postprocess([INPUT_EMPTY_TOKEN], entry_metadata_list=entry_metadata_list,
                                     index_list=prompt_token_empty_list,
                                     skip_inference=True)

        # check prefill out of range data
        # 检查任务是否有提示词超出范围的情况。
        out_of_range_index_list = self._check_prompt_out_of_range_index_list(entry_metadata_list)
        logging.debug("out of range prompt index_list {}".format(out_of_range_index_list))
        if len(out_of_range_index_list) > 0:
            return self._postprocess([INPUT_OUT_OF_TOKEN], entry_metadata_list=entry_metadata_list,
                                     index_list=out_of_range_index_list,
                                     skip_inference=True)

        # filter prompt data batch list
        # 获取当前批次中有效的提示词任务及其索引列表。
        input_entry_metadata_list, index_list = self._get_prompt_batch_list(entry_metadata_list)
        logging.debug("_get_prompt_batch_list prompt index_list {}, input_entry_metadata_list {}"
                      .format(index_list, input_entry_metadata_list))
        # prefill predict
        # 如果有效任务数量大于 0，调用 worker.predict() 执行预填充推理任务。
        if len(input_entry_metadata_list) >0:
            output = self.worker.predict(current_batch_size, entry_metadata_list=input_entry_metadata_list)
        else:
            input_entry_metadata_list = entry_metadata_list
            index_list = None
            logging.debug('decode len of input entry_metadata_list is {}'.format(len(input_entry_metadata_list)))
            output = self.worker.predict(current_batch_size, entry_metadata_list=input_entry_metadata_list)


        post_process_time = time.time()
        # 推理完成后，调用 _postprocess() 对推理结果进行后处理。这里是否能确认一下这个indexlist，不再需要全部遍历后处理
        result = self._postprocess(output, entry_metadata_list=entry_metadata_list, index_list=index_list)
        logging.info('post_process_time time is {}'.format((time.time() - post_process_time) * 1000))
        logging.info('e-to-e time is {}'.format((time.time() - e_t_e_time) * 1000))
        return result

    async def _mock_run_workers_async(self, batch_size: int):
        # 这是一个模拟推理的方法，用于生成随机 token，模拟模型的推理输出
        outputs = []
        for i in range(batch_size):
            output = random.randint(0, 32000)
            outputs.append(output)
        return outputs

    def stop(self):
        # 调用 worker 的 stop() 方法停止工作进程
        self.worker.stop()
