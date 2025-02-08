import logging
from typing import Dict, List, Tuple, Set
import asyncio

from .utils import ResponseOutput
from .response_async_queue import AsyncResultsOfOneRequest
from mindspore_serving.serving_utils import PADDING_REQUEST_ID


class RequestEngine:
    # 管理生成任务的生命周期，处理请求并管理结果的异步流式输出。
    def __init__(self):
        self._results_streams: Dict[str, AsyncResultsOfOneRequest] = {}         #存储 request_id 和异步结果流的映射"request_id : AsyncResults"
        self._done_requests: asyncio.Queue[str] = asyncio.Queue() # 存储已完成的请求ID队列
        self._new_requests: asyncio.Queue[Tuple[AsyncResultsOfOneRequest, dict]] = asyncio.Queue() # 存储新请求队列

    def process_request_output(self, request_output: ResponseOutput):
        """将一个 ResponseOutput（生成任务的结果）放入对应请求的异步结果的队列中"""
        # 首先获取请求的 request_id。
        request_id = request_output.request_id
        # 如果 request_id 是一个特殊的填充请求（PADDING_REQUEST_ID），则不处理这个请求。
        if request_id == PADDING_REQUEST_ID:
            logging.debug("process padding request output")
            return
        # 否则，将 ResponseOutput 对象放入该 request_id 关联的结果流中（即 AsyncResultsOfOneRequest 的队列）。
        self._results_streams[request_id].put(request_output)
        # 如果 ResponseOutput 指示任务已经完成（request_output.finished 为 True）
        # 则调用 abort_request 方法标记该请求为已完成。
        if request_output.finished:
            self.abort_request(request_id)
    
    def abort_request(self, request_id: str) -> None:
        # 用于中止或结束某个请求，标记它为已完成。
        self._done_requests.put_nowait(request_id)  # 将完成的请求ID放入完成队列
        if request_id not in self._results_streams or \
            self._results_streams[request_id].finished:
            return
        # # 标记请求的结果流完成
        self._results_streams[request_id].finish()
       
    def get_requests_from_register_pool(self) -> Tuple[List[dict], Set[str]]:
        """从注册池中获取新请求和已完成的请求，准备发送到主服务器。"""
       
        new_requests: List[dict] = []
        finished_requests: Set[str] = set()
        # finished requests，pop from results_streams
        # 从完成的请求队列中获取已完成的请求
        while not self._done_requests.empty():
            request_id = self._done_requests.get_nowait()
            finished_requests.add(request_id)
            self._results_streams[request_id] = AsyncResultsOfOneRequest(request_id)
       
        # get a request from _new_request queue
        # 从新请求队列中获取新请求
        while not self._new_requests.empty():
            # request process logic FIFO 
            stream, new_request = self._new_requests.get_nowait()
            # check request status, if finished, put it into finished request set
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._results_streams[stream.request_id] = stream
            new_requests.append(new_request)

        return new_requests, finished_requests

    def register_request(self, request_id: str, **add_request_info) -> AsyncResultsOfOneRequest:
        # 用于注册新的生成请求，并将其放入新请求队列。
        # Check if the request exists
        if request_id in self._results_streams:
            raise KeyError(f"Request {request_id} already exists.")
        # put a new request into queue(asyncio), and return it's results
        # 将新请求放入异步队列中，并返回其结果流
        res_stream = AsyncResultsOfOneRequest(request_id)
        self._new_requests.put_nowait((res_stream, {
            "request_id": request_id,
            **add_request_info
        }))
        return res_stream