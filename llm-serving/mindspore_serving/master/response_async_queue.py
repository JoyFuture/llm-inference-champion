from queue import Queue
import asyncio
from .utils import Counter, ResponseOutput


class AsyncResultsOfOneRequest:
    # 管理异步生成任务的结果
    def __init__(self, request_id: str):

        self.request_id = request_id
        self._queue:asyncio.Queue[ResponseOutput] = asyncio.Queue()    # 用于保存推理结果的队列 using to save inference result(a token and it's decode text) 
        self._finished = False          

    def put(self, item: ResponseOutput) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        # 标记生成任务已经完成，并通过队列通知消费者生成结束。
        self._queue.put_nowait(StopIteration)
        self._finished = True
    
    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        # 实现异步迭代器协议中的 __aiter__ 方法，使得该类实例可以通过 async for 循环来迭代。
        return self

    async def __anext__(self) -> ResponseOutput:
        # 实现异步迭代器协议中的 __anext__ 方法，用于逐步获取生成任务的结果。
        result = await self._queue.get()# 从队列中异步获取下一个结果。
        if result is StopIteration:
            raise StopAsyncIteration #通知迭代结束，这是异步迭代器的结束信号。
        elif isinstance(result, Exception):
            raise result
        return result


"""
类的整体工作流程
生成任务初始化：当有一个新的生成任务开始时，会创建 AsyncResultsOfOneRequest 实例，并且 request_id 表示当前任务的唯一标识。

异步结果存储：生成模型在生成过程中，会不断地调用 put 方法，将部分结果（ResponseOutput）放入异步队列中。这些结果可以是模型在每一步生成的文本或者其他信息。

异步获取结果：消费者可以通过 async for 循环（异步迭代）来逐步获取生成任务的结果。__anext__ 方法负责从队列中取出下一个结果。

任务完成：当生成任务完成时，finish 方法会被调用，向队列中放入 StopIteration，并将 _finished 标志设为 True，通知消费者生成结束。

处理结束条件：在异步迭代器中，如果获取到 StopIteration，则抛出 StopAsyncIteration，结束异步循环。
"""