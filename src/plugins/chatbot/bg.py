"""后台任务登记：fire-and-forget 的任务要留一个强引用，直到它跑完。

为什么（2026-10-19）：事件循环对任务**只持弱引用**——`asyncio.create_task(...)` 不存返回值，
任务在 await 间隙可能被 GC 回收、半路消失（Python 文档 `asyncio.create_task` 一节的 Important 提示）。
心跳循环、B站/微博监听、记忆提取都是这么起的。调度方式和原来完全一样，只是多存一份引用。
"""
import asyncio

_tasks: set = set()


def spawn(coro) -> asyncio.Task:
    """等价于 `asyncio.create_task(coro)`，但任务跑完之前一直有人拿着它。"""
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
