"""bg.spawn：fire-and-forget 的任务跑完之前要一直有人拿着（事件循环只持弱引用）。"""
import asyncio

from src.plugins.chatbot import bg


async def test_spawn_holds_reference_until_done():
    gate = asyncio.Event()

    async def job():
        await gate.wait()
        return 42

    task = bg.spawn(job())
    assert task in bg._tasks
    gate.set()
    assert await task == 42
    await asyncio.sleep(0)
    assert task not in bg._tasks
