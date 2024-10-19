import asyncio
import functools
from asyncio import AbstractEventLoop, Future, IncompleteReadError
import logging
import random
import time
from typing import List, Callable, Optional
from .timestamped_unsigned_char_vector import TimestampedUnsignedCharVector


logger = logging.getLogger()


class TCPServer:
    def __init__(self,
                 io_service: AbstractEventLoop,
                 port: int,
                 callback: Callable[[TimestampedUnsignedCharVector], None],
                 log_name: str):
        self.io_service = io_service
        self.port = port
        self.onMessageReceived = callback
        self.confirm_with_fixed_reply = False
        self.fixed_reply = b''
        self.expect_size_header = True
        self.log_name = log_name
        self.server: Optional[Server] = None
        self.writer: List[asyncio.StreamWriter] = []
        self.closing = False

        assert(not asyncio.iscoroutinefunction(callback))

    def start(self):
        asyncio.run_coroutine_threadsafe(self.startAccept(), self.io_service).result()

    async def startAccept(self) -> None:
        port = self.port
        if port == 0:
            # attempt to assign a port from a predefined range
            port_min = 10000
            port_max = 11000 # TODO: could be configurable
            while True:
                port = random.randint(port_min, port_max)
                try:
                    logger.info('starting sever with port %i', port)
                    self.server = await asyncio.start_server(self.__cb, None, port)
                    logger.info('ok')
                    self.port = port
                    return
                except OSError as e:
                    logger.exception(e)
                    continue
        else:
            self.server = await asyncio.start_server(self.__cb, None, self.port)

    async def __cb(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.writer.append(writer)
        while not self.closing:
            # read header
            try:
                err = 'reading size in bytes'
                data = await reader.readexactly(4)
                expected = int.from_bytes(data, byteorder='big', signed=False)
                err = 'reading bytes'
                data = await reader.readexactly(expected)
                result = TimestampedUnsignedCharVector(data=data, timestamp=time.time())
            except exceptions.IncompleteReadError as e:
                if self.closing:
                    continue
                logger.debug("exception reading from stream in " + err
                        + " " + self.log_name, exc_info=e)
                await asyncio.sleep(1)
                continue

            if self.confirm_with_fixed_reply:
                writer.write(self.fixed_reply)
                await writer.drain()

            try:
                # run in threadpool, who knows how fast is our callback
                fut = self.io_service.run_in_executor(None, functools.partial(self.onMessageReceived, result))
                fut.add_done_callback(self.__done)
            except RuntimeError as e:
                # work around https://github.com/python/cpython/issues/99704
                logger.debug('error scheduling callback, closing the server', exc_info=e)
                self.close()
                break
        writer.close()

    def __done(self, fut: Future) -> None:
        e = fut.exception()
        if e is not None:
            logger.exception(f"Error running callback in {self.log_name}", exc_info=e)
            return
        if not fut.done():
            return
        result = fut.result()
        if result is not None:
            logger.info(f'done with result {result}')

    def expectSizeHeader(self, expect_size_header: bool):
        pass

    def confirmWithFixedReply(self, reply: str) -> None:
        assert(isinstance(reply, str))
        self.confirm_with_fixed_reply = True
        self.fixed_reply = reply.encode()

    def close(self):
        assert self.server is not None
        if not self.server.is_serving() or self.closing:
            return
        self.closing = True
        try:
            self.server.close()
        except TypeError as e:
            logger.error(f"error on stopping server {self}", exc_info=e)
        asyncio.run_coroutine_threadsafe(self.server.wait_closed(), self.io_service).result()
        for writer in self.writer:
            logger.debug(f'closing {writer}')
            self.io_service.call_soon_threadsafe(writer.close)
            asyncio.run_coroutine_threadsafe(writer.wait_closed(), self.io_service).result()
        logger.debug("server " + self.log_name + " closed")

    def getPort(self) -> int:
        return self.port

    def isRunning(self) -> bool:
        return self.server is not None and self.server.is_serving()
