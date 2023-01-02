import time
from typing import Dict, Iterator, Iterable, List

import ffmpeg
import numpy as np
import streamlink.plugins.twitch
from streamlink import Streamlink

from simple_twitch_stream_receiver.exceptions import StreamNotAvailable


class SimpleTwitchStreamReceiver(Iterable):
    """
    An iterable created given a URL to a twitch stream and a quality preset.
    Will raise `StreamNotAvailable` if streamer is offline or does not exist.
    """

    def __init__(self, url: str, quality: str = "best", timeout: float = 5.0):
        self.quality: str = quality
        self.url: str = url
        self.timeout = timeout
        self._session = Streamlink()

    def __iter__(self) -> Iterator[np.ndarray]:
        width, height = self.width, self.height
        process = (
            ffmpeg
            .input(self._stream_url)
            .video
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run_async(pipe_stdout=True)
        )

        try:
            while True:
                in_bytes = process.stdout.read(width * height * 3)
                if not in_bytes:
                    time.sleep(self.timeout)
                    continue
                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                yield frame
        finally:
            process.stdout.close()
            process.wait()

    @property
    def _stream_url(self) -> str:
        streams = self._session.streams(self.url)
        self._assert_stream_available(streams)
        stream = streams[self.quality]
        return stream.to_url()

    def _assert_stream_available(self, streams: Dict[str, streamlink.plugins.twitch.TwitchHLSStream]) -> None:
        is_stream_available = len(streams) > 0
        if not is_stream_available:
            raise StreamNotAvailable(self.url)

    @property
    def _probe(self) -> Dict:
        return ffmpeg.probe(self._stream_url, select_streams='v')

    @property
    def width(self) -> int:
        return self._probe['streams'][0]['width']

    @property
    def height(self) -> int:
        return self._probe['streams'][0]['height']
