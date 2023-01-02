from typing import Dict, Iterator, Iterable, List

import ffmpeg
import numpy as np
from streamlink import Streamlink


class SimpleTwitchStreamReceiver(Iterable):
    """
    An iterable created given URL to a twitch stream and a quality preset.
    """

    def __init__(self, url: str, quality: str = "best"):
        self.quality: str = quality
        self.url: str = url

    @property
    def available_qualities(self) -> List[str]:
        session = Streamlink()
        streams = session.streams(self.url)
        return list(streams.keys())

    @property
    def _stream_url(self) -> str:
        session = Streamlink()
        streams = session.streams(self.url)
        stream = streams[self.quality]
        return stream.to_url()

    @property
    def probe(self) -> Dict:
        return ffmpeg.probe(self._stream_url, select_streams='v')

    @property
    def width(self) -> int:
        return self.probe['streams'][0]['width']

    @property
    def height(self) -> int:
        return self.probe['streams'][0]['height']

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
                    continue
                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                yield frame
        finally:
            process.stdout.close()
            process.wait()
