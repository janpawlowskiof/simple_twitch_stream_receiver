# Simple Twitch stream receiver

Tiny project that allows one to create a generator of `np.ndarray` frames given a URL to a Twitch stream.

Example usage:
```python
import cv2

from simple_twitch_stream_receiver import SimpleTwitchStreamReceiver

receiver = SimpleTwitchStreamReceiver("https://www.twitch.tv/darkviperau", quality="best")

for frame in receiver:
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

Will raise `StreamNotAvailable` if streamer is offline or does not exist.
