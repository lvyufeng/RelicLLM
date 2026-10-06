"""Client stop sequences, matched on a stream of text deltas.

The runtime family's own streamer (:class:`relicllm.backends.base.TokenStreamer`) matches stop
strings inside its token callback, where it re-decodes the whole run every step and so always has
the full text in hand. The DeepSeek-V4 stream does not go through that path: its events carry the
*new* tokens of a step, and the text a client sees is the concatenation of those deltas. This is
the same match for that shape -- deltas in, the part that may go on the wire out -- so a stop
sequence ends the run on both routes rather than only the serial one.

Why a delta and not the full text: the events this reads are already deltas, and a filter that took
full text would have to be handed a cumulative string built from them -- but the event ids are the
*new* tokens per step, not the run's whole sequence, so a cumulative buffer built from them is not
the text the model produced. Deltas are the one thing this path actually carries.

A tail is held back exactly the way :func:`~relicllm.backends.base.hold_back` holds one: a suffix
that could still turn out to be the start of a marker waits for the delta that decides it, because
a stream cannot take a character back.
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import hold_back


class StopFilter:
    """One streamed answer's stop matching: what has been sent, and what may be sent next.

    Held in an object rather than a closure because the fields are written by one delta and read by
    the next. One instance serves one row; a request for several choices gets one per choice, which
    is why nothing here is module-level state.
    """

    def __init__(self, stops: Sequence[str] = ()) -> None:
        #: The client's sequences, in the order they arrived. Empty means the filter is a no-op that
        #: still answers ``tail``/``partial`` correctly, so a caller never has to branch on it.
        self.stops = tuple(stop for stop in stops if stop)
        #: Everything the model has produced so far, sent or held.
        self.text = ""
        #: The prefix of :attr:`text` that has gone out. `text[len(emitted):]` is what is held back.
        self.emitted = ""
        #: Whether a marker has been matched. Once set, every later delta is dropped.
        self.hit = False

    @property
    def partial(self) -> bool:
        """Whether text is being held back because it may yet be the start of a marker."""
        return len(self.emitted) < len(self.text)

    def feed(self, delta: str) -> str:
        """Append ``delta``, and return what of it may be sent now.

        Empty when the delta is entirely held back, entirely past a marker, or arrives after one was
        already found. A whole marker inside the accumulated text cuts there and is not sent; a
        suffix that is a partial marker is held; anything else is sent.
        """
        if not delta:
            return ""
        if self.hit:
            return ""
        self.text += delta
        cut = self._cut(self.text)
        if cut >= 0:
            self.hit = True
            return self._advance(self.text[:cut])
        return self._advance(hold_back(self.text, self.stops))

    def tail(self) -> str:
        """The held-back tail, once the answer is known to be over.

        Empty after a match -- the held text was the marker or what followed it, and neither is
        sent -- and otherwise whatever the holdback was keeping, which is text the model really
        wrote and a partial marker that never completed.
        """
        if self.hit:
            return ""
        return self._advance(self.text)

    def _advance(self, target: str) -> str:
        """Send what ``target`` adds to what has gone out, and remember that it has."""
        if len(target) <= len(self.emitted):
            return ""
        delta = target[len(self.emitted):]
        self.emitted = target
        return delta

    def _cut(self, text: str) -> int:
        return min((text.find(stop) for stop in self.stops if stop in text), default=-1)