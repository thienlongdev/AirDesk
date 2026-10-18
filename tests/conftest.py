import os
import tempfile

import pytest

# Never touch the real %APPDATA%\AirDesk from tests.
os.environ.setdefault("AIRDESK_HOME", tempfile.mkdtemp(prefix="airdesk-test-"))

from airdesk.config import Settings  # noqa: E402
from airdesk.core.engine import InteractionEngine  # noqa: E402
from airdesk.core.screen import ScreenLayout  # noqa: E402
from airdesk.core.types import Finger  # noqa: E402
from airdesk.sim.synthetic import Pose, frame, observation  # noqa: E402


class RecMouse:
    """MouseSink that records everything and tracks held buttons."""

    def __init__(self):
        self.events = []
        self.held = set()
        self.pos = (960.0, 540.0)

    def move(self, x, y):
        self.pos = (x, y)
        self.events.append(("move", round(x), round(y)))

    def button(self, button, down):
        if down:
            assert button not in self.held, "second button-down without release"
            self.held.add(button)
        else:
            self.held.discard(button)
        self.events.append(("button", button, down))

    def click(self, button, count=1):
        self.events.append(("click", button, count))

    def scroll(self, dx, dy):
        self.events.append(("scroll", dx, dy))

    def xbutton(self, which):
        self.events.append(("xbutton", which))

    def zoom(self, n):
        self.events.append(("zoom", n))

    def release_all(self):
        for b in list(self.held):
            self.events.append(("button", b, False))
        self.held.clear()

    def held_buttons(self):
        return tuple(sorted(self.held))

    def cursor_pos(self):
        return self.pos

    def of(self, kind):
        return [e for e in self.events if e[0] == kind]

    def clicks(self):
        """Logical clicks: ("click", b, n) events plus button down->up pairs
        with no cursor motion in between (a press that never dragged)."""
        out, down = [], {}
        for i, e in enumerate(self.events):
            if e[0] == "click":
                out.append((e[1], e[2], self._pos_before(i)))
            elif e[0] == "button" and e[2]:
                down[e[1]] = (i, self._pos_before(i))
            elif e[0] == "button" and e[1] in down:
                j, pos = down.pop(e[1])
                moved = any(m[0] == "move" and (m[1], m[2]) != pos for m in self.events[j:i])
                if not moved:
                    out.append((e[1], 1, pos))
        return out

    def drags(self):
        """Button down->up pairs with cursor motion in between."""
        out, down = [], {}
        for i, e in enumerate(self.events):
            if e[0] == "button" and e[2]:
                down[e[1]] = (i, self._pos_before(i))
            elif e[0] == "button" and e[1] in down:
                j, pos = down.pop(e[1])
                if any(m[0] == "move" and (m[1], m[2]) != pos for m in self.events[j:i]):
                    out.append((e[1], pos, self._pos_before(i)))
        return out

    def _pos_before(self, i):
        for e in reversed(self.events[:i]):
            if e[0] == "move":
                return (e[1], e[2])
        return (round(self.pos[0]), round(self.pos[1]))


class FakeKeyboard:
    def __init__(self, rect=(0, 700, 1920, 1080)):
        self.visible = False
        self.rect = rect
        self.pressed = []
        self.hovered = None

    def is_visible(self):
        return self.visible

    def hit_test(self, x, y):
        l, t, r, b = self.rect
        if self.visible and l <= x < r and t <= y < b:
            return "a"
        return None

    def hover(self, hand, key, progress=0.0):
        self.hovered = key

    def press_key(self, key, hand=0):
        self.pressed.append(key)
        return True

    def release_all(self):
        pass


class Sim:
    """Drives an InteractionEngine with synthetic hands at 30 fps."""

    def __init__(self, settings=None, armed=True, mode=None):
        self.s = settings or Settings()
        if mode is not None:
            # "classic" = the v1.1.1 contract (down at the pinch, up at the
            # release, hold = drag); the v1.1.2 default is "instant".
            self.s.gesture.click_mode = mode
        self.mouse = RecMouse()
        self.kb = FakeKeyboard()
        self.commands = []
        self.engine = InteractionEngine(self.s, ScreenLayout.single(1920, 1080), self.mouse, self.kb,
                                        on_command=self.commands.append)
        self.t = 0.0
        self.x, self.y = 0.5, 0.75
        if armed:
            self.engine.set_armed(True, self.t)

    def step(self, pose=None, n=1, dx=0.0, dy=0.0, hand=True, **kw):
        status = None
        for _ in range(n):
            self.t += 1 / 30
            self.x += dx
            self.y += dy
            hands = [observation(pose or Pose.pointing(Finger.INDEX), self.x, self.y, **kw)] if hand else []
            status = self.engine.process(frame(self.t, *hands))
        return status

    def settle(self, finger=Finger.INDEX, n=20):
        return self.step(Pose.pointing(finger), n)


@pytest.fixture
def sim():
    return Sim()
