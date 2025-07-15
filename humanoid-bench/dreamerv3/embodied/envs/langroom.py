from collections import deque

import embodied
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = ("red", "green", "blue", "yellow")
OBJECTS = ("ball", "rug", "table", "vase")
VOCAB = ("", *COLORS, *OBJECTS, "what", "color", "is", "the", "?", "it")
TEXTURES = list((embodied.Path(__file__).parent / "langroom_assets").glob("*.png"))

LAYOUT = """
#########
#0     1#
#       #
#       #
#       #
#       #
#       #
#2     3#
#########
""".strip("\n")


class LangRoom(embodied.Env):
    def __init__(self, task="talk", view=2, length=200, resolution=64, vocab_size=15, seed=None):
        assert task in ("talk", "acc"), task
        assert length > 0, length
        assert vocab_size >= len(VOCAB)
        self.task = task
        self.view = view
        self.length = length
        self.resolution = resolution
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)
        self.layout = np.array([list(line) for line in LAYOUT.split("\n")]).T
        self.textures = self._load_textures(view)
        self.player = None
        self.colors = None
        self.words = None
        self.duration = None
        self.done = True
        self.history_env = deque(maxlen=5)
        self.history_reward = deque(maxlen=5)
        self.history_agent = deque(maxlen=5)

    @property
    def obs_space(self):
        res = self.resolution
        return {
            "image": embodied.Space(np.uint8, (res, res, 3)),
            # 'text': embodied.Space(np.uint32, (), 0, len(VOCAB)),
            "text": embodied.Space(np.uint32, (), 0, self.vocab_size),
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
            "log_image": embodied.Space(np.uint8, (res, 4 * res, 3)),
            # 'log_text': embodied.Space(str),
        }

    @property
    def act_space(self):
        return {
            "move": embodied.Space(np.int32, (), 0, 5),
            # 'talk': embodied.Space(np.int32, (), 0, len(VOCAB)),
            "talk": embodied.Space(np.int32, (), 0, self.vocab_size),
            "reset": embodied.Space(bool),
        }

    def step(self, action):
        if action["reset"] or self.done:
            self.duration = 0
            self.done = False
            self.player = tuple(x // 2 for x in self.layout.shape)
            self.colors = self._new_colors()
            self.words = self._new_words()
            self.target = self.words.pop(0)
            self.history_reward.clear()
            self.history_env.clear()
            self.history_agent.clear()
            self.history_reward.append("")
            self.history_env.append(self.target)
            return self._obs(0.0, is_first=True)

        move = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)][action["move"]]
        player = (self.player[0] + move[0], self.player[1] + move[1])
        if self.layout[player[0], player[1]] == " ":
            self.player = player

        self.target = self.words.pop(0)
        if not self.words:
            self.colors = self._new_colors()
            self.words = self._new_words()

        # talk = VOCAB[action['talk']]
        index = action["talk"]
        talk = VOCAB[index] if index < len(VOCAB) else f"<{index}>"

        # if self.task == 'talk':
        #   if talk == '':
        #     reward = 0
        #   elif self.target not in COLORS:
        #     reward = -0.01
        #   elif talk == self.target:
        #     reward = 1
        #   else:
        #     reward = -0.1

        # if self.task == 'talk':
        #   if self.target == '':
        #     if talk == self.target:
        #       reward = 0
        #     else:
        #       reward = -0.001
        #   else:
        #     if talk == self.target:
        #       reward = 10
        #     else:
        #       reward = -0.001

        if self.task == "talk":
            if self.target == "":
                if talk == self.target:
                    reward = +0.001
                else:
                    reward = -0.001
            else:
                if talk == self.target:
                    reward = 10
                else:
                    reward = -0.001

        elif self.task == "acc":
            if talk == self.target:
                reward = 1
            else:
                reward = 0

        else:
            raise NotImplementedError(self.task)

        self.duration += 1
        if self.duration >= self.length:
            self.done = True

        self.history_agent.append(talk)
        self.history_reward.append(str(reward or ""))
        self.history_env.append(self.target)

        return self._obs(reward, is_last=self.done)

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        image = self.render()

        lines = list(self.history_env)
        for prev, line in zip(lines[:-1], lines[1:]):
            if prev == "the":
                assert line in OBJECTS, (prev, line)

        log_image = np.concatenate(
            [
                image,
                self._display(["ENV:"] + list(self.history_env)),
                self._display(["AGENT:"] + list(self.history_agent)),
                self._display(["REWARD:"] + list(self.history_reward)),
            ],
            0,
        )
        return dict(
            image=image.transpose((1, 0, 2)),
            text=int(VOCAB.index(self.target)),
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            # log_text=str(self.target),
            log_image=log_image.transpose((1, 0, 2)),
        )

    def _new_words(self):
        obj = str(self.rng.choice(OBJECTS))
        color = str(self.colors[obj])
        assert obj in OBJECTS, obj
        assert color in COLORS, color
        return [
            *([""] * self.rng.integers(5, 10)),
            "what",
            "color",
            "is",
            "the",
            obj,
            "?",
            *([""] * 10),
            "it",
            "is",
            color,
            *([""] * self.rng.integers(5, 10)),
        ]

    def _new_colors(self):
        # return {k: self.rng.choice(COLORS) for k in OBJECTS}
        return {k: v for k, v in zip(OBJECTS, self.rng.permutation(COLORS))}

    def render(self):
        view = 2 * self.view + 1
        grid = int(np.floor(self.resolution / view))
        canvas = np.ones((view * grid, view * grid, 3), np.float32)
        for dx in range(-self.view, self.view + 1):
            for dy in range(-self.view, self.view + 1):
                x = self.player[0] + dx
                y = self.player[1] + dy
                if 0 <= x < self.layout.shape[0] and 0 <= y < self.layout.shape[1]:
                    tile = self.layout[x, y]
                else:
                    tile = "#"
                if (x, y) == self.player:
                    tex = self.textures["player"]
                elif tile in [str(x) for x in range(len(OBJECTS))]:
                    tex = self.textures[OBJECTS[int(tile)]]
                    col = np.array(
                        {
                            "red": (1, 0.3, 0.1),
                            "green": (0.1, 0.7, 0.2),
                            "blue": (0.1, 0.3, 1),
                            "yellow": (1.0, 0.9, 0),
                        }[self.colors[OBJECTS[int(tile)]]]
                    )[None, None]
                    tex = tex.copy()
                    tex[..., :3] *= col
                else:
                    tex = {
                        "#": self.textures["wall"],
                        " ": self.textures["floor"],
                    }[tile]
                bg = self.textures["floor"]
                alpha = tex[..., -1:]
                tex = alpha * tex[..., :3] + (1 - alpha) * bg[..., :3]
                i = dx + self.view
                j = dy + self.view
                canvas[i * grid : (i + 1) * grid, j * grid : (j + 1) * grid] = tex
        pad = (self.resolution - canvas.shape[0]) // 2
        canvas = np.pad(canvas, ((pad, pad), (pad, pad), (0, 0)))
        return (255 * canvas).astype(np.uint8)

    def _display(self, lines):
        font = ImageFont.load_default()
        image = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        y = 0
        for line in lines:
            # _, _, w, h = font.getbbox(line)
            w, h = font.getsize(line)
            draw = ImageDraw.Draw(image)
            draw.text((0, y), line, (255, 255, 255), font=font)
            y += h
        return np.array(image).transpose((1, 0, 2))

    def _load_textures(self, view):
        view = 2 * self.view + 1
        grid = int(np.floor(self.resolution / view))
        textures = {}
        for filename in TEXTURES:
            with filename.open("rb") as f:
                image = np.array(Image.open(f))
            image = np.array(Image.fromarray(image).resize((grid, grid)))
            image = image.transpose((1, 0, 2))
            image = image.astype(np.float32) / 255
            if image.shape[-1] == 3:
                image = np.concatenate([image, np.ones_like(image[..., :1])], -1)
            textures[filename.stem] = image
        return textures


if __name__ == "__main__":
    import imageio

    env = LangRoom()
    img = env._display(["hello", "world"])
    imageio.imsave("test.png", img)
    import sys

    sys.exit()

    for i in range(30):
        act = {k: v.sample() for k, v in env.act_space.items()}
        act["reset"] = False
        obs = env.step(act)
        imageio.imsave(f"test{i}.png", obs["log_image"])
