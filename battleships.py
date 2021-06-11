#!/bin/env python3
"""

"""
import numpy as np

rng = np.random.default_rng()


class Game:
    def __init__(self, h=10, w=10):
        self.hw = np.asarray([h, w], np.uint32)
        self._grid = np.ones(shape=(h, w), dtype=np.int32) * -1
        self._ships = []

    @property
    def grid(self):
        return self._grid

    @property
    def occupancy(self):
        return self._grid != -1

    def _do_place(self, ship, position):
        ship_id = len(self._ships)

        self._grid[
            position[0] : position[0] + ship.h, position[1] : position[1] + ship.w
        ][ship.s > 0] = ship_id
        self._ships.append(ship)

    def place(self, ship, position):
        top = position[0]
        bottom = position[0] + ship.h + 2
        left = position[1]
        right = position[1] + ship.w + 2

        occ = np.pad(self.occupancy, 1)

        if np.any((occ[top:bottom, left:right])):
            raise ValueError("Invalid placement")

        self._do_place(ship, position)

    def place_random(self, ships):
        ships = np.asarray(ships)
        if len(ships) > 1:
            np.random.shuffle(ships)

        for s in ships:
            p = self.possible_places(s)
            if len(p) == 0:
                raise ValueError("No place to insert the ship")
            coords = np.stack(np.where(p), axis=-1)
            index = rng.integers(0, len(coords), dtype=np.int32, endpoint=False)
            self._do_place(s, coords[index])

    def possible_places(self, ship):
        occ = np.pad(self.occupancy, 1)
        empty_size = np.asarray(ship.s.shape) + 2
        windows = np.lib.stride_tricks.sliding_window_view(
            occ, empty_size, writeable=False
        )
        valid = (
            (~windows).reshape([windows.shape[0], windows.shape[1], -1]).all(axis=-1)
        )
        valid = np.pad(
            valid,
            [
                [0, self.grid.shape[0] - valid.shape[0]],
                [0, self.grid.shape[1] - valid.shape[1]],
            ],
            constant_values=False,
        )
        return valid


class Ship:
    def __init__(self, orientation: str = "horizontal"):
        if orientation not in ["horizontal", "vertical"]:
            raise ValueError("Invalid orientation")

        self.s = np.ones(shape=(1, 3), dtype=np.int32)

        if orientation == "vertical":
            self.s = self.s.transpose()

    @property
    def h(self):
        return self.s.shape[0]

    @property
    def w(self):
        return self.s.shape[1]


def print_console(game: Game):
    """

    :param game:
    :return:
    """
    row_separator = "-" * (game.hw[1] * (3 + 1) + 1)
    print(row_separator)
    for g_row in game.grid:
        s_row = "|".join(map(lambda x: "   " if x == -1 else " x ", g_row))
        s_row = f"|{s_row}|"
        print(s_row)
        print(row_separator)


def main():
    game = Game()

    ship1 = Ship(orientation="horizontal")
    ship2 = Ship(orientation="vertical")

    # game.place(ship1, position=(2, 6))
    # game.place(ship2, position=(4, 1))

    try:
        while True:
            game.place_random([ship1])
    except ValueError:
        pass

    try:
        while True:
            game.place_random([ship2])
    except ValueError:
        pass

    print_console(game)


if __name__ == "__main__":
    main()
