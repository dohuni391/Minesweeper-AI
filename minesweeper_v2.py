from collections import deque
import numpy as np
from termcolor import colored
import torch

REVEALED = 0
COVERED = -1
FLAGGED = -2
MINE = -99

class Minesweeper():

    def __init__(self, rows: int, columns: int, mines: int):

        #Game Data
        self.num_of_rows, self.num_of_cols = rows, columns
        self.num_of_mines, self.num_of_flags, self.num_of_remains = mines, 0, self.num_of_rows * self.num_of_cols
        self.status = 0 # Win: 1, In game: 0, Loss: -1

        #Training Data
        self.num_of_clicks = 0
        self.num_of_mine_remains = mines

        self.map = self._init_map()
        self.mask = np.full_like(self.map, -1, dtype=int) # FLAGGED: -2, COVERED: -1, KNOWN: 0

        self.used3BV = np.zeros((self.num_of_rows,self.num_of_cols),dtype=bool)
        self.is3BV = np.zeros((self.num_of_rows,self.num_of_cols),dtype=bool)
        self.cleared3BV = 0
        self._3BV = self._get_3BV()
        
    def _get_adjacent_cell(self, row: int, col: int):
        for r in range(max(row - 1, 0), min(self.num_of_rows, row + 2)):
            for c in range(max(col - 1, 0), min(self.num_of_cols, col + 2)):
                if r != row or c != col:
                    yield(r, c)

    def _get_3BV(self) -> int:
        value3BV = 0
        for r in range(self.num_of_rows):
            for c in range(self.num_of_cols):
                if self.used3BV[r][c] or self.map[r][c] != 0:
                    continue

                value3BV += 1
                self.used3BV[r][c] = True
                self.is3BV[r][c] = True

                toReveal = deque([(r,c)])
                safety = 100000

                while toReveal and safety > 0:
                    ri, ci = toReveal.popleft()

                    if self.map[ri][ci] != 0:
                        continue

                    for rii, cii in self._get_adjacent_cell(ri, ci):
                        if not self.used3BV[rii][cii]:
                            self.used3BV[rii][cii] = True
                            if self.map[rii][cii] == 0:
                                toReveal.append((rii, cii))

                    safety-=1

        for r in range(self.num_of_rows):
            for c in range(self.num_of_cols):
                if self.map[r][c] != -1 and not self.used3BV[r][c]:
                    value3BV += 1
                    self.is3BV[r][c] = True
                    
        return value3BV

    def _init_map(self):
        map = np.zeros((self.num_of_rows,self.num_of_cols),dtype=int) #[MINE(-99), 0, 1, 2, 3, 4, 5, 6, 7, 8]
        mine_locs = np.random.choice(range(self.num_of_remains), self.num_of_mines, replace=False)
        for loc in mine_locs:
            row = int(loc/self.num_of_cols)
            col = int(loc%self.num_of_cols)
            map[row][col] = MINE
            for r, c in self._get_adjacent_cell(row, col):
                if map[r][c] != MINE:
                    map[r][c] += 1
        return map
    
    def get_state(self):
        state = np.zeros((self.num_of_rows,self.num_of_cols),dtype=int)
        for r in range(self.num_of_rows):
            for c in range(self.num_of_cols):
                if self.mask[r][c] == REVEALED:
                    state[r][c] = self.map[r][c]
                else:
                    state[r][c] = self.mask[r][c]
        return state
    
    def _reveal(self, row: int, col: int):
        if self.mask[row][col] == COVERED:
            self.mask[row][col] = REVEALED
            self.num_of_remains -= 1
            if self.map[row][col] == MINE:
                self.status = -1
            elif self.num_of_remains == self.num_of_mines:
                self.status = 1
            elif self.map[row][col] == 0:
                for r, c in self._get_adjacent_cell(row, col):
                    if self.mask[r][c] == COVERED:
                        self._reveal(r, c)
    
    def _flag(self, row: int, col: int):
        if self.mask[row][col] == COVERED:
            self.mask[row][col] = FLAGGED
            self.num_of_flags += 1
        elif self.mask[row][col] == FLAGGED:
            self.mask[row][col] = COVERED
            self.num_of_flags -= 1

    def _chord(self, row: int, col: int):
        if self.mask[row][col] == REVEALED:
            num_of_adjacent_flags = 0
            for r, c in self._get_adjacent_cell(row, col):
                if (self.mask[r][c] == FLAGGED):
                    num_of_adjacent_flags += 1
        
        if self.map[row][col] == num_of_adjacent_flags:
            for r, c in self._get_adjacent_cell(row, col):
                self._reveal(r, c)

    def step(self, row: int, col: int, operation = 0):
        assert self.status == 0
        assert 0 <= row < self.num_of_rows and 0 <= col < self.num_of_cols

        cells_revealed_before = np.sum(self.mask == REVEALED)

        self.num_of_clicks += 1

        # Perform operation
        do = {0: self._reveal, 1: self._flag, 2: self._chord}[operation]
        do(row, col)

        cells_revealed_after = np.sum(self.mask == REVEALED)

        # Basic reward: difference in number of revealed cells
        reward = cells_revealed_after - cells_revealed_before

        # Penalize if the number of clicks exceeds the 3BV value
        if self.num_of_clicks > self._3BV:
            penalty = (self.num_of_clicks - self._3BV) * 0.05  # Penalty for exceeding optimal clicks
            reward -= penalty

         # Check game status
        if self.status == 1:
            reward += 10
        elif self.status == -1:
            reward -= 10

        return self.get_state(), reward, self.status
       
    def show(self):
        signs = {COVERED: '◼', FLAGGED: colored('⚐', 'red'), MINE: colored('✷', 'black'), 0: '☐'}

        print('  ', end='')
        for col in range(self.num_of_cols):
            print('{:2d}'.format(col), end='')
        print('')

        for row in range(self.num_of_rows):
            print('{:2d} '.format(row), end='')
            for col in range(self.num_of_cols):
                state = self.get_state()[row][col]
                if state in signs:
                    c = signs[state]
                else:
                    c = str(state)
                print(c, end=' ')
            print('\n', end='')

    def info(self):
        if self.status != 0:
            status = 'WON' if (self.status==1) else 'LOST'
            print('Game Result: {}\n'.format(status))