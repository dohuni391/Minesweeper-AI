import time
import numpy as np
from termcolor import colored

class Minesweeper():
    def __init__(self, rows: int, columns: int, mines: int):
        self.rows, self.cols, self.mines = rows, columns, mines
        self.reset()
                        
    def reset(self):
        self.remains = self.rows * self.cols
        self.flagged, self.clicks, self.done = 0, 0, 0

        self.start_time, self.__duration = time.time(), 0

        self.map = np.zeros((self.rows,self.cols),dtype=int) # the value of the cell {-1(mine), 0, 1, 2, 3, 4, 5, 6, 7, 8}
        self._init_map()
        self.mask = np.full_like(self.map, -1, dtype=int) # the label of the cell {FLAGGED: -2, UNKNOWN: -1, KNOWN: 0}

        self.used3BV = np.zeros((self.rows,self.cols),dtype=int)
        self.is3BV = np.zeros((self.rows,self.cols),dtype=int)
        self.cleared3BV = 0
        self.value3BV = self._calculate3BV()

        self.efficiency = 0
        self.completion_percentage = 0
        
    def _init_map(self):
        mine_locs = np.random.choice(range(self.remains), self.mines, replace=False)
        for loc in mine_locs:
            row = int(loc/self.cols)
            col = int(loc%self.cols)
            self.map[row][col] = -1
            for ri in range(max(0, row - 1), min(self.rows, row + 2)):
                for ci in range(max(0, col - 1), min(self.cols, col + 2)):
                    if self.map[ri][ci] != -1:
                        self.map[ri][ci] += 1
        
    def _calculate3BV(self) -> int:
        value3BV = 0
        for r in range(self.rows):
            for c in range(self.cols):
                value = self.map[r][c]
                if self.used3BV[r][c] == 0 and value == 0:
                    value3BV+=1
                    self.used3BV[r][c] = 1
                    self.is3BV[r][c] = 1
                    toReveal = [(r,c)]
                    safety = 100000
                    for ri, ci in toReveal:
                        if self.map[ri][ci] == 0:
                            for rii in range(max(ri - 1, 0), min(self.rows, ri + 2)):
                                for cii in range(max(ci - 1, 0), min(self.cols, ci + 2)):
                                    if rii != ri or cii != ci:
                                        if self.used3BV[rii][cii] == 0:
                                            self.used3BV[rii][cii] = 1
                                            if self.map[rii][cii] == 0:
                                                toReveal.append((rii, cii))
                        safety-=1
                        if safety < 0:
                            break

        for r in range(self.rows):
            for c in range(self.cols):
                value = self.map[r][c]
                if value != -1 and self.used3BV[r][c] == 0:
                    value3BV+=1
                    self.is3BV[r][c] = 1
                    
        return value3BV
    
    def get_state(self):
        state = np.zeros((self.rows,self.cols),dtype=int)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.mask[r][c] == 0:
                    state[r][c] = self.map[r][c]
                else:
                    state[r][c] = self.mask[r][c]
        return state
                    
    # Operation [0: Uncover, 1: Flag]
    def step(self, row: int, col: int, operation = 0):
        assert self.done == 0
        assert 0 <= row < self.rows and 0 <= col < self.cols

        self.clicks += 1

        # Perform operation
        do = {0: self._uncover, 1: self._flag}[operation]
        reward = do(row, col)
        
        if self.done != 0:
            if self.done == 1:
                self.efficiency = 100 * self.value3BV / self.clicks
                self.completion_percentage = 100
                reward = 5
            elif self.done == -1:
                self.efficiency = 100 * self.cleared3BV / self.clicks
                self.completion_percentage = 100 * (((self.rows * self.cols) - self.mines) - (self.remains - self.mines))/((self.rows * self.cols) - self.mines)
                reward = -1
            self.__duration = time.time() - self.start_time
            
        return self.get_state(), reward, self.done
    
    def _uncover(self, row: int, col: int):
        reward = -0.5
        # Uncover cell
        if self.mask[row][col] == -1:  # Unknown cell
            reward = 0.5
            self.mask[row][col] = 0
            self.remains -= 1
            if self.is3BV[row][col] == 1: #
                self.cleared3BV+=1 #
                reward += 0.5 #
            if self.map[row][col] == -1:
                self.done = -1
            elif self.remains == self.mines:
                self.done = 1
            elif self.map[row][col] == 0:
                # Recursively uncover adjacent cells
                for ri in range(max(row - 1, 0), min(self.rows, row + 2)):
                    for ci in range(max(col - 1, 0), min(self.cols, col + 2)):
                        if ri != row or ci != col:
                            if self.mask[ri][ci] == -1:
                                self._uncover(ri, ci)
        return reward

    def _flag(self, row: int, col: int):
        # Flag cell
        if self.mask[row][col] == -1:
            self.mask[row][col] = -2
            self.flagged += 1
        elif self.mask[row][col] == -2:  # Unflag
            self.mask[row][col] = -1
            self.flagged -= 1
    
    @property
    def duration(self) -> float:
        if self.done == 0:
            return time.time() - self.start_time
        else:
            return self.__duration
        
    def show(self):
        signs = {-1: '█', -3: colored('F', 'red')}
        # Show grid line
        index_row = '    '
        for col in range(self.cols):
            index_row += str(col % 10) + ' '
        sep_row = '  ╔' + '═' * (len(index_row) - 3) + '╗'
        print(index_row + '\n' + sep_row, end='')
        print('\n', end='')
        for row in range(self.rows):
            print('{:2d}║ '.format(row), end='')
            for col in range(self.cols):
                state = self.mask[row][col]
                value = self.map[row][col]
                if state in signs:
                    c = signs[state]
                elif value == -1:
                    c = colored('*', 'black')
                elif value == 0:
                    c = ' '
                else:
                    c = str(value)
                print(c, end=' ')
            print('║\n', end='')
        sep_row = '  ╚' + '═' * (len(index_row) - 3) + '╝'
        print(sep_row)
        
    def info(self):
        if self.map is None:
            print(colored('RESTART GAME','red'))
            return
        if self.done == 0:
            print(' * * * * * * * * *')
        if self.done == 1:
            print(colored(' * * * WIN * * *', 'yellow'))
        elif self.done == -1:
            print(colored(' * * * LOST * * *', 'red'))

        if self.done != 0:
            print(colored(' Time: {:.2f}s\n'
                        ' Mines: {}\n'
                        ' Remain: {}\n'
                        ' Marked: {}\n'
                        ' Clicks: {}\n'
                        ' 3BV: {}/{}\n'
                        ' Efficiency: {:.2f}%\n'
                        ' Completion: {:.2f}%\n'.format(self.duration,
                                                self.mines,
                                                self.remains,
                                                self.flagged,
                                                self.clicks,
                                                self.cleared3BV,
                                                self.value3BV,
                                                self.efficiency,
                                                self.completion_percentage),
                        'yellow'))