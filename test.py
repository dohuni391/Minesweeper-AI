from minesweeper_v2 import Minesweeper

env = Minesweeper(9, 9, 9)

while env.status == 0:
    env.show()
    row, col, operation = map(int, input("Input (row, column, operation[reveal: 0, flag: 1, chord: 2])\n").split())
    env.step(row, col, operation)

env.show()
env.info()