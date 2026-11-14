from src.environments.caro import Board

if __name__ == '__main__':
    board = Board(size=3)
    for i in range(3):
        action = int(input())
        board.play(action)
        board.render_board(printable=True)
    print(board.state)

