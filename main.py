from src.environment.caro import Board

if __name__ == '__main__':
    board = Board(size=5)
    for i in range(9):
        action = int(input())
        print(board.play(action))
        board.render_board(printable=True)
    print(board.state)

