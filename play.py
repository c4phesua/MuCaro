from src.environment.caro import Board

if __name__ == '__main__':
    board = Board(size=10)
    board.render_board(printable=True)
    for i in range(9):
        print(f'player {board.current_player + 1} play: ')
        action = int(input())
        result = board.play(action)
        board.render_board(printable=True)
        if result['done']:
            print('Game over')
            print(f'winner = player {result["winner"] + 1}')
    print(board.state)

