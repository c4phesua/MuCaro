from src.environment.caro import Board

if __name__ == '__main__':
    board = Board(size=10)
    board.render_board(printable=True)
    while True:
        action = int(input(f'player {board.current_player + 1} play: '))
        result = board.play(action)
        board.render_board(printable=True)
        if result['done']:
            print('Game over')
            print(f'winner = player {result["winner"] + 1}')
            break

