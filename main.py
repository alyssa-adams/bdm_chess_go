import pickle, os, re
import pandas as pd
import chess
import ast
from pybdm import BDM
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import statistics


# need to set up an initial board
# white goes first
# update the board based on the sequence of moves
# make a pair of states per 'round'
# (w, b) until a win
# then can measure bdm of each player separately, or then of the game
# for each state, count the number of possible moves allowed for the active player


# import the file, read in the data, and save to a nice structured dict
def translate_data(file, data_directory):

    # define the regexes
    match_re = '(\[Event[\s\S\w\W]+?)\[Event'
    white_re = '\[White ".+"\]'
    black_re = '\[Black ".+"\]'
    white_elo_re = '\[WhiteElo "\d+"\]'
    black_elo_re = '\[BlackElo "\d+"\]'
    winner_re = 'Result \"[1\-0]+'
    moves_re = '1\. .+'

    data = {}
    data['white_players'] = []
    data['black_players'] = []
    data['white_elos'] = []
    data['black_elos'] = []
    data['winners'] = []
    data['moves'] = []

    # read in the file as a string
    with open(os.path.join(data_directory, file), mode='r') as file:
        file_data = file.read()

    # get whole match, match is game is wanted first
    matches = re.findall(match_re, file_data)

    for match in matches:

        # if there isn't a real match, just skip
        try:
            white_player = re.findall(white_re, match)[0]
            white_player = white_player.split('"')[1]
            black_player = re.findall(black_re, match)[0]
            black_player = black_player.split('"')[1]
            white_elo = re.findall(white_elo_re, match)[0]
            white_elo = int(white_elo.split('"')[1])
            black_elo = re.findall(black_elo_re, match)[0]
            black_elo = int(black_elo.split('"')[1])
            winner = re.findall(winner_re, match)[0]
            winner = winner.split('"')[1]
            if winner == '1-0':
                winner = 'white'
            else:
                winner = 'black'
            moves = re.findall(moves_re, match)[0]

        except:
            continue

        # only keep games with at least ten turns total
        n_moves = len(re.findall('\.', moves))
        if n_moves < 10:
            continue

        data['white_players'].append(white_player)
        data['black_players'].append(black_player)
        data['white_elos'].append(white_elo)
        data['black_elos'].append(black_elo)
        data['winners'].append(winner)
        data['moves'].append(moves)

    return data


def make_chess_db():

    # list all the files in this directory
    data_directory = os.path.join('data', 'Human vs Human games', 'top 200 players_ Lichess.com', 'top 200 classical-- lichess')
    files = os.listdir(data_directory)
    files = list(filter(lambda x: re.search('\.pgn', x), files))

    df = pd.DataFrame()
    for file in files:
        print(file)
        data = translate_data(file, data_directory)
        df2 = pd.DataFrame.from_dict(data)
        df = df.append(df2, ignore_index=True)

    # pickle to save the resulting chess db
    with open('pickle_jar/matches_df', 'wb') as handle:
        pickle.dump(df, handle)


###### Read in the files, clean, translate, save all in single db
#make_chess_db()


###### Load in the files, use chess module to turn the board into a matrix

def make_states():

    # load in the db
    df = pickle.load(open('pickle_jar/matches_df', 'rb'))

    # need to save these to a file one at a time, because it takes too long to hold in memory

    for k, match in enumerate(df['moves']):

        print(k)

        bad_move = False

        # parse the moves into a list
        moves = re.split('\d+\.', match)[1:]
        moves = list(map(lambda x: x.strip(), moves))

        # get rid of the match result at the end of the last move
        moves = list(map(lambda x: re.sub(' [01]{1,}\-[01]{1,}', '', x), moves))
        moves = list(map(lambda x: x.split(), moves))

        # make into a single list
        all_moves = []
        for m in moves:
            [all_moves.append(x) for x in m]

        # move the board forward with each move, turn into a list of boards
        states = []
        n_possible_moves = []

        # set up the board
        board = chess.Board()

        for move in all_moves:

            # make the move
            try:
                board.push_san(move)

            # if there's an invalid move, just skip the entire match. Note which match this is and delete it from the df
            except:
                bad_move = True
                break

            state = str(board)
            state = re.sub('[A-Z]{1,}', '1', state)
            state = re.sub('[a-z]{1,}', '2', state)
            state = re.sub('\.', '0', state)
            state = state.split('\n')
            state = list(map(lambda x: x.split(), state))
            state = [[int(j) for j in i] for i in state]
            states.append(state)

            # count number of legal moves here
            n_moves = board.legal_moves.count()
            n_possible_moves.append(n_moves)

        # breaks to here. If it's actually a valid move, save states and things to list
        if not bad_move:
            with open("pickle_jar/all_states", "a+") as myfile:
                myfile.write(str(states) + '\n')
            with open("pickle_jar/all_n_moves", "a+") as myfile:
                myfile.write(str(n_possible_moves) + '\n')

        else:
            with open("pickle_jar/remove_matches", "a") as myfile:
                myfile.write(str(k) + '\n')


#make_states()


def refine_df():

    # get the bdm values

    # load in the db
    df = pickle.load(open('pickle_jar/matches_df', 'rb'))

    # remove the matches with the bad moves from the df
    remove_matches = []
    with open('pickle_jar/remove_matches') as fp:
        for line in fp:
            remove = int(re.sub('\n', '', line))
            remove_matches.append(remove)
    df = df.drop(remove_matches)

    # add one new column for n possible states
    all_n_moves = []
    with open('pickle_jar/all_n_moves') as fp:
        for line in fp:
            line = ast.literal_eval(line)
            all_n_moves.append(line)
    df['n_possible_moves'] = all_n_moves

    # drop moves column to save memory a little
    #df = df.drop(columns=['moves'])

    # pickle to save the resulting chess db
    with open('pickle_jar/matches_df_refined', 'wb') as handle:
        pickle.dump(df, handle)


#refine_df()


def calculate_bdm():

    # init bdm class
    bdm_one = BDM(ndim=2, nsymbols=2, warn_if_missing_ctm=False)


    with open('pickle_jar/all_states') as fp:

        for line in fp:

            print(line)

            bdms_white = []
            bdms_black = []

            # go between black and white each move (white goes first)
            for i, state in enumerate(ast.literal_eval(line)):

                # if even (starting at 0), means it is white turn (just after a move was made
                if (i % 2) == 0:
                    state_white = re.sub('2', '0', line)
                    state_white = ast.literal_eval(state_white)[i]
                    measure_white = bdm_one.bdm(np.array(state_white), normalized=True)
                    bdms_white.append(measure_white)

                # else it means black just made the move
                else:
                    state_black = re.sub('1', '0', line)
                    state_black = re.sub('2', '1', state_black)
                    state_black = ast.literal_eval(state_black)[i]
                    measure_black = bdm_one.bdm(np.array(state_black), normalized=True)
                    bdms_black.append(measure_black)


            # save to file as you chug along. Then this can be used for a plot later
            with open("pickle_jar/bdms_white_all", "a+") as myfile:
                myfile.write(str(bdms_white) + '\n')
            with open("pickle_jar/bdms_black_all", "a+") as myfile:
                myfile.write(str(bdms_black) + '\n')


#calculate_bdm()


def plot_bdm_results():

    # make a folder for the figures
    figure_folder = 'figures'
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)

    # load in the db
    df = pickle.load(open('pickle_jar/matches_df_refined', 'rb'))
    df = df.reset_index(drop=True)

    '''
    These are the DF columns:
    
    data['white_players'] = []
    data['black_players'] = []
    data['white_elos'] = []
    data['black_elos'] = []
    data['winners'] = []
    data['moves'] = []
    data['n_possible_moves'] = []
    '''

    bdm_pickles = ['bdms_white_all', 'bdms_black_all']

    # plot a line
    # elo vs bdm
    # n moves vs bdm

    for bw in bdm_pickles:

        # do this for the mean, media, and mode
        for m in ['mean', 'median', 'max']:

            # plot outfile name
            player_type = bw.split('_')[1]
            plot_out = 'moves' + '_' + 'bdm' + '_' + player_type + '_' + m + '.png'
            plot_out = os.path.join(figure_folder, plot_out)

            xs = []
            ys = []

            with open(bw) as fp:
                for i, line in enumerate(fp):

                    y = re.sub('\n', '', line)
                    y = ast.literal_eval(y)

                    x = list(df.iloc[[i]]['n_possible_moves'])[0]

                    if m == 'mean':
                        x = sum(x)/len(x)
                        y = sum(y)/len(y)

                    elif m == 'median':
                        x = statistics.median(x)
                        y = statistics.median(y)

                    elif m == 'max':
                        x = max(x)
                        y = max(y)

                    ys.append(y)
                    xs.append(x)

            plt.plot()
            sns.scatterplot(x=xs, y=ys, linewidth=0, s=1, alpha=0.7, legend=False)
            plt.xlabel(m + ' n possible moves')
            plt.ylabel(m + ' ' + player_type + ' bdm')
            plt.savefig(plot_out)
            plt.clf()


plot_bdm_results()
