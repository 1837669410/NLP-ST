# hmm: predict action by weather
import numpy as np

# sunny:0 rain:1
weather = [0,0,1,0,1,0,0,0,1,0,0,1,1,0,0,0,0,0,1]
w2v = {0:"sunny", 1:"rain"}
# football:0 tennis:1 stay-home:2
action =  [0,1,2,1,2,0,0,1,1,0,1,2,2,0,1,1,1,0,1]
a2v = {0:"football", 1:"tennis", 2:"stay-home"}

# get init mat
def get_init(a):
    # [3,1]
    _a, _a_count = np.unique(a, return_counts=True)
    _init_mat = np.zeros(shape=[len(_a),1])
    for i in range(len(_a)):
        _init_mat[i][0] = _a_count[i]
    _init_mat = _init_mat / np.sum(_a_count)
    return _init_mat
# get transfer mat
def get_transfer(a):
    # [3,3]
    _a = np.unique(a)
    _transfer_mat = np.zeros(shape=[len(_a), len(_a)])
    for i in range(len(a)-1):
        _transfer_mat[a[i]][a[i+1]] += 1
    _transfer_mat = _transfer_mat / np.sum(_transfer_mat, axis=1, keepdims=True)
    return _transfer_mat
# get launch mat
def get_launch(a, w):
    # [2,3]
    _a = np.unique(a)
    _w = np.unique(w)
    _launch_mat = np.zeros(shape=[len(_a), len(_w)])
    for i, j in zip(a, w):
        _launch_mat[i][j] += 1
    _launch_mat = _launch_mat / np.sum(_launch_mat, axis=1, keepdims=True)
    return _launch_mat
# greedy predict
def greedy_predict(q, i_mat, t_mat, l_mat):
    # q:query, i_mat:init_mat, t_mat:transfer_mat, l_mat:launch_mat
    result = []
    init_state = i_mat   # [3,1]
    for i in range(len(q)):
        init_state = l_mat * init_state   # [3,2] * [3,1] -> [3,2]([action,weather])
        max_prob_action = np.argsort(init_state[:,q[i]])[::-1][0]   # max prob action
        result.append(max_prob_action)
        init_state = init_state[max_prob_action,q[i]] * t_mat[max_prob_action,:].reshape(3,1)   # max prob value * [3,1] -> [3,1]
    return result
# print result
def Print(q_weather, q_result):
    for w, a in zip(q_weather, q_result):
        print("q_w:{} -> a:{}".format(w2v[w],a2v[a]))

# train
init_mat = get_init(action)   # get init mat [3,1]
transfer_mat = get_transfer(action)   # get transfer mat [3,3]
launch_mat = get_launch(action, weather)   # get launch mat [3,2]

q_weather = [0,1,0,0,0,1]   # q_weather
# greedy predict
q_result = greedy_predict(q_weather, init_mat, transfer_mat, launch_mat)
Print(q_weather, q_result)
