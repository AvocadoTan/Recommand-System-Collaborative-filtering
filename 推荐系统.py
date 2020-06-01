###############################
#基于用户的协同过滤算法
import pandas as pd
import numpy as np
import scipy.sparse as sp
import time

def read_trainset():
    train_1 = pd.read_csv("/Users/tanjiale/大数据作业/HW2/Project2-data/netflix_train.txt", sep=" ")
    train_df = pd.DataFrame(train_1)
    train_df.columns = ['userID', 'movieID', 'rank', 'date']
    user_movie_train = train_df.pivot(index='userID', columns="movieID",
                                      values="rank")
    user_movie_train = user_movie_train.fillna(0)
    return user_movie_train


def read_user():
    user_1 = pd.read_csv("/Users/tanjiale/大数据作业/HW2/Project2-data/usernew.csv")
    user_1_df = pd.DataFrame(user_1)
    user_1_df = user_1_df.sort_values(by='userID', ascending=True)
    user_2_df = user_1_df.reset_index(drop=True)
    return user_2_df


def read_cosin():
    cosine = pd.DataFrame(pd.read_csv("/Users/tanjiale/大数据作业/HW2/Project2-data/cos.csv"))
    return cosine


def csc_trainmatrix_transform():
    train_csc = sp.csc_matrix(train_set_2)
    return (train_csc)


def cos_sim(train_set_2):
    sim = train_set_2 * train_set_2.T
    sim_dia = sim.diagonal()
    inv_sim_dia = 1 / sim_dia
    inv_sim_dia[np.isinf(inv_sim_dia)] = 0
    inv_mag = np.sqrt(inv_sim_dia)
    t = (sim.multiply(inv_mag).T.multiply(inv_mag))
    t = round(t, 3)
    t = t.todense()
    return t


def predict(user, movie):
    k = 11
    user_1 = user_list[(user_list.userID == user)].index.values
    user_1 = user_1[0]

    choice = cosine_train[user_1].argsort()[::-1][:k]
    choice_1 = np.delete(choice, 0)

    try:
        prediction = (train_set_2[choice_1, movie]).dot(cosine_train[choice_1, user_1]) \
                     / sum(cosine_train[user_1, choice_1])
        if prediction > 5: prediction = 5
        if prediction < 1: prediction = 1

    except IndexError:
        return 2.5
    else:
        print(prediction)
        return prediction


def rmse(pred, actual):
    from sklearn.metrics import mean_squared_error

    return np.sqrt(mean_squared_error(pred, actual))


print('loading trainset...')
start = time.clock()
train_set_1 = read_trainset()
train_set_2 = train_set_1.values
print('loaded trainset, computing cosine matrix...')
cosine_train = read_cosin().values
print('cosine matrix completed, loading userset...')
user_list = read_user()
print('loaded user list')
print('loading test list')
test_1 = pd.read_csv("/Users/tanjiale/大数据作业/HW2/Project2-data/netflix_test.txt", sep=" ")
test_1 = pd.DataFrame(test_1)
test_1.columns = ['userID', 'movieID', 'rank', 'date']
user_movie_test = test_1.pivot(index='userID', columns="movieID",
                               values="rank")
print('loaded test list')

predictions = []
targets = []
for row in test_1.itertuples():
    print(row)
    user, movie, actual = row[1], row[2], row[3]

    predictions.append(predict(user, movie))
    targets.append(actual)

end = (time.clock() - start)
print("Time used:", end)
print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))


#######################################
###基于梯度下降的矩阵分解算法

import pandas as pd
import numpy as np
from numpy import random,mat
import time


def read_trainset():
    train_1 = pd.read_csv("/Users/tanjiale/大数据作业/HW2/Project2-data/netflix_train.txt", sep=" ")
    train_df = pd.DataFrame(train_1)
    train_df.columns = ['userID', 'movieID', 'rank', 'date']
    user_movie_train = train_df.pivot(index='userID', columns="movieID",
                                      values="rank")
    user_movie_train = user_movie_train.fillna(0)
    return user_movie_train

def read_testset():
    test_1 = pd.read_csv("/Users/tanjiale/大数据作业/HW2/Project2-data/netflix_test.txt", sep=" ")
    test_df = pd.DataFrame(test_1)
    test_df.columns = ['userID', 'movieID', 'rank', 'date']
    user_movie_test = test_df.pivot(index='userID', columns="movieID",
                                      values="rank")
    user_movie_test_1 = user_movie_test.fillna(0)
    return user_movie_test_1

def read_user():
    user_1 = pd.read_csv("/Users/tanjiale/大数据作业/HW2/Project2-data/usernew.csv")
    user_1_df = pd.DataFrame(user_1)
    user_1_df = user_1_df.sort_values(by='userID',ascending=False)
    user_2_df = user_1_df.reset_index(drop=True)
    return user_2_df

def rmse(pred, actual):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(pred, actual))


def initial_p(p_k):
    p_array = np.random.randint(low=1,high=100,size=(10000,p_k))

    p_mat =  mat(p_array*(1/200))
    return p_mat

def initial_q(q_k):
    q_array = np.random.randint(low=1,high=100, size=(10000,q_k))
    q_mat = mat(q_array*(1/200))
    return q_mat

def indicating_train_matrix():
    train_set1 = train_set.replace([1,2,3,4,5],1)
    return train_set1

def indicating_test_matrix():
    test_set_3 = test_set_2.replace([1,2,3,4,5],1)
    return test_set_3

def all_zero_matrix():
    all_zero_matrix = np.zeros((10000, 10001))
    all_zero_matrix = pd.DataFrame(all_zero_matrix, index=user_set)
    all_zero_matrix = all_zero_matrix.drop([0], axis=1)
    return all_zero_matrix

def minimize_loss_function(p,q):
    alpha = 0.00001
    lambda_1 = 1
    iteration_count = 30
    function_result = []
    rmse_result = []
    tar_1 = np.array(test_set_2).flatten()
    tar_2 = list(filter(lambda x: x != 0, tar_1))
    for i in range(iteration_count):
        equation_t = p * q.T - train_set_1
        equation_1 = np.multiply(indi_train_mat, equation_t)
        print("1")
        equation_2 = -1 * equation_1
        print("2")
        derivative_U = equation_1 * q + 2 * lambda_1 * p
        print(derivative_U)
        derivative_V = equation_1.T * p + 2 * lambda_1 * q
        print(derivative_V)
        loss_function = 0.5 * ((np.linalg.norm(equation_2)) ** 2) + lambda_1 * (np.linalg.norm(p, ord=2) ** 2) + \
                        lambda_1 * (np.linalg.norm(q, ord=2) ** 2)
        print(loss_function)
        function_result.append(loss_function)
        predict_matrix = np.multiply((p * q.T),indi_test_mat)
        pre_1 = np.array(predict_matrix).flatten()
        pre_2 = list(filter(lambda x: x != 0, pre_1))
        result = rmse(pre_2,tar_2)
        print(result)
        rmse_result.append(result)
        p = p - alpha * derivative_U
        print(p)
        q = q - alpha * derivative_V
        print(q)
    print(function_result)
    print(rmse_result)

start = time.clock()
test_set_1 = read_testset()
train_set = read_trainset()
train_set_1 = train_set.values
user_set = np.array(read_user())
user_set = user_set.flatten()
user_set = user_set.tolist()
all_zero_matrix = all_zero_matrix()
test_set_2 = (test_set_1+all_zero_matrix).fillna(0)

p_k = 100
q_k = 100
p = initial_p(p_k)
q = initial_q(q_k)
indi_train_mat = indicating_train_matrix().values
indi_test_mat = indicating_test_matrix().values
minimize_loss_function(p,q)
end = (time.clock() - start)
print("Time used:",end)





