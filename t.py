def random_mask(view_num,input_len,missing_rate):
    assert missing_rate is not None
    one_rate = 1 - missing_rate

    if one_rate <= (1 / view_num):
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0,view_num,size=(input_len,1))).toarray()
        return view_preserve

    if one_rate == 1:
        matrix = randint(1,2,size=(input_len,view_num))
        return matrix

    if input_len < 32:
        alldata_len = 32
    else:
        alldata_len = input_len
    error = 1
    while error >= 0.005:
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0,view_num,size=(alldata_len,1))).toarray()

        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0,100,size=(alldata_len,view_num)) < int(ratio * 100)).astype(np.int_)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int_))
        one_num_iter = one_num / (1- a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0,100,size=(alldata_len,view_num)) < int(ratio * 100)).astype(np.int_)
        matrix = ((matrix_iter + view_preserve) >0).astype(np.int_)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    matrix = matrix[:input_len,:]
    return matrix