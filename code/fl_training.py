import numpy as np

def GetUserDataFunc(news_title,train_user_id_sample,train_user,train_sess,train_label,train_user_id):
    def _get_user_data(uid):
        click = []
        sample = []
        label = []
        for sid in train_user_id_sample[uid]:
            click.append(train_user['click'][train_user_id[sid]])
            sample.append(train_sess[sid])
            label.append(train_label[sid])
        click = np.array(click)
        sample = np.array(sample)
        label = np.array(label)
        click = news_title[click]
        sample = news_title[sample]        
        return click,sample,label
    return _get_user_data


def add_noise(weights,lambd):
    for i in range(len(weights)):
        weights[i] += np.random.laplace(scale = lambd,size=weights[i].shape)
    return weights

def fed_single_update(model,doc_encoder,user_encoder,num,lambd,get_user_data,train_uid_table):
    random_index = np.random.permutation(len(train_uid_table))[:num]
    
    all_news_weights = []
    all_user_weights = []
    old_news_weight = doc_encoder.get_weights()
    old_user_weight = user_encoder.get_weights()
    
    sample_nums = []
    
    loss = []

    for uinx in random_index:
        doc_encoder.set_weights(old_news_weight)
        user_encoder.set_weights(old_user_weight)

        uid = train_uid_table[uinx]
        click,sample,label = get_user_data(uid)
        #print(label)
        g = model.fit([sample,click],label,batch_size = label.shape[0],verbose=False)
        loss.append(g.history['loss'][0])
        news_weight = doc_encoder.get_weights()
        user_weight = user_encoder.get_weights()
        if lambd>0:
            news_weight = add_noise(news_weight,lambd)
            user_weight = add_noise(user_weight,lambd)
        #noise = 
        #weight += noise
        all_news_weights.append(news_weight)
        all_user_weights.append(user_weight)
        sample_nums.append(label.shape[0])
    
    sample_nums = np.array(sample_nums)
    sample_nums = sample_nums/sample_nums.sum()
    
    doc_weights = [np.average(weights, axis=0,weights=sample_nums) for weights in zip(*all_news_weights)]
    user_weights = [np.average(weights, axis=0,weights=sample_nums) for weights in zip(*all_user_weights)]
    
    doc_encoder.set_weights(doc_weights)
    user_encoder.set_weights(user_weights)
    loss = np.array(loss).mean()
    #print('average loss',loss)
    return loss