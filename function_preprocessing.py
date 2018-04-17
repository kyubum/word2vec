from gensim.models import Word2Vec
from gensim.models.word2vec import Word2Vec
import gensim.models.keyedvectors as word2vec
import pandas
from pandas import Series, DataFrame
import numpy
import math
from scipy.spatial import distance
import csv
from nltk.corpus import stopwords
import json
#############################################################################
# Basic_data_preprocessing function
#############################################################################
"""
*input*
 1차원 text 리스트    ex) ['it was nice','freindly staff!']

 *output*
 2차원 text 리스트    ex) [['it','nice'],['freindly','staff']]
"""
def basic_preprocessing(input_list):
    stop_words = set(stopwords.words('english'))
    text_list2 = [x.lower() for x in input_list]
    text_list3 = [x.replace(".","") for x in text_list2]
    text_list3 = [x.replace("!","") for x in text_list3]
    text_list3 = [x.replace("(","") for x in text_list3]
    text_list3 = [x.replace(")","") for x in text_list3]
    text_list3 = [x.replace("[","") for x in text_list3]
    text_list3 = [x.replace("]","") for x in text_list3]
    text_list3 = [x.replace("-","") for x in text_list3]
    text_list3 = [x.replace("\'","") for x in text_list3]
    text_list3 = [x.replace("\"","") for x in text_list3]
    text_list3 = [x.replace("@","") for x in text_list3]
    text_list3 = [x.replace("#","") for x in text_list3]
    text_list3 = [x.replace("^","") for x in text_list3]
    text_list3 = [x.replace(",","") for x in text_list3]
    text_list3 = [x.replace("?","") for x in text_list3]
    text_list3 = [x.replace("\n"," ") for x in text_list3]
    text_list3 = [x.replace(":"," ") for x in text_list3]
    text_list3 = [x.replace(";"," ") for x in text_list3]
    text_list4 = []
    for i in text_list3:
        text_list4.append(i.split(' '))

    text_list5 = []
    for i in text_list4:
        filtered_sentence = [w for w in i if not w in stop_words]
        text_list5.append(filtered_sentence)
    return(text_list5)


#############################################################################
# N_gram_case_dictionary function
#############################################################################
"""
*input*
 불용어 등 제거된 text의 2차원 리스트
*output*
 {case:count} dictionary
                ex) {burger <--> king : 100,
                     big <--> mac : 90}
"""
def n_gram_case_dic(input_list):
    tem_key_list = []
    for i in range(0, len(input_list)):
        for j in range(1, len(input_list[i])):
            tem_key = str(input_list[i][j-1]) + ' ' + str(input_list[i][j])
            tem_key_list.append(tem_key)

    out_dic = {}        
    for i in tem_key_list:
        if i in out_dic:
            out_dic[i] += 1
        else:
            out_dic[i] = 1

    # criteria _ min count
    criteria = 100
    final_out_dic1 = {key:val for key, val in out_dic.items() if val >= criteria and (key[0] != ' ' and key[-1] != ' ')}
    
    change_list = []
    for key,value in final_out_dic1.items():
        change_list.append(key)

    final_out_dic2 = {}
    for i in change_list:
        value = i.split(' ')
        key = i
        final_out_dic2[key] = value[0] + '_' + value[1]

    #case_dic save
    output_json = json.dumps(final_out_dic2)
    f = open('./n_gram_case_dic.json', 'w')
    f.write(output_json)
    f.close()
    return(final_out_dic2)


#############################################################################
# N_gram_case를 기준으로 Data preprocessing
#############################################################################
"""
*input / output*
 1차원 string list, n_gram_case_dic -> 1차원 string list
                
                ex) 'big <--> mac' 케이스가 충분히 많다면 'big mac' -> 'big_mac'
"""
def replace_all(text, dic):
    count = 1
    for i, j in dic.items():
        if count % 100 == 0:
            print(round((count / len(dic))*100, 3), '%', flush = True)
        text = text.replace(i, j)
        count += 1
    return(text)    


#############################################################################
# Create Word_Count Table(json)
#############################################################################
"""
 [API1 : 유사어에 대한 빈도 찾기] 
   -각 word에 대한 빈도 테이블
                           *input : compound 처리 완료된 리뷰 text / output : json
"""
def create_word_count_table(input_list):
    wordcount = {}
    for i in input_list:
        for word in i:
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1

    output_json = json.dumps(wordcount)
    f = open('./word_count_table.json', 'w')
    f.write(output_json)
    f.close()


#############################################################################
# Create Weight Matrix (-> best seller)
#############################################################################
def create_weight_matrix (any_input):
    #Load Word2Vec model
    model = Word2Vec.load('./embedding_model_SkipGram')

    #create distance matrix
    food_row = model.most_similar(positive=['food'],topn=10000**10000)
    service_row = model.most_similar(positive=['service'],topn=10000**10000)
    ambience_row = model.most_similar(positive=['ambience'],topn=10000**10000)
    value_row = model.most_similar(positive=['value'],topn=10000**10000)

    col_name_list = []
    col_value_list = []
    for i in food_row:
        col_name_list.append(i[0])
        col_value_list.append(distance.euclidean(model['food'],model[i[0]]))
    food_df = pandas.DataFrame(columns = col_name_list)
    food_df.loc[0] = col_value_list

    col_name_list = []
    col_value_list = []
    for i in service_row:
        col_name_list.append(i[0])
        col_value_list.append(distance.euclidean(model['service'],model[i[0]]))
    service_df = pandas.DataFrame(columns = col_name_list)
    service_df.loc[0] = col_value_list

    col_name_list = []
    col_value_list = []
    for i in ambience_row:
        col_name_list.append(i[0])
        col_value_list.append(distance.euclidean(model['ambience'],model[i[0]]))
    ambience_df = pandas.DataFrame(columns = col_name_list)
    ambience_df.loc[0] = col_value_list

    col_name_list = []
    col_value_list = []
    for i in value_row:
        col_name_list.append(i[0])
        col_value_list.append(distance.euclidean(model['value'],model[i[0]]))
    value_df = pandas.DataFrame(columns = col_name_list)
    value_df.loc[0] = col_value_list

    distance_matrix = pandas.concat([food_df,service_df,ambience_df,value_df])
    distance_matrix.index = ['food','service','ambience','value']
    distance_matrix = distance_matrix.fillna(0)

    #Create Weight Matrix
    dis_matrix = distance_matrix
    dis_values = dis_matrix.values
    dis_values2 = dis_matrix.values
    dis_values2 = list(dis_values2)

    w_list = []
    for i in dis_values:
        w_list2 = []
        for j in i:
            w_list2.append(math.exp(-(j**2)/2))
        w_list.append(w_list2)

    dis_values2[0] = w_list[0]
    dis_values2[1] = w_list[1]
    dis_values2[2] = w_list[2]
    dis_values2[3] = w_list[3]

    final_col_list = list(distance_matrix.columns.values)
    weight_final_df = pandas.DataFrame(columns = final_col_list)

    weight_final_df.loc[0] = w_list[0]
    weight_final_df.loc[1] = w_list[1]
    weight_final_df.loc[2] = w_list[2]
    weight_final_df.loc[3] = w_list[3]
    weight_final_df.index = ['food','service','ambience','value']

    #food, service, ambience, value 각 가중치의 편차가 0.06 이하인 단어들은 가중치 * 0.1
    def std(string):
        x1 = weight_final_df[string][0]
        x2 = weight_final_df[string][1]
        x3 = weight_final_df[string][2]
        x4 = weight_final_df[string][3]
        return(numpy.std([x1, x2, x3, x4]))

    col_list = list(weight_final_df)

    for i in col_list:
        if std(str(i)) <= 0.06:
            weight_final_df[i][0] = weight_final_df[i][0] * 0.1
            weight_final_df[i][1] = weight_final_df[i][1] * 0.1
            weight_final_df[i][2] = weight_final_df[i][2] * 0.1
            weight_final_df[i][3] = weight_final_df[i][3] * 0.1

    weight_final_df.to_csv('./weight_matrix.csv', encoding = 'utf-8', index=True)