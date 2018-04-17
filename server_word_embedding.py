from gensim.models import Word2Vec
from gensim.models.word2vec import Word2Vec
import gensim.models.keyedvectors as word2vec
from function_preprocessing import basic_preprocessing
from function_preprocessing import n_gram_case_dic
from function_preprocessing import replace_all
from function_preprocessing import create_word_count_table
from function_preprocessing import create_weight_matrix
import json
import sys
sys.stdout.flush()


#test data load
print('Data load', flush = True)
"""
with open('/root/yelp_review/dataset/review.json') as f:
    reviews = f.readlines()
reviews = [json.loads(review, encoding = 'utf-8') for review in reviews]
text_list = [review['text'] for review in reviews]
"""
f=open('/Users/kyubum/PycharmProjects/word2vec_enhance/n_gram/train_reviews_final2.txt', 'r')
text = f.read()
f.close()
text_list = text.split(':flag:')



#preprocessing
print('Data basic preprocessing', flush = True)
text_list2 = basic_preprocessing(text_list)

print('Create & Save n_gram_case_dic', flush = True)
n_gram_dic = n_gram_case_dic(text_list2)
print('Count n_gram_case_dic : ', len(n_gram_dic), flush = True)

print('Text list -> Text string', flush = True)
text_str = "<END>".join(text_list)

print('Compound_word_preprocessing', flush = True)
text_str2 = replace_all(text_str, n_gram_dic)

print('Text string -> Text list Again', flush = True)
text_list3 = text_str2.split("<END>")
print('Data basic preprocessing', flush = True)
text_list4 = basic_preprocessing(text_list3)


#create word_count_table
print('Create & Save word_count_table', flush = True)
create_word_count_table(text_list4)


#word_embedding
print('SKIP_GRAM_word embedding', flush = True)
model_skip_gram = Word2Vec(text_list4, size = 100, window = 5, min_count = 50, iter = 50, sg = 1)
model_skip_gram.init_sims(replace=True)
model_skip_gram.save('./embedding_model_SkipGram')

print('CBOW_word embedding', flush = True)
model_cbow = Word2Vec(text_list4, size = 100, window = 5, min_count = 50, iter = 50, sg = 0)
model_cbow.init_sims(replace=True)
model_cbow.save('./embedding_model_Cbow')

#create weight matrix
create_weight_matrix('any_input')