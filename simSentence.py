# -*- coding:UTF-8 -*-
from gensim import corpora, similarities, models
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import jieba
import logging
jieba.setLogLevel(logging.INFO)

class SentensSim(object):
    def __init__(self, documents, sentence,top_num=2):
        '''
        :param documents: 训练文本库
        :param sentence: 待匹配的句子
        :param top_num:返回相似结果的数量
        '''

        self.documents = documents
        self.sentence = sentence
        ###
        self.top_num = top_num

    def similarity(self):
        corpora_documents = []
        stopwords = {}.fromkeys([line.rstrip() for line in open('chineseStopWords.txt')])
        # 文本处理
        for item_text in self.documents:
            item_seg = list(jieba.cut(item_text)) #分词
            words = []
            for seg in item_seg:
                if seg not in stopwords:
                    words.append(seg) #去停词
            corpora_documents.append(words)
        # #生成字典和向量语料
        dictionary = corpora.Dictionary(corpora_documents)
        # 通过下面一句得到语料中每一篇文档对应的稀疏向量（这里是bow向量）
        corpus = [dictionary.doc2bow(text) for text in corpora_documents]
        #corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
        tiidf_model = models.TfidfModel(corpus)
        corpus_tfidf = tiidf_model[corpus]

        self.sim = similarities.Similarity('Similarity-tfidf-index', corpus_tfidf, num_features=600)
        self.sim.num_best = self.top_num #如果等于3，则返回最相似的3个结果
        sentence_cut_temp = list(jieba.cut(self.sentence))
        sentence_cut = []
        for word in sentence_cut_temp:
            if word not in stopwords:
                sentence_cut.append(word)
        sentence_cut_corpus = dictionary.doc2bow(sentence_cut)
        self.sentence_sim = tiidf_model[sentence_cut_corpus]
        self.resultShow()

    def resultShow(self):
        string = 'Answer:'
        for tpl in range(len(self.sim[self.sentence_sim])):
            if tpl != len(self.sim[self.sentence_sim]) - 1:
                string = string + str(self.sim[self.sentence_sim][tpl][0]) + '(' + str(
                    self.sim[self.sentence_sim][tpl][1]) + '),'
            else:
                string = string + str(self.sim[self.sentence_sim][tpl][0]) + '(' + str(self.sim[self.sentence_sim][tpl][1]) + ')'
        print(string)

# string = 'The Most Similar material is:'
if __name__ == '__main__':
    raw_documents = ['用火可以烧烤食物、照明、御寒、驱赶野兽。火的使用，提高了人类适应自然环境的能力，促进了体质的发展和脑的进化。',
                     '火可以烧烤食物，照明，取暖，保护自己。提高了人类适应自然环境的能力，促进了体质的发展和脑的进化。',
                     '用火可以制作熟食，照明，抵御寒冷，驱赶野兽，保护人类。提高了人类适应自然环境的能力，促进了体质的发展和脑的进化。',
                     '烧制食物，发光，照亮环境，烧火取暖。提高了人们对环境的抵抗力，更加适应环境。促进了人类历史的进步。',
                     '做食物，促进了体质发展。取暖，驱寒，提高了人适应环境的能力。长期来看，促进了人类的发展和大脑的进化。',
                     '用火可以烧烤食物，照明。提高了人们应对寒冷环境的能力，推动了人类历史的进步和发展。',
                     '制作更有营养的食物，提高了身体素质，对身体更健康。烧火取暖，防止生病。驱赶野兽，探险山洞，提高了生产力',
                     '用火可以做饭，烹饪。可以烧火取暖，可以烧火照明，可以烧火驱赶野兽。人类有更多的生活选择，促进了人类历史的发展与进步',
                     '使用火可以烤肉，煮饭，还可以制作火把，夜间照明，防止野兽入侵。使人类可以更好的适应环境，是人类历史的一大进步',
                     '火的使用大大提高了人类对于环境的适应性，人类可以烧火取暖应对寒冷天气。烤肉做饭，从而不再吃生的食物，提高身体素质。',
                     '用火可以做熟食，消灭细菌，更加卫生，有利于人类大脑的发展。用火可以取暖，应对寒冷季节，提高存活率，适应环境。'
                     '使用火是人类历史的重大进步，人类可以用火烧制食物，照明，照亮环境，生活取暖',
                     '用火烧烤食物，吃饭，保护自己。对身体有利，身体更健康',
                     '用火提高食物质量，有利身体健康，生活取暖，保护身体，更好的适应环境',
                     '用火很好，增加了人类对恶劣环境的适应性。人类可以用火做饭，取暖，烧水，使得个人更加健康。寿命更长。',
                     '火的作用：（1）熟食（烧烤食物），缩短消化过程，增强人的体质；（2）驱赶野兽，增强人类自卫和狩猎能力；（3）照明，从而扩大生活领域；（4）防寒，火的使用，增强了人们适应自然的能力，是人类进化过程中的一大进步。',
                     '火的使用对原始人类的生存和进化的作用：火的使用提高了原始人类适应自然环境的能力,促进了体质的发展和脑的进化.',
                     '吃熟食易于吸收，使猿人获得了更加丰富的营养，熟食、开水使猿人少生疾病;刀耕火种，火促进了农业的发展，增加了产量；铜的使用、铁的使用、陶器的发明都离不开火,火是原始人黑夜里驱赶虫蛇野兽的最有利工具'
                     ]
    ans1 = '用火可以烧烤食物、照明、御寒、驱赶野兽。火的使用，提高了人类适应自然环境的能力，促进了体质的发展和脑的进化。'
    ans2 = '可以加速，能跑步，做饭，可以制作食物，提高温度'
    ans3 = '垃圾，没用'

    test1 = SentensSim(raw_documents, ans1).similarity()
    test2 = SentensSim(raw_documents, ans2).similarity()
    test3 = SentensSim(raw_documents, ans3).similarity()

