import traceback

from adversarialsearcher import AdversarialSearcher
from common import common
from realextractor import RealExtractor
import numpy as np
import re
from interactive_predict import InteractivePredictor
from gensim.models import KeyedVectors as word2vec

import timeit
SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'
FULL_DICT_PATH_AND_NAME = 'data/java-large/java-large'


class InteractivePredictorAdvMonoSearch(InteractivePredictor):
    exit_keywords = ['exit', 'quit', 'q']

    def __init__(self, config, model):
        super().__init__(config, model)

    def predict(self):
        input_filename = 'Input.java'
        MAX_ATTEMPTS = 50
        MAX_NODES_TO_OPEN = 10

        print('Starting interactive prediction with mono adversarial search...')
        while True:
            print(
                'Modify the file: "%s" and press any key when ready, or "q" / "quit" / "exit" to exit' % input_filename)
            user_input = input()
            if user_input.lower() in self.exit_keywords:
                print('Exiting...')
                return

            print("select variable to rename:")
            var_to_rename = newname_of_var = input()

            name_found = False

            opened = []
            closed = []
            with open(input_filename, "r") as f:
                original_code = f.read()

            for i in range(MAX_ATTEMPTS):
                try:
                    predict_lines, hash_to_string_dict = self.path_extractor.extract_paths(input_filename)
                except ValueError as e:
                    print(e)
                    continue

                bfs = AdversarialSearcher(2, 2, self.model)
                r = bfs.find_adversarial(predict_lines)
                print(r)
                print(timeit.timeit(lambda: bfs.find_adversarial(predict_lines), number=1000))


                return
                results = self.model.predict(predict_lines)
                prediction_results = common.parse_results(results, hash_to_string_dict, topk=SHOW_TOP_CONTEXTS)
                for method_prediction in prediction_results:
                    print('Original name:\t' + method_prediction.original_name)
                    for name_prob_pair in method_prediction.predictions:
                        print('\t(%f) predicted: %s' % (name_prob_pair['probability'], name_prob_pair['name']))

                if '|'.join(method_prediction.predictions[0]['name']) == method_prediction.original_name:
                    print("MATCH FOUND!", newname_of_var)
                    print("Tried (total:", len(closed), ") :: ", closed)
                    name_found = True
                    break

                    # print('Attention:')
                    # for attention_obj in method_prediction.attention_paths:
                    #     print('%f\tcontext: %s,%s,%s' % (
                    #     attention_obj['score'], attention_obj['token1'], attention_obj['path'], attention_obj['token2']))

                loss, all_strings, all_grads = self.model.calc_loss_and_gradients_wrt_input(predict_lines)
                indecies_of_var = np.argwhere(all_strings == newname_of_var.lower()).flatten()
                grads_of_var = all_grads[indecies_of_var]
                if grads_of_var.shape[0] > 0:
                    # print("current loss:",loss)
                    total_grad = np.sum(grads_of_var, axis=0)
                    # # words to increase loss
                    # top_replace_with = np.argsort(total_grad)[::-1][:5]
                    # result = [(i, total_grad[i], self.model.index_to_word[i]) for i in top_replace_with]
                    # print("words to increase loss:")
                    # print(result)
                    # words to decrease loss
                    top_replace_with = np.argsort(total_grad)[:5]
                    result = [(i, total_grad[i], self.model.index_to_word[i]) for i in top_replace_with]

                    # select new name
                    for r in result:
                        if r[2] not in closed and r[2] != method_prediction.original_name.replace("|",""):
                            print(r)
                            newname_of_var = r[2]
                            break
                    else:
                        newname_of_var = None
                    if newname_of_var is None:
                        break
                    closed.append(newname_of_var)

                    print("rename", var_to_rename, "to", newname_of_var)

                    code = InteractivePredictor.rename_variable(original_code,var_to_rename,newname_of_var)
                    with open("input.java", "w") as f:
                        f.write(code)


            if not name_found:
                print("FAILED!")
                print("Tried (total:", len(closed),") :: ", closed)

class InteractivePredictorAdvSimilarSearch(InteractivePredictor):
    exit_keywords = ['exit', 'quit', 'q']

    def __init__(self, config, model):
        super().__init__(config, model)
        vectors_text_path_javalarge = 'models/java-large/tokens.txt'
        # self.tokens_model = word2vec.load_word2vec_format(vectors_text_path_javalarge, binary=False)
        self.tokens_model = word2vec.load("models/java-large/model.model")
        # temp_token_similarity = {w : [sw for sw,_ in tokens_model.most_similar([w], topn=50)]
        #                          for w in tokens_model.wv.vocab}
        # self.token_similarity = {w: [sw for sw in temp_token_similarity[w] if w in temp_token_similarity[sw]]
        #                               for w in temp_token_similarity}
        self.token_similarity = {}
        self.token_similarity_already_processed = []
        pass

    def get_similar_words(self,word):
        if word in self.token_similarity:
            return self.token_similarity[word]

        self.token_similarity[word] = [w for w, _ in self.tokens_model.most_similar([word], topn=100)
                                               if self.tokens_model.rank(w, word) < 50]
        return self.token_similarity[word]


    def measureSimilarity(self,words_vocab, word, mode):
        if mode == "cosine":
            vocab_norm = words_vocab / np.linalg.norm(words_vocab, axis=1).reshape((-1, 1))
            return 1 - np.dot(vocab_norm,vocab_norm[self.model.word_to_index[word]])
        elif mode == "euclidean":
            return np.linalg.norm(words_vocab - words_vocab[self.model.word_to_index[word]],axis=1)

    def predict(self):
        input_filename = 'Input.java'
        MAX_ATTEMPTS = 50
        MAX_NODES_TO_OPEN = 10

        words_vocab = self.model.get_words_vocab_embed()
        # words_vocab = words_vocab / np.linalg.norm(words_vocab, axis=1).reshape((-1, 1))

        print('Starting interactive prediction with similar adversarial search...')
        while True:
            print(
                'Modify the file: "%s" and press any key when ready, or "q" / "quit" / "exit" to exit' % input_filename)
            user_input = input()
            if user_input.lower() in self.exit_keywords:
                print('Exiting...')
                return

            print("select variable to rename:")
            var_to_rename = newname_of_var = input()

            name_found = False
            closed = [var_to_rename]
            with open(input_filename, "r") as f:
                original_code = f.read()

            for i in range(MAX_ATTEMPTS):
                try:
                    predict_lines, hash_to_string_dict = self.path_extractor.extract_paths(input_filename)
                except ValueError as e:
                    print(e)
                    continue
                results = self.model.predict(predict_lines)
                prediction_results = common.parse_results(results, hash_to_string_dict, topk=SHOW_TOP_CONTEXTS)
                for method_prediction in prediction_results:
                    print('Original name:\t' + method_prediction.original_name)
                    for name_prob_pair in method_prediction.predictions:
                        print('\t(%f) predicted: %s' % (name_prob_pair['probability'], name_prob_pair['name']))

                if '|'.join(method_prediction.predictions[0]['name']) != method_prediction.original_name:
                    print("MATCH FOUND!", newname_of_var)
                    print("Tried (total:", len(closed), ") :: ", closed)
                    name_found = True
                    break

                    # print('Attention:')
                    # for attention_obj in method_prediction.attention_paths:
                    #     print('%f\tcontext: %s,%s,%s' % (
                    #     attention_obj['score'], attention_obj['token1'], attention_obj['path'], attention_obj['token2']))

                loss, all_strings, all_grads = self.model.calc_loss_and_gradients_wrt_input(predict_lines)
                indecies_of_var = np.argwhere(all_strings == newname_of_var.lower()).flatten()
                grads_of_var = all_grads[indecies_of_var]
                if grads_of_var.shape[0] > 0:
                    # print("current loss:",loss)
                    total_grad = np.sum(grads_of_var, axis=0)
                    # # words to increase loss
                    # top_replace_with = np.argsort(total_grad)[::-1][:5]
                    # result = [(i, total_grad[i], self.model.index_to_word[i]) for i in top_replace_with]
                    # print("words to increase loss:")
                    # print(result)
                    # words to decrease loss
                    # top_replace_with = np.argsort(total_grad)[:5]

                    similarity_to_var = self.get_similar_words(newname_of_var) #self.measureSimilarity(words_vocab,newname_of_var, "euclidean")
                    result = [(self.model.word_to_index[i], i, total_grad[self.model.word_to_index[i]]) for i in similarity_to_var]

                    result.sort(key=lambda v: (-v[2]))

                    print(result)
                    # similarity_to_var = self.measureSimilarity(words_vocab,newname_of_var, "cosine")
                    # resulte = [(i, self.model.index_to_word[i], similarity_to_var[i], total_grad[i]) for i in range(1, words_vocab.shape[0])]
                    #
                    # resulte.sort(key=lambda v: (v[2],-v[3]))

                    # select new name
                    for r in result:
                        if r[1] not in closed and r[1] != method_prediction.original_name.replace("|","")\
                                and r[2] > 0:
                            print(r)
                            newname_of_var = r[1]
                            break
                    else:
                        newname_of_var = None
                    if newname_of_var is None:
                        break
                    closed.append(newname_of_var)

                    print("rename", var_to_rename, "to", newname_of_var)

                    code = InteractivePredictor.rename_variable(original_code,var_to_rename,newname_of_var)
                    with open("input.java", "w") as f:
                        f.write(code)


            if not name_found:
                print("FAILED!")
                print("Tried (total:", len(closed),") :: ", closed)