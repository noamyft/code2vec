from common import common
from realextractor import RealExtractor
import numpy as np
import re
from interactive_predict import InteractivePredictor
from gensim.models import KeyedVectors as word2vec

class AdversarialSearcher():

    def __init__(self, topk, max_depth, model):
        self.topk = topk
        self.max_depth = max_depth
        self.model = model

    def get_init_state(self, code):
        # TODO: use .tolower when get vars
        return ("i","i")

    def apply_state(self,code,state):
        original_var, new_var = state

        new_code = code[0].replace(" " + original_var + ",", " " + new_var + ",")\
            .replace("," + original_var + " ", "," + new_var + " ")

        return [new_code]

    # def rename_var(self, state, src, dst):
    #     return (src, dst)

    def create_bfs_node(self, state, level, score):
        return {"state":state, "level":level,"score":score}

    def is_target_found(self, model_results):
        results, loss, all_strings, all_grads = model_results
        # prediction_results = common.parse_results(results, None, topk=0)
        original_name, predictions, _, _ = results[0]

        return predictions[0] != original_name

    def create_states(self, state, model_results, topk):
        original_var, new_var = state

        results, loss, all_strings, all_grads = model_results
        indecies_of_var = np.argwhere(all_strings == new_var).flatten()
        grads_of_var = all_grads[indecies_of_var]
        assert grads_of_var.shape[0] > 0
            # print("current loss:",loss)
        total_grad = np.sum(grads_of_var, axis=0)
        # # words to increase loss
        top_replace_with = np.argsort(total_grad)[::-1][:topk]
            # result = [(i, total_grad[i], self.model.index_to_word[i]) for i in top_replace_with]
            # print("words to increase loss:")
            # print(result)
            # words to decrease loss
        # top_replace_with = np.argsort(total_grad)[:topk]
        # TODO: check if len total_grads == len index_to_word -1
        result = [((original_var, self.model.index_to_word[i]), total_grad[i]) for i in top_replace_with]

        return result

    def state_exist(self, open, close, state):
        a = close + open
        return any([True for node in a if node["state"] == state])


    def find_adversarial(self, code):
        # input_filename = 'Input.java'
        # MAX_ATTEMPTS = 50
        # MAX_NODES_TO_OPEN = 10

        open = [self.create_bfs_node(self.get_init_state(code), 0, 0)]
        close =[]

        # print('Starting interactive prediction with mono adversarial search...')
        while open:
            # open.sort(key=lambda n : -n["score"])
            current_node_index, current_node  = max(enumerate(open), key=lambda n: n[1]["score"])
            del open[current_node_index]
            close.append(current_node)

            new_code = self.apply_state(code, current_node["state"])

            model_results = self.model.calc_loss_and_gradients_wrt_input(new_code)

            if self.is_target_found(model_results):
                # print("MATCH FOUND!", current_node)
                # print("Tried (total:", len(close), ") :: ", close)
                return current_node

            # find best renaming
            if current_node["level"] < self.max_depth:
                new_states = self.create_states(current_node["state"], model_results, self.topk)
                new_nodes = [self.create_bfs_node(state, current_node["level"] + 1, score)
                             for state, score in new_states if not self.state_exist(open, close, state)]
                open = open + new_nodes


        # print("FAILED!")
        # print("Tried (total:", len(close),") :: ", close)
        return None

            # print(
            #     'Modify the file: "%s" and press any key when ready, or "q" / "quit" / "exit" to exit' % input_filename)
            # user_input = input()
            # if user_input.lower() in self.exit_keywords:
            #     print('Exiting...')
            #     return

            # print("select variable to rename:")
            # var_to_rename = newname_of_var = input()
            #
            # name_found = False
            # with open(input_filename, "r") as f:
            #     original_code = f.read()
            #
            # for i in range(MAX_ATTEMPTS):
            #     try:
            #         predict_lines, hash_to_string_dict = self.path_extractor.extract_paths(input_filename)
            #     except ValueError as e:
            #         print(e)
            #         continue
            #     results = self.model.predict(predict_lines)
            #     prediction_results = common.parse_results(results, hash_to_string_dict, topk=SHOW_TOP_CONTEXTS)
            #     for method_prediction in prediction_results:
            #         print('Original name:\t' + method_prediction.original_name)
            #         for name_prob_pair in method_prediction.predictions:
            #             print('\t(%f) predicted: %s' % (name_prob_pair['probability'], name_prob_pair['name']))
            #
            #     if '|'.join(method_prediction.predictions[0]['name']) == method_prediction.original_name:
            #         print("MATCH FOUND!", newname_of_var)
            #         print("Tried (total:", len(closed), ") :: ", closed)
            #         name_found = True
            #         break

                    # print('Attention:')
                    # for attention_obj in method_prediction.attention_paths:
                    #     print('%f\tcontext: %s,%s,%s' % (
                    #     attention_obj['score'], attention_obj['token1'], attention_obj['path'], attention_obj['token2']))

                # loss, all_strings, all_grads = self.model.calc_loss_and_gradients_wrt_input(predict_lines)
                # indecies_of_var = np.argwhere(all_strings == newname_of_var.lower()).flatten()
                # grads_of_var = all_grads[indecies_of_var]
                # if grads_of_var.shape[0] > 0:
                #     # print("current loss:",loss)
                #     total_grad = np.sum(grads_of_var, axis=0)
                #     # # words to increase loss
                #     # top_replace_with = np.argsort(total_grad)[::-1][:5]
                #     # result = [(i, total_grad[i], self.model.index_to_word[i]) for i in top_replace_with]
                #     # print("words to increase loss:")
                #     # print(result)
                #     # words to decrease loss
                #     top_replace_with = np.argsort(total_grad)[:5]
                #     result = [(i, total_grad[i], self.model.index_to_word[i]) for i in top_replace_with]
                #
                #     # select new name
                #     for r in result:
                #         if r[2] not in closed and r[2] != method_prediction.original_name.replace("|",""):
                #             print(r)
                #             newname_of_var = r[2]
                #             break
                #     else:
                #         newname_of_var = None
                #     if newname_of_var is None:
                #         break
                #     closed.append(newname_of_var)
                #
                #     print("rename", var_to_rename, "to", newname_of_var)
                #
                #     code = InteractivePredictor.rename_variable(original_code,var_to_rename,newname_of_var)
                #     with open("input.java", "w") as f:
                #         f.write(code)
