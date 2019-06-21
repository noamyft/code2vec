import numpy as np
import random

# def overrideVariables(newVarList, code):
#     var_code_split_index = code.find(" ")
#     return ",".join(newVarList) + code[var_code_split_index:]

def init_deadcode_variable(code, variables):
    return [("zpkjxq","zpkjxq")]

class AdversarialSearcher():

    def __init__(self, topk, max_depth, model, code, initial_state_generator=None):
        self.topk = topk
        self.max_depth = max_depth
        self.model = model


        # process data line - get vars
        var_code_split_index = code.find(" ")
        self.original_code = code[var_code_split_index + 1:]
        self.vars = code[:var_code_split_index]
        if self.vars != "":
            self.vars = self.vars.lower().split(",")

        # process states
        if self.can_be_adversarial():
            # get original name
            contexts = self.original_code.split(" ")
            self.original_name = contexts[0]

            # get forbidden words
            contexts = [c.split(",") for c in contexts[1:] if c != ""]
            self.forbidden_varnames = set()
            for tup in contexts:
                if tup[0] in model.word_to_index:
                    self.forbidden_varnames.add(model.word_to_index[tup[0]])
                if tup[2] in model.word_to_index:
                    self.forbidden_varnames.add(model.word_to_index[tup[2]])
            self.forbidden_varnames = list(self.forbidden_varnames)

            self.open_state_to_node = {}
            self.close_state_to_node = {}
            self.unchecked_nodes = []

            if initial_state_generator == None:
                initial_state_generator = self._get_init_state

            init_states = [(s, 0) for s in initial_state_generator(self.original_code, self.vars)]
            self._update_open(init_states, 0)

            current_state, self.current_node = self._select_best_state()
            del self.open_state_to_node[current_state]
            self.close_state_to_node[current_state] = self.current_node
            pass

    def can_be_adversarial(self):
        return self.vars != ""

    def get_adversarial_code(self, return_with_vars=False):
        assert self.current_node is not None
            # return None
        return self._apply_state(self.original_code, self.current_node["state"], return_with_vars=return_with_vars)

    def pop_unchecked_adversarial_code(self, return_with_vars=False):
        if not self.unchecked_nodes:
            self.unchecked_nodes = [self.current_node]

        res = [(node, self._apply_state(self.original_code, node["state"], return_with_vars=return_with_vars))
               for node in self.unchecked_nodes]
        del self.unchecked_nodes
        self.unchecked_nodes = []
        return res

    def get_original_name(self):
        return self.original_name

    def get_current_node(self):
        return self.current_node

    # def get_adversarial_name(self):
    #     return self.adversarial_code

    def _get_init_state(self, code, variables):
        return [(variables[0],variables[0])]

    def _apply_state(self, code, state, return_with_vars=False):
        original_var, new_var = state

        new_code = code.replace(" " + original_var + ",", " " + new_var + ",")\
            .replace("," + original_var + " ", "," + new_var + " ")

        if return_with_vars:
            return_vars = [v for v in self.vars if v != original_var] + [new_var]
            new_code = ",".join(return_vars) + " " + new_code

        return new_code

    def _create_bfs_node(self, state, level, score):
        return {"state":state, "level":level,"score":score}

    def is_target_found(self, predictions):
        # results = model_results
        # prediction_results = common.parse_results(results, None, topk=0)
        # original_name, predictions = results[0]

        return predictions[0] != self.original_name

    def next(self, model_grads):
        # create new states
        if self.current_node["level"] < self.max_depth:
            new_states = self._create_states(self.current_node["state"], model_grads, self.topk)

            self._update_open(new_states, self.current_node["level"] + 1)

        # find best renaming
        if not self.open_state_to_node:
            return False

        current_state, self.current_node = self._select_best_state()
        del self.open_state_to_node[current_state]
        self.close_state_to_node[current_state] = self.current_node

        return True

    def _update_open(self, new_states, new_level):

        new_valid_states = [(state, score) for state, score in new_states
                        if state not in self.close_state_to_node and
                            (state not in self.open_state_to_node or score > self.open_state_to_node[state]["score"])]

        new_nodes = {state: self._create_bfs_node(state, new_level, score) for state, score in new_valid_states}
        self.unchecked_nodes += new_nodes.values()
        self.open_state_to_node.update(new_nodes)


    def _create_states(self, state, model_results, topk):
        original_var, new_var = state

        loss, all_strings, all_grads = model_results
        indecies_of_var = np.argwhere(all_strings == new_var).flatten()
        grads_of_var = all_grads[indecies_of_var]
        if grads_of_var.shape[0] == 0:
            return []
            # print("current loss:",loss)
        total_grad = np.sum(grads_of_var, axis=0)
        # # words to increase loss
        top_replace_with = np.argsort(total_grad)[::-1]
        # filter forbidden words
        top_replace_with = top_replace_with[~np.isin(top_replace_with,self.forbidden_varnames)]
        # select top-k
        top_replace_with = top_replace_with[:topk]
            # result = [(i, total_grad[i], self.model.index_to_word[i]) for i in top_replace_with]
            # print("words to increase loss:")
            # print(result)
            # words to decrease loss
        # top_replace_with = np.argsort(total_grad)[:topk]
        # TODO: check if len total_grads == len index_to_word -1
        result = [((original_var, self.model.index_to_word[i]), total_grad[i]) for i in top_replace_with]

        return result

    def _select_best_state(self):
        return max(self.open_state_to_node.items(), key=lambda n: n[1]["score"])

class AdversarialTargetedSearcher(AdversarialSearcher):

    def __init__(self, topk, max_depth, model, code, new_target, initial_state_generator=None):
        self.new_target = new_target
        # replace original name with targeted name
        start_original_name = code.find(" ") + 1
        end_original_name = code.find(" ", start_original_name)
        true_target = code[start_original_name:end_original_name]
        code = code[:start_original_name] + self.new_target + code[end_original_name:]

        super().__init__(topk, max_depth, model, code, initial_state_generator=initial_state_generator)
        self.original_name = true_target


    def is_target_found(self, predictions):

        return predictions[0] == self.new_target

    def get_adversarial_name(self):
        return self.new_target

    def _create_states(self, state, model_results, topk):
        loss, all_strings, all_grads = model_results

        res = super()._create_states(state,(loss, all_strings, -all_grads),topk)

        return res


#######################################################################################
    # def rename_var(self, state, src, dst):
    #     return (src, dst)
    # def find_adversarial(self):
    #     # input_filename = 'Input.java'
    #     # MAX_ATTEMPTS = 50
    #     # MAX_NODES_TO_OPEN = 10
    #
    #     open = [self.create_bfs_node(self._get_init_state(self.code), 0, 0)]
    #     close =[]
    #
    #     # print('Starting interactive prediction with mono adversarial search...')
    #     while open:
    #         # open.sort(key=lambda n : -n["score"])
    #         current_node_index, current_node  = self._select_best_state(open)
    #         del open[current_node_index]
    #         close.append(current_node)
    #
    #         new_code = self._apply_state(self.code, current_node["state"])
    #
    #         # feed forward to evaluate
    #         results = self.model.predict(new_code)
    #
    #         if self.is_target_found(results):
    #             # print("MATCH FOUND!", current_node)
    #             # print("Tried (total:", len(close), ") :: ", close)
    #             return current_node
    #
    #         # feed backward to find adversarial
    #         model_results = self.model.calc_loss_and_gradients_wrt_input(new_code)
    #
    #         # find best renaming
    #         if current_node["level"] < self.max_depth:
    #             new_states = self._create_states(current_node["state"], model_results, self.topk)
    #             new_nodes = [self.create_bfs_node(state, current_node["level"] + 1, score)
    #                          for state, score in new_states if not self._state_exist(open, close, state)]
    #             open = open + new_nodes
    #
    #
    #     # print("FAILED!")
    #     # print("Tried (total:", len(close),") :: ", close)
    #     return None

class AdversarialSearcherTrivial(AdversarialSearcher):
    def __init__(self, topk, max_depth, model, code, initial_state_generator=None):
        super().__init__(topk,max_depth, model,code,initial_state_generator)
        self.random_candidates = []
        self.num_of_trials = int(topk**(max_depth + 1) / (topk - 1) - 2)

    def _create_states(self, state, model_results, topk):
        original_var, new_var = state

        all_name = np.array(list(self.model.index_to_word.keys()))
        # filter forbidden words
        replace_with = all_name[~np.isin(all_name, self.forbidden_varnames)]

        self.random_candidates = random.sample(replace_with.tolist(), self.num_of_trials)

        result = [((original_var, self.model.index_to_word[i]), 0) for i in self.random_candidates]

        return result

    def next(self, model_grads):
        if self.random_candidates:
            return False

        # create new states
        new_states = self._create_states(self.current_node["state"], model_grads, self.topk)

        self._update_open(new_states, self.current_node["level"] + 1)
        # clear open (because we already generate enough possibilities)
        self.open_state_to_node.clear()

        return True

class AdversarialTargetedSearcherTrivial(AdversarialTargetedSearcher):
    def __init__(self, topk, max_depth, model, code, new_target, initial_state_generator=None):
        super().__init__(topk, max_depth, model, code, new_target, initial_state_generator)
        self.random_candidates = []

    def _create_states(self, state, model_results, topk):
        original_var, new_var = state

        self.random_candidates = [self.new_target.replace("|","")]

        result = [((original_var, self.new_target.replace("|","")), 0)]

        return result

    def next(self, model_grads):
        if self.random_candidates:
            return False

        # create new states
        new_states = self._create_states(self.current_node["state"], model_grads, self.topk)

        self._update_open(new_states, self.current_node["level"] + 1)
        # clear open (because we already generate enough possibilities)
        self.open_state_to_node.clear()

        return True