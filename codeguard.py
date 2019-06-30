import json
import requests
import common_adversarial

ONLY_VARIABLE = "VUNK"

def guard_by_n2p(code_sample_with_vars, is_word_in_vocab_func):
    variables, code = common_adversarial.separate_vars_code(code_sample_with_vars)
    if variables == "":
        return code
    else:
        variables = variables.lower().split(",")

    existed_variables = [v for v in variables if is_word_in_vocab_func(v)]
    # get all tokens
    contexts = code.split(" ")
    contexts = [c.split(",") for c in contexts[1:] if c != ""]
    name_src, _, name_dst = zip(*contexts)
    id_to_name = list(set(name_src + name_dst))
    names_to_id = {name: i for i, name in enumerate(id_to_name)}

    assign_field = [{"v": id, "inf" if name in existed_variables else "giv": name}
                    for name, id in names_to_id.items()]
    query_field = [{"a": names_to_id[p[0]], "b": names_to_id[p[2]], "f2": p[1]}
                   for p in contexts]
    query_field.append({"cn":"!=", "n":list(range(len(id_to_name)))})

    # build request
    request = {"params": {"assign": assign_field, "query": query_field},
               "method": "infer",
               "id": "1",
               "jsonrpc": "2.0"}

    response = requests.post('http://pomela4.cslcs.technion.ac.il:5745', json.dumps(request))

    # parse response
    response = json.loads(response.text)
    replace_variables = {id_to_name[r["v"]]: r["inf"].lower() for r in response["result"]
                         if "inf" in r and id_to_name[r["v"]] != r["inf"].lower()}
    # method_name = code.split(" ")[0]
    # print("method:", method_name)
    # print("total vars:", len(variables), variables)
    for original_var, new_var in replace_variables.items():
        # print("replace:", original_var, "(", is_word_in_vocab_func(original_var), ")",
        #       "with", new_var, "(", is_word_in_vocab_func(new_var), ")")
        code = code.replace(" " + original_var + ",", " " + new_var + ",") \
            .replace("," + original_var + " ", "," + new_var + " ")

    return code

def guard_by_vunk(code_sample_with_vars):
    variables, code = common_adversarial.separate_vars_code(code_sample_with_vars)
    if variables != "":
        variables = variables.lower().split(",")
        for original_var in variables:
            code = code.replace(" " + original_var + ",", " " + ONLY_VARIABLE + ",") \
                .replace("," + original_var + " ", "," + ONLY_VARIABLE + " ")

    return code