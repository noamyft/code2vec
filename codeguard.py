import json
import requests

ONLY_VARIABLE = "VUNK"

def guard_by_n2p(code_sample_with_vars):
    var_code_split_index = code_sample_with_vars.find(" ")
    code = code_sample_with_vars[var_code_split_index + 1:]
    variables = code_sample_with_vars[:var_code_split_index]
    if variables == "":
        return code
    else:
        variables = variables.lower().split(",")

    # get all tokens
    contexts = code.split(" ")
    contexts = [c.split(",") for c in contexts[1:] if c != ""]
    name_src, _, name_dst = zip(*contexts)
    id_to_name = list(set(name_src + name_dst))
    names_to_id = {name: i for i, name in enumerate(id_to_name)}

    assign_field = [{"v": id, "inf" if name in variables else "giv": name}
                    for name, id in names_to_id.items()]
    query_field = [{"a": names_to_id[p[0]], "b": names_to_id[p[2]], "f2": p[1]}
                   for p in contexts]
    # build request
    request = {"params": {"assign": assign_field, "query": query_field},
               "method": "infer",
               "id": "1",
               "jsonrpc": "2.0"}

    response = requests.post('http://pomela4.cslcs.technion.ac.il:5745', json.dumps(request))

    # parse response
    response = json.loads(response.text)
    replace_variables = {id_to_name[r["v"]]: r["inf"] for r in response["result"]
                         if "inf" in r and id_to_name[r["v"]] != r["inf"]}
    for original_var, new_var in replace_variables.items():
        code = code.replace(" " + original_var + ",", " " + new_var + ",") \
            .replace("," + original_var + " ", "," + new_var + " ")

    return code

def guard_by_vunk(code_sample_with_vars):
    var_code_split_index = code_sample_with_vars.find(" ")
    code = code_sample_with_vars[var_code_split_index + 1:]
    variables = code_sample_with_vars[:var_code_split_index]
    if variables != "":
        variables = variables.lower().split(",")
        for original_var in variables:
            code = code.replace(" " + original_var + ",", " " + ONLY_VARIABLE + ",") \
                .replace("," + original_var + " ", "," + ONLY_VARIABLE + " ")

    return code