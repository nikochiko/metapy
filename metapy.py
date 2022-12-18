
class MetapyError(Exception):
    pass

def make_expressions(lines, indent=0):
    if not lines:
        return []

    line = lines[0]
    if not line or is_comment(line):
        return make_expressions(lines[1:], indent=indent)
    elif get_indentation(line) < indent:
        return []
    elif get_indentation(line) > indent:
        return make_expressions(lines[1:], indent=indent)
    elif get_indentation(line) == indent and is_start_of_indent_block(line):
        line = line.replace(":", " :")
        inner_block = make_expressions(lines[1:], indent=get_indentation(lines[1]))
        return [(line.strip(), inner_block)] + make_expressions(lines[1:], indent=indent)
    elif get_indentation(line) == indent:
        return [line.strip()] + make_expressions(lines[1:], indent)
    else:
        raise MetapyError("make_expressions: unhandled case")

def get_indentation(line):
    if line and line[0] == " ":
        return 1 + get_indentation(line[1:])
    else:
        return 0

def has_indentation(line, indent):
    return get_indentation(line) == indent

def is_comment(line):
    return line.lstrip(" ").startswith("#")

def is_start_of_indent_block(line):
    return line.rstrip().endswith(":")

def tokenize_expressions(expressions):

    def tokenize_block(expr):
        return make_block(tokenize_expression(block_start(expr)), tokenize_expressions(block_body(expr)))

    return [tokenize_block(expr) if is_block(expr) else tokenize_expression(expr) for expr in expressions]

def is_block(expr):
    return isinstance(expr, tuple)

def block_start(expr):
    return expr[0]

def block_body(expr):
    return expr[1]

def make_block(start, body):
    return (start, body)

def tokenize_expression(expression, sep=" "):
    result = []
    word = ""
    in_parens = 0

    def refresh_word():
        nonlocal word
        if word:
            result.append(word.strip())
            word = ""

    for char in expression:
        if char == "(":
            in_parens += 1
        elif char == ")":
            in_parens -= 1

        if char == sep and in_parens == 0:
            refresh_word()
        else:
            word += char

    refresh_word()
    return result

def attach_tag(tag, value):
    return [tag, value]

import string

def parse(expr):
    if is_constant(expr):
        return make_constant(expr)
    elif is_variable(expr):
        return make_variable(expr)
    elif is_if(expr):
        return make_if(parse(if_get_condition(expr)), parse_all(if_get_body(expr)))
    elif is_else(expr):
        return make_else(parse_all(else_get_body(expr)))
    elif is_elif(expr):
        return make_elif(parse(elif_get_condition(expr)), parse_all(elif_get_body(expr)))
    elif is_definition(expr):
        return make_definition(definition_get_name(expr), definition_get_params(expr), parse_all(definition_get_body(expr)))
    elif is_assignment(expr):
        return make_assignment(assignment_get_variable(expr), parse(assignment_get_value(expr)))
    elif is_application(expr):
        return make_application(parse(application_get_name(expr)), parse_all(application_get_args(expr)))
    elif is_infix_application(expr):
        return infix_to_prefix(expr)
    elif is_grouping(expr):
        return parse_grouping(expr)
    elif is_singleton(expr):
        return parse(expr[0])
    elif is_compound(expr):
        return parse_all(expr)
    else:
        raise MetapyError(f"unknown expression {repr(expr)}")

def parse_all(exprs):
    parsed_exprs = [parse(expr) for expr in exprs]
    return post_parse(parsed_exprs)

def is_constant(expr):
    try:
        int(expr)
    except (ValueError, TypeError):
        return False
    else:
        return True

def is_variable(expr):
    return isinstance(expr, str) and is_valid_variable_name(expr)

def is_if(expr):
    return is_block(expr) and block_start(expr)[0] == "if"

def is_else(expr):
    return is_block(expr) and block_start(expr)[0] == "else"

def is_elif(expr):
    return is_block(expr) and block_start(expr)[0] == "elif"

def is_assignment(expr):
    return is_compound(expr) and len(expr) >= 3 and expr[1] == "="

def is_definition(expr):
    return is_block(expr) and block_start(expr)[0] == "def"

def is_application(expr):
    if not isinstance(expr, str):
        return False

    depth = 0
    was_depth_1 = False
    for char in expr:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1

        if depth == 0 and char != ")" and (char == " " or not is_valid_variable_name(char)):
            return False
        elif depth == 1:
            was_depth_1 = True

    return was_depth_1 and not is_grouping(expr)

def is_grouping(expr):
    if not isinstance(expr, str) or not expr.startswith("("):
        return False

    depth = 1
    for char in expr[1:]:
        if depth == 0:
            return False
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1

    return True

infix_operators = {"+": "__add__", "-": "__sub__", "*": "__mul__", "/": "__div__",
                   "==": "__eq__", "<": "__lt__", ">": "__gt__", "!=": "__ne__"}

def preprocess_infix(expr):
    if isinstance(expr, str):
        for op in infix_operators:
            expr = expr.replace(op, f" {op} ")
        return tokenize_expression(expr)
    elif isinstance(expr, list):
        for op in infix_operators:
            op_index = find_in_list(expr, op)
            if op_index != -1:
                lhs, rhs = expr[:op_index], expr[op_index+1:]
                return [make_grouping(lhs), op, make_grouping(rhs)]
        return expr
    else:
        return expr

def find_in_list(ls, item):
    try:
        return ls.index(item)
    except ValueError:
        return -1

def is_infix_application(expr):
    expr = preprocess_infix(expr)
    return isinstance(expr, list) and len(expr) >= 3 and any(op in expr for op in infix_operators)

def infix_to_prefix(expr):
    expr = preprocess_infix(expr)
    return make_application(parse(infix_operators[infix_get_operator(expr)]),
                            parse_all([infix_get_left_operand(expr),
                                       infix_get_right_operand(expr)]))

def is_compound(expr):
    return isinstance(expr, list)

def is_singleton(expr):
    return isinstance(expr, list) and len(expr) == 1

def is_keyword(expr):
    return expr in ['if', 'else', 'elif']

def is_valid_variable_name(expr):
    return isinstance(expr, str) and expr and all(c in string.ascii_letters+"_" for c in expr) and not is_keyword(expr)

def is_compound_expression(expr):
    return isinstance(expr, list)

def make_constant(expr):
    return attach_tag("constant", int(expr))

def make_variable(expr):
    return attach_tag("variable", expr)

def make_if(condition, body):
    return attach_tag("if", [condition, body, empty_body])

def make_else(body):
    return attach_tag("else", body)

def make_elif(condition, body):
    return attach_tag("elif", [condition, body, empty_body])

def make_definition(name, params, body):
    return attach_tag("definition", [name, params, body])

def make_assignment(variable, value):
    return attach_tag("assignment", [variable, value])

def make_application(name, args):
    return attach_tag("application", [name, args])

def parse_grouping(expr):
    return parse(expr[1:-1])

def make_grouping(expr):
    return expr

def if_get_condition(expr):
    return block_start(expr)[1:-1]

def if_get_body(expr):
    return block_body(expr)

def else_get_body(expr):
    return block_body(expr)

def elif_get_condition(expr):
    return if_get_condition(expr)

def elif_get_body(expr):
    return if_get_body(expr)

def definition_get_name(expr):
    name_and_params = block_start(expr)[1]
    return name_and_params[:name_and_params.index("(")]

def definition_get_params(expr):
    name_and_params = block_start(expr)[1]
    unstripped_params = name_and_params[name_and_params.index("(")+1:name_and_params.rindex(")")].split(",")
    return [p.strip() for p in unstripped_params]

def definition_get_body(expr):
    return block_body(expr)

def assignment_get_variable(expr):
    return expr[0]

def assignment_get_value(expr):
    return expr[2:]

def application_get_name(expr):
    return expr[:expr.index("(")]

def application_get_args(expr):
    return tokenize_expression(application_get_args_string(expr), sep=",")

def application_get_args_string(expr):
    return expr[expr.index("(")+1:expr.rindex(")")]

def infix_get_operator(expr):
    return expr[1]

def infix_get_left_operand(expr):
    return expr[0]

def infix_get_right_operand(expr):
    return expr[2]

def post_parse(exptree):
    look_for_else = False
    last_node = None

    to_remove = []
    for handle, node in enumerate(exptree):
        if is_if_node(node):
            look_for_else = True
        elif look_for_else:
            if is_else_node(node):
                add_else_body(last_node, else_get_body(node))
                to_remove.append(handle)
                look_for_else = False
            elif is_elif_node(node):
                node = elif_to_if(node)
                add_else_body(last_node, [node])
                to_remove.append(handle)
                look_for_else = True
            else:
                add_else_body(last_node, empty_body)
                look_for_else = False
        elif is_untagged_list(node):
            for child in parse:
                post_parse(child)
        last_node = node
        
    return [node for (i, node) in enumerate(exptree) if i not in to_remove]

def elif_to_if(node):
    return make_if(get_elif_node_condition(node), get_elif_node_body(node))

empty_body = []

def add_else_body(node, body):
    node[1][2] = body

def xapply(function, args):
    if is_primitive_procedure(function):
        return apply_primitive_procedure(function, args)
    elif is_compound_procedure(function):
        return apply_compound_procedure(function, args)
    else:
        raise MetapyError(f"xapply: unknown kind of function: {function}")

def xeval(exp, env):
    if is_constant_node(exp):
        return eval_constant(exp, env)
    elif is_variable_node(exp):
        return eval_variable(exp, env)
    elif is_if_node(exp):
        return eval_if(exp, env)
    elif is_assignment_node(exp):
        return eval_assignment(exp, env)
    elif is_definition_node(exp):
        return eval_definition(exp, env)
    elif is_application_node(exp):
        return eval_application(exp, env)
    elif is_untagged_list(exp):
        return xeval_all(exp, env)
    else:
        raise MetapyError(f"eval: unknown expression: {exp}")

def xeval_sequence(exps, env):
    result = none
    for exp in exps:
        result = xeval(exp, env)

    return result

def xeval_all(exps, env):
    return [xeval(exp, env) for exp in exps]        

def is_untagged_list(exps):
    return isinstance(exps, list) and exps and not isinstance(exps[0], str) or False

def is_tagged_list(ls, tag):
    return isinstance(ls, list) and ls and ls[0] == tag or False

def is_constant_node(exp):
    return is_tagged_list(exp, "constant")

def is_variable_node(exp):
    return is_tagged_list(exp, "variable")

def is_if_node(exp):
    return is_tagged_list(exp, "if")

def is_elif_node(exp):
    return is_tagged_list(exp, "elif")

def is_else_node(exp):
    return is_tagged_list(exp, "else")

def is_definition_node(exp):
    return is_tagged_list(exp, "definition")

def is_assignment_node(exp):
    return is_tagged_list(exp, "assignment")

def is_application_node(exp):
    return is_tagged_list(exp, "application")

none = 0

def eval_constant(exp, env):
    return get_constant_node_value(exp)

def eval_variable(exp, env):
    variable = get_variable_node_name(exp)
    return lookup_variable_value(variable, env)

def eval_if(exp, env):
    condition = get_if_node_condition(exp)
    body = get_if_node_body(exp)
    else_body = get_if_node_else_body(exp)
    if xeval(condition, env):
        return xeval_sequence(body, env)
    else:
        return xeval_sequence(else_body, env)

def eval_definition(exp, env):
    func = make_compound_procedure(params=get_definition_node_params(exp),
                                   body=get_definition_node_body(exp),
                                   env=env)
    name = get_definition_node_name(exp)
    set_variable_value(name, func, env)

def eval_assignment(exp, env):
    variable = get_assignment_node_variable(exp)
    value_exp = get_assignment_node_value(exp)
    value = xeval(value_exp, env)
    set_variable_value(variable, value, env)
    return value

def eval_application(exp, env):
    op = get_application_node_op(exp)
    args = get_application_node_params(exp)
    return xapply(xeval(op, env), xeval_all(args, env))

def get_variable_node_name(exp):
    return exp[1]

def get_if_node_condition(exp):
    return exp[1][0]

def get_if_node_body(exp):
    return exp[1][1]

def get_if_node_else_body(exp):
    return exp[1][2]

def get_elif_node_condition(exp):
    return exp[1][0]

def get_elif_node_body(exp):
    return exp[1][1]

def get_assignment_node_variable(exp):
    return exp[1][0]

def get_assignment_node_value(exp):
    return exp[1][1]

def get_application_node_op(exp):
    return exp[1][0]

def get_application_node_params(exp):
    return exp[1][1]

def get_constant_node_value(exp):
    return exp[1]

def get_definition_node_name(exp):
    return exp[1][0]

def get_definition_node_params(exp):
    return exp[1][1]

def get_definition_node_body(exp):
    return exp[1][2]

def is_true(value):
    return value != 0

def make_frame(table=None):
    return table or {}

def lookup_frame_variable(var, frame):
    return var in frame and frame[var]

def set_frame_variable(var, val, frame):
    frame[var] = val

def make_env(frames=None):
    return frames or [make_frame()]

def get_top_frame(env):
    return env[0]

def get_parent_env(env):
    return env[1:]

def extend_env(variables, values, env):
    frame = make_frame(dict(zip(variables, values)))
    return [frame] + env

def lookup_variable_value(variable, env):
    if not env:
        raise MetapyError(f"variable not found: {variable}")

    value = lookup_frame_variable(variable, get_top_frame(env))
    if value is not False:
        return value
    else:
        return lookup_variable_value(variable, get_parent_env(env))

def set_variable_value(variable, value, env):
    frame = get_top_frame(env)
    set_frame_variable(variable, value, frame)

def make_primitive_procedure(proc):
    return attach_tag("primitive-procedure", proc)

def make_compound_procedure(params, body, env):
    return attach_tag("compound-procedure", [params, body, env])

def is_primitive_procedure(obj):
    return is_tagged_list(obj, "primitive-procedure")

def primitive_procedure_get_proc(obj):
    return obj[1]

def is_compound_procedure(obj):
    return is_tagged_list(obj, "compound-procedure")

def compound_procedure_get_params(obj):
    return obj[1][0]

def compound_procedure_get_body(obj):
    return obj[1][1]

def compound_procedure_get_env(obj):
    return obj[1][2]

def apply_primitive_procedure(procedure, args):
    func = primitive_procedure_get_proc(procedure)
    return func(*args)

def apply_compound_procedure(procedure, args):
    params = compound_procedure_get_params(procedure)
    body = compound_procedure_get_body(procedure)
    parent_env = compound_procedure_get_env(procedure)
    return xeval_sequence(body, env=extend_env(variables=params, values=args, env=parent_env))

## trying out

import operator
import sys

primitives = {"__add__": operator.add,
              "__sub__": operator.sub,
              "__mul__": operator.mul,
              "__div__": operator.floordiv,
              "__eq__": lambda x, y: int(operator.eq(x, y)),
              "__lt__": lambda x, y: int(operator.lt(x, y)),
              "__gt__": lambda x, y: int(operator.gt(x, y)),
              "__ne__": lambda x, y: int(operator.ne(x, y)),
              "print": print,
              "exit": sys.exit}

def setup_env():
    return make_env(frames=[
        make_frame(table={key: make_primitive_procedure(val)
                          for key, val in primitives.items()})
    ])

global_env = setup_env()


def try_eval(code, env=global_env):
    lines = parse_all(tokenize_expressions(make_expressions(code.split("\n"))))
    try:
        return xeval_sequence(lines, env)
    except MetapyError as e:
        return f"Error: {e}"

def main():
    code = ""
    nested = False
    prompt = ">>> "
    while True:
        line = input(prompt)
        if line.rstrip().endswith(":"):
            nested = True
            code += "\n" + line
            prompt = "... "
        elif nested and line == "":
            nested = False
            print(try_eval(code))
            code = ""
            prompt = ">>> "
        elif nested:
            code += "\n" + line
            prompt = "... "
        else:
            code += "\n" + line
            print(try_eval(code))
            code = ""


if __name__ == "__main__":
    main()
