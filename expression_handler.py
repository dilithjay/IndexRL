import numpy as np


def eval_expression(exp: list, image: np.ndarray = None):
    expression = ""
    constant = 1

    for token in exp:
        if isinstance(token, float):
            constant = token
            continue

        if token[0] == "c":
            channel = eval(token[1:]) - 1
            expression += f"{constant}*image[{channel}]"
        elif token == "p":
            expression += f"**{constant}"
        elif token == "(":
            expression += f"{constant}*("
        elif token == "1":
            expression += str(constant)
        else:
            expression += token
        constant = 1

    try:
        return eval(expression)
    except:
        return False


if __name__ == "__main__":
    image = np.random.rand(3, 5, 5)
    print(eval_expression(["(", "c1", "+", "c3", ")"], image))
