import numpy as np


def eval_expression(exp: list, image: np.ndarray = None):
    expression = ""

    for token in exp:
        if token[0] == "c":
            channel = eval(token[1:]) - 1
            expression += f"image[{channel}]"
        elif token == "sq":
            expression += "**2"
        elif token == "sqrt":
            expression += "**0.5"
        elif token == "=":
            expression += "="
            break
        else:
            expression += token
    
    if expression[-1] != '=':
        return False
    try:
        return eval(expression)
    except:
        return False


if __name__ == "__main__":
    image = np.random.rand(3, 2, 2)
    print(image)
    print(eval_expression(["(", "c1", "/", "c3", "sqrt", ")", "sq"], image))
