import numpy as np


def eval_expression(exp: list, image: np.ndarray = None):
    expression = ""

    for token in exp:
        if token[0] == "c":
            channel = eval(token[1:])
            expression += f"(image[{channel}] + 0.00001)"  # To prevent divide by zero
        elif token == "sq":
            expression += "**2"
        elif token == "sqrt":
            expression += "**0.5"
        elif token == "=":
            break
        else:
            expression += token

    try:
        return eval(expression)
    except (SyntaxError, FloatingPointError):
        return False


if __name__ == "__main__":
    image = np.random.rand(3, 2, 2)
    print(image)
    print(eval_expression(["(", "c1", "/", "c3", "sqrt", ")", "sq"], image))
