def split_list(inp: list) -> tuple[list, list]:
    """
    Attempts to split a list of even length in half

    ## Arguments:
        `inp` (`list`): The list to split

    ## Returns:
        `tuple[list, list]`: 2 halves of the `inp`

    ## Raises:
        `ValueError`: When the list is not even
    """

    if not len(inp) % 2:
        half = len(inp) // 2
        return inp[:half], inp[half:]

    raise ValueError("List provided was not of even length")


def sigmoid(x: float) -> float:
    """
    Sigmoid activation function, normalizes any number to [0, 1)

    ## Arguments:
        `x` (`float`): The value to normalize

    ## Returns:
        `float`: The normalized value
    """

    return 1 / (1 + pow(2.71828, -x))


def sigmoid_derivative(x: float) -> float:
    """
    Derivative activation function, used in backpropagation

    ## Arguments:
        `x` (`float`): The activated value to derive

    ## Returns:
        `float`: The derived value
    """

    return x * (1 - x)
