import numpy as np
from PyModuli.TestModule1 import hello_from_test_module
from PyModuli.SimpleTest import experimental_train_and_test


if __name__ == "__main__":
    # Prijmer našeg modula:
    hello_from_test_module()

    # Test za .venv
    print(np.e)

    # Poziv za jednostavan test summary.
    experimental_train_and_test()
