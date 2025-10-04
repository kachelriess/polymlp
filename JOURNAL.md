# Journal

## Python

I decided to go with Python for the first implementation, as it's my main language.
If you [look at the code](implementations/python/), you will find that I chose not to focus on micro-optimizations.
Instead of specialized loops and views, I used lambdas to define element-wise operations and indexing.
This resulted in a modular matrix data structure, with a [NumPy](https://numpy.org/) / [PyTorch](https://pytorch.org/) hybrid API.
The layers closely emulate the [PyTorch](https://pytorch.org/) API and can be easily extended later on.

Overall, Python is (and might stay) my favorite scripting language, as it's incredibly versatile, and the syntax is clean and simple.
However, without third-party libraries written in higher-performance languages, Python's performance is evidently not satisfactory.
I still like it very much though! :)
