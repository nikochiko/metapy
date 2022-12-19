# MetaPy

A toy metacircular evaluator for Python, without using an AST library.

#### Supported:
- Only int data type
- Variables
- Function definition and application
- Infix operations
- if-elif-else statements

#### Notes
- This assumes the syntax to be correct. Validating syntax must be done before passing to the evaluator.
- There are no return statements. The value of the last evaluated expression will be taken to be the return value of a function.
- Syntax support is limited. Semi-colons are not allowed. `def` or `if` with bodies on the same line after colon are not supported. Variable names must be made of letters or underscores.

There are no constructs for loops. Iteration must be represented as recursion instead.

#### Example:

```python
>>> def ack(x, y):
...    if y == 0:
...        0
...    elif x == 0:
...        2 * y
...    elif y == 1:
...        2
...    else:
...        ack(x - 1, ack(x, y - 1))
...
None
>>> ack(1, 5)
32
>>> x = 2
>>> ack(x, x * x)
65536
```

To start the interactive console, run the `metapy.py` file from terminal:

```shell
$ python3 metapy.py
>>> 
```
