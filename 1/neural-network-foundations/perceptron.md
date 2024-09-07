# Perceptron

Originally, the perceptron was designed to take a number of binary inputs and produce one binary output.

However, if an input was only represented by `1`s and `0`s, it would be difficult to determine which input is more important than the others.

The idea was to use different <u>weights</u> to represent the importance of each input. The sum of these weighted values should be greater than a set <u>threshold</u> value before making a decision (output).

<p align="center">
  <img src="./_attachments/perceptron.jpg" alt="perceptron" />
</p>
`x(n)` = Input
`w(n)` = Weights

---

## Example:

Let’s simulate how a perceptron processes inputs to make a decision.

# Should You Learn Rust?

| Criteria                                                                     | Input (1 or 0) | Weight |
| ---------------------------------------------------------------------------- | -------------- | ------ |
| "You love fighting the borrow checker?"                                      | 1              | 1.0    |
| "You want to write code that's faster than your caffeine intake?"            | 1              | 0.9    |
| "You enjoy syscalls at 3 AM?"                                                | 1              | 0.7    |
| "You dream about the stack and heap?"                                        | 0              | 0.6    |
| "Error: The method exists but the following trait bounds were not satisfied" | 1              | 1.0    |

### Based on Frank Rosenblatt

1. Set a threshold value
2. Multiply all inputs with its weights
3. Sum all the results
4. Activate the outputs

**1. Set a threshold value**

`Threshold` = `1.5`

**2. Multiply all inputs with its weights**

```plaintext
x1 × w1 = 1 × 1.0 = 1
x2 × w2 = 1 × 0.9 = 0.9
x3 × w3 = 1 × 0.7 = 0.7
x4 × w4 = 0 × 0.6 = 0
x5 × w5 = 1 × 1.0 = 1
```

**3. Sum all the results**

```plaintext
1 + 0.9 + 0.7 + 0 + 1 = 3.6
```

_The result is called the <u>Weighted Sum</u>_

**4. Activate the output**

```python
def activation_function(
    threshold: float = 1.5,
    weighted_sum: float = 3.6
) -> bool:
    """
    Determines if the weighted sum exceeds the threshold.

    Args:
        threshold (float): The threshold value for activation. Default is 1.5.
        weighted_sum (float): The calculated weighted sum. Default is 3.6.

    Returns:
        bool: True if the weighted sum is greater than the threshold, otherwise False.
    """
    return weighted_sum > threshold
```

_The result would be <u>True</u> since our `weighted_sum` is bigger than the set `threshold`_

---

# Further Topics

- MLP
- Activation Functions
- Perceptron Learning Algorithms
