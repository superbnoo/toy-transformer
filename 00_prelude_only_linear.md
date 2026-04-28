# Proof: Neural Networks Without Activation Functions Are Linear

## 1. Single Perceptron

A single perceptron without an activation function computes:

$$
y = \mathbf{w}^\top \mathbf{x} + b
$$

Expanding:

$$
y = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

This is a linear (affine) function of $\mathbf{x}$.

---

## 2. One Layer with Multiple Neurons

A layer with multiple neurons can be written as:

$$
\mathbf{h} = W_1 \mathbf{x} + \mathbf{b}_1
$$

This is still a linear transformation.

---

## 3. Two-Layer Network

Add another layer:

$$
\mathbf{y} = W_2 \mathbf{h} + \mathbf{b}_2
$$

Substitute $\mathbf{h}$:

$$
\mathbf{y} = W_2 (W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

Distribute:

$$
\mathbf{y} = W_2 W_1 \mathbf{x} + W_2 \mathbf{b}_1 + \mathbf{b}_2
$$

Let:

$$
W = W_2 W_1
$$

$$
\mathbf{b} = W_2 \mathbf{b}_1 + \mathbf{b}_2
$$

Then:

$$
\mathbf{y} = W \mathbf{x} + \mathbf{b}
$$

This is still a linear (affine) function.

---

## 4. General Case: L Layers

For an $L$-layer network:

$$
\mathbf{y} = W_L W_{L-1} \cdots W_1 \mathbf{x} + \mathbf{b}
$$

(where $\mathbf{b}$ is the combined bias term)

Thus, the entire network is equivalent to a single linear transformation.

---

## 5. Conclusion

A neural network without activation functions satisfies:

- Composition of linear functions is linear
- Multiple layers collapse into one

$$
\text{Linear} \circ \text{Linear} \circ \cdots \circ \text{Linear} = \text{Linear}
$$

Therefore, such a network can only represent linear (affine) functions.

---

## 6. Key Insight

Without activation functions, depth does not increase expressive power.

To model non-linear relationships, activation functions are required.