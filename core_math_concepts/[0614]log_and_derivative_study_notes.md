
# Summary

- Learned the fundamental properties of logarithms (constant multiple, addition, subtraction, base change, reciprocal rules).  
- Understood the concept of limits and the rigorous epsilon-delta definition for function convergence.  
- Learned the definition of derivatives and how to compute the instantaneous slope of a function with examples.  
- Explored the sigmoid function in logistic regression to convert inputs into probability values between 0 and 1.  
- Understood the cost function to quantify prediction errors and interpret its meaning in logistic regression.  
- Learned the principle of gradient descent for minimizing the cost function to find optimal parameters.

---

# Study Notes (Complete)

## 1. Logarithm Properties

### 1-1. Definition  
\[
\log_b a = c \quad \Longleftrightarrow \quad b^c = a
\]  
Where \(b\) is the base, \(a\) the argument, and \(c\) the logarithm value.

### 1-2. Key Properties

- **Constant Multiple Rule**  
\[
\log_b (a^k) = k \log_b a
\]

- **Addition Rule (Product Rule)**  
\[
\log_b (xy) = \log_b x + \log_b y
\]

- **Subtraction Rule (Quotient Rule)**  
\[
\log_b \left(\frac{x}{y}\right) = \log_b x - \log_b y
\]

- **Base Change Rule**  
\[
\log_b a = \frac{\log_k a}{\log_k b}
\]  
(Can convert logarithms to any base \(k\))

- **Reciprocal Rule**  
\[
\log_b \left(\frac{1}{a}\right) = -\log_b a
\]

---

## 2. Limits and Epsilon-Delta Definition

### 2-1. Intuitive Meaning  
Function \( f(x) \) approaches \( L \) as \( x \) approaches \( a \), denoted:  
\[
\lim_{x \to a} f(x) = L
\]

### 2-2. Rigorous Definition (Epsilon-Delta)  
For every small number \( \epsilon > 0 \), there exists a \( \delta > 0 \) such that:  
\[
|x - a| < \delta \implies |f(x) - L| < \epsilon
\]

Meaning if \(x\) is within \(\delta\) of \(a\), then \(f(x)\) is within \(\epsilon\) of \(L\).

---

## 3. Derivatives

### 3-1. Definition  
The derivative of \( f(x) \) at \( x=a \) is the slope of the tangent:  
\[
f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}
\]

### 3-2. Example: Find derivative of \( f(x) = x^2 \) at \( x=2 \)

- Calculate:  
\[
f(2+h) = (2+h)^2 = 4 + 4h + h^2
\]  
\[
f(2) = 2^2 = 4
\]  
\[
\frac{f(2+h) - f(2)}{h} = \frac{4 + 4h + h^2 - 4}{h} = \frac{4h + h^2}{h} = 4 + h
\]

- Take limit \( h \to 0 \):  
\[
f'(2) = \lim_{h \to 0} (4 + h) = 4
\]

So the slope at \( x=2 \) is 4.

---

## 4. Logistic Regression Related Functions

---

### 4-1. Sigmoid Function

- Formula:  
\[
h(x, \theta) = \frac{1}{1 + e^{-\theta^T x}}
\]

- Purpose:  
Convert input \( \theta^T x \) into a probability between 0 and 1.

- Example:  
If \( \theta^T x = 2.2 \), then  
\[
h(x, \theta) = \frac{1}{1 + e^{-2.2}} \approx 0.9
\]

This means the probability that \( x \) belongs to class 1 is about 90%.

---

### 4-2. Cost Function

- Purpose:  
Quantify the error between predicted values and actual labels, and find parameters \( \theta \) that minimize this error.

- Formula:  
\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h(x^{(i)}, \theta)) + (1 - y^{(i)}) \log(1 - h(x^{(i)}, \theta)) \right]
\]

- Interpretation:  
  - If \( y^{(i)} = 1 \), cost is:  
  \[
  -\log(h(x^{(i)}, \theta))
  \]  
  Lower if predicted probability close to 1.

  - If \( y^{(i)} = 0 \), cost is:  
  \[
  -\log(1 - h(x^{(i)}, \theta))
  \]  
  Lower if predicted probability close to 0.

- Example:  
For \( y^{(1)} = 1 \), \( h(x^{(1)}, \theta) = 0.9 \),  
\[
\text{Cost} = -\log(0.9) \approx 0.105
\]

---

## 5. Gradient Descent

- Purpose:  
Find the optimal parameters \( \theta \) that minimize the cost function \( J(\theta) \).

- Update rule:  
\[
\theta := \theta - \alpha \cdot \nabla_\theta J(\theta)
\]

- Explanation:  
  - \( \alpha \) is the learning rate  
  - \( \nabla_\theta J(\theta) \) is the gradient of the cost  
  - Move \( \theta \) in the direction that reduces cost iteratively

---
