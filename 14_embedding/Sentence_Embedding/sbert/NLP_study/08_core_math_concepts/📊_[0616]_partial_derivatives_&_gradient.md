
# 📘 0615 Partial Derivatives & Gradient

## 📌 Summary  
This note explains the concept of **partial derivatives** and **gradient**, how they differ from ordinary derivatives, and how to compute them step-by-step with examples.  
These are essential concepts in multivariable calculus and deep learning, where functions often depend on more than one variable.

---

## 📚 What I Studied Today

### 1. What is a Partial Derivative?
- When a function has multiple variables (like x, y, z, ...), a **partial derivative** measures how the function changes with respect to **one variable**, **keeping the others constant**.
- In contrast to regular derivatives, partial derivatives focus on one direction of change.

---

### 2. Notation (Partial Derivatives)
- \( \frac{\partial f}{\partial x} \): The partial derivative of function \( f \) with respect to \( x \).
- The symbol **∂** (called "partial") is used instead of **d**, which is used for ordinary derivatives.

---

### 3. Basic Rules

| Expression                    | Result |
|-------------------------------|--------|
| \( \frac{\partial x}{\partial x} \) | 1      |
| \( \frac{\partial y}{\partial x} \) | 0 (y is constant) |

---

### 4. Examples (Partial Derivatives)

#### Example 1:
Let \( f(x, y) = 3xy \)

- \( \frac{\partial f}{\partial x} = 3y \) (treat y as constant)  
- \( \frac{\partial f}{\partial y} = 3x \) (treat x as constant)

#### Example 2:
Let \( f(x) = (x^2 + 1)^2 \)

Step-by-step:

1. Let \( u = x^2 + 1 \), so \( f(x) = u^2 \)
2. Use chain rule:
   \[
   \frac{\partial f}{\partial x} = 2(x^2 + 1) \cdot \frac{\partial (x^2 + 1)}{\partial x}
   \]
   \[
   = 2(x^2 + 1) \cdot 2x = 4x(x^2 + 1)
   \]

---

### 5. What Does "Treating as Constant" Mean?

- When taking a partial derivative with respect to one variable, all other variables are considered fixed numbers.
- For example:
  \[
  f(x, y) = x^2 y
  \]
  \[
  \frac{\partial f}{\partial x} = 2xy \quad (\text{y is a constant})
  \]

---

## 🧭 Gradient (∇f)

### What is a Gradient?
- The **gradient** of a function is a **vector** that contains all its partial derivatives.
- It points in the direction of the **steepest increase** of the function.
- Notation: **∇f** or **grad f**

### Formula (for a scalar-valued function f):

If \( f(x, y, z) \) is a function of three variables, then:
\[
\nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \right)
\]

### Example:

Let \( f(x, y, z) = x^2 + y^2 + z^2 \)

- \( \frac{\partial f}{\partial x} = 2x \)  
- \( \frac{\partial f}{\partial y} = 2y \)  
- \( \frac{\partial f}{\partial z} = 2z \)  

So the gradient is:  
\[
\nabla f = (2x, 2y, 2z)
\]

### ✅ One-Line Summary:
> The **gradient** combines all partial derivatives into a vector that shows how a function increases most rapidly.
