# Study Notes - 2025-06-15

---

# Summary of Today’s Study

- **Derivative**: Instantaneous rate of change of a function, slope of the tangent line  
- **Power Function Differentiation**: \( \frac{d}{dx} x^n = n x^{n-1} \)  
- **Chain Rule (Outer and Inner Differentiation)**: Differentiation of composite functions, derivative of outer function × derivative of inner function  
- **Binomial Coefficient**: Number of combinations, \( \binom{n}{k} = \frac{n!}{k!(n-k)!} \)  
- **Methods for Calculating Binomial Coefficients**: Factorial formula, simplification, symmetry, Pascal’s triangle

---

# Detailed Notes

## 1. Derivative

- **Concept**  
  The instantaneous rate of change of a function at a point, representing the slope of the tangent line.

- **Definition**  
  \[
  f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
  \]

- **Geometric Meaning**  
  - \( f'(x) > 0 \): function increasing  
  - \( f'(x) < 0 \): function decreasing  
  - \( f'(x) = 0 \): possible local max, min, or inflection point

---

## 2. Differentiation of Power Functions

- **Function form**  
  \[
  f(x) = x^n, \quad n \in \mathbb{R}
  \]

- **Differentiation formula**  
  \[
  \frac{d}{dx} x^n = n x^{n-1}
  \]

- **Examples**  
  \[
  \frac{d}{dx} x^3 = 3x^2, \quad \frac{d}{dx} x^{-2} = -2 x^{-3}
  \]

---

## 3. Chain Rule (Outer and Inner Differentiation)

- **Differentiation of composite functions**  
  For \( y = f(g(x)) \),  
  \[
  \frac{dy}{dx} = f'(g(x)) \cdot g'(x)
  \]

- **Outer differentiation**  
  Differentiating the outer function \( f \) treating the inner function \( g(x) \) as the variable.

- **Inner differentiation**  
  Differentiating the inner function \( g(x) \) with respect to \( x \).

- **Examples**  
  - \( y = \sin(x^2) \)  
    \[
    \frac{dy}{dx} = \cos(x^2) \times 2x
    \]

  - \( y = e^{3x + 1} \)  
    \[
    \frac{dy}{dx} = e^{3x + 1} \times 3
    \]

---

## 4. Binomial Coefficient

- **Definition**  
  Number of ways to choose \( k \) items from \( n \) items:  
  \[
  \binom{n}{k} = \frac{n!}{k! (n-k)!}
  \]

- **Notation meaning**  
  - \( n! \): factorial of \( n \) (product of all positive integers from 1 to \( n \))  
  - \( k! \) and \( (n-k)! \) likewise factorials

- **Binomial theorem**  
  \[
  (x + y)^n = \sum_{k=0}^n \binom{n}{k} x^{n-k} y^k
  \]

- **Example**  
  \[
  (x + y)^3 = \binom{3}{0} x^3 y^0 + \binom{3}{1} x^2 y^1 + \binom{3}{2} x^1 y^2 + \binom{3}{3} x^0 y^3
  \]
  Which expands to:  
  \[
  x^3 + 3x^2 y + 3 x y^2 + y^3
  \]

- **Properties**  
  - \(\binom{n}{0} = \binom{n}{n} = 1\)  
  - \(\binom{n}{k} = \binom{n}{n-k}\) (symmetry)

---

## 5. Methods for Calculating Binomial Coefficients

### 5-1. Factorial Basics

- \( n! = n \times (n-1) \times \cdots \times 1 \)  
- Example: \( 5! = 120 \), and by definition \( 0! = 1 \)

### 5-2. Binomial coefficient formula

\[
\binom{n}{k} = \frac{n!}{k! (n-k)!}
\]

### 5-3. Example calculation

\[
\binom{5}{2} = \frac{5!}{2! \times 3!} = \frac{120}{2 \times 6} = 10
\]

### 5-4. Simplify by canceling terms

\[
\binom{n}{k} = \frac{n \times (n-1) \times \cdots \times (n-k+1)}{k \times (k-1) \times \cdots \times 1}
\]

- Example:  
\[
\binom{5}{2} = \frac{5 \times 4}{2 \times 1} = 10
\]

### 5-5. Use symmetry

\[
\binom{n}{k} = \binom{n}{n-k}
\]

### 5-6. Pascal’s Triangle

```
          1
        1   1
      1   2   1
    1   3   3   1
  1   4   6   4   1
1   5  10  10   5   1
```

---

If you want, I can also provide Python code to calculate binomial coefficients!  
Feel free to ask if you want me to add that or explain anything else.
