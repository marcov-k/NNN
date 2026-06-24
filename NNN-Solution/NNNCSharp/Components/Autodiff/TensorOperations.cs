using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NNNCSharp.Components.Autodiff;

public partial class Tensor
{
    // Element-wise algebraic operations

    /// <summary>
    /// Adds two tensors using element-wise addition.
    /// </summary>
    /// <param name="a">First tensor to add.</param>
    /// <param name="b">Second tensor to add.</param>
    /// <returns>Tensor containing the element-wise sum of the two input tensors.</returns>
    public static Tensor operator +(Tensor a, Tensor b)
    {
        var owner = a.RequiresGrad ? a : b;
        Tensor result = GetResultTensor(owner, owner.Dimensions, a.RequiresGrad || b.RequiresGrad);

        // Vectorize inputs and results
        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
        var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized sum
        for (int i = 0; i < aVecs.Length; i++)
        {
            rVecs[i] = aVecs[i] + bVecs[i];
        }

        // Clean up unvectorized tail
        for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = a[i] + b[i];
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.AddRange([a, b]);

            // Gradient calculation function for r = a + b -> dr/da = 1, dr/db = 1
            result._backward = () =>
            {
                if (!a.RequiresGrad && !b.RequiresGrad) return;

                // Vectorize gradients
                var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients of parents
                for (int i = 0; i < rgVecs.Length; i++)
                {
                    if (a.RequiresGrad) agVecs[i] += rgVecs[i];
                    if (b.RequiresGrad) bgVecs[i] += rgVecs[i];
                }

                // Clean up unvectorized tails
                for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    if (a.RequiresGrad) a.Grad[i] += result.Grad[i];
                    if (b.RequiresGrad) b.Grad[i] += result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Adds a scalar to every element in a tensor.
    /// </summary>
    /// <param name="a">Tensor to add.</param>
    /// <param name="b">Scalar to add.</param>
    /// <returns>Tensor containing the result of adding the scalar to every element in the input tensor.</returns>
    public static Tensor operator +(Tensor a, double b)
    {
        Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

        // Vectorize inputs and results
        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
        var vb = new Vector<double>(b); // splat scalar into a vector
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized sum
        for (int i = 0; i < aVecs.Length; i++)
        {
            rVecs[i] = aVecs[i] + vb;
        }

        // Clean up unvectorized tail
        for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = a[i] + b;
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(a);

            // Gradient calculation function for r = a + b -> dr/da = 1, dr/db = 1
            result._backward = () =>
            {
                if (!a.RequiresGrad) return;

                // Vectorize gradients
                var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parent
                for (int i = 0; i < agVecs.Length; i++)
                {
                    agVecs[i] += rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                {
                    a.Grad[i] += result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Adds a scalar to every element in a tensor.
    /// </summary>
    /// <param name="a">Scalar to add.</param>
    /// <param name="b">Tensor to add.</param>
    /// <returns>Tensor containing the result of adding the scalar to every value in the input tensor.</returns>
    public static Tensor operator +(double a, Tensor b) => b + a; // commutative operation -> a + b = b + a

    /// <summary>
    /// Subtracts a tensor from another tensor using element-wise subtraction.
    /// </summary>
    /// <param name="a">Tensor to subtract from.</param>
    /// <param name="b">Tensor to subtract.</param>
    /// <returns>Tensor containing the element-wise difference of the two input tensors.</returns>
    public static Tensor operator -(Tensor a, Tensor b)
    {
        var owner = a.RequiresGrad ? a : b;
        Tensor result = GetResultTensor(owner, owner.Dimensions, a.RequiresGrad || b.RequiresGrad);

        // Vectorize inputs and results
        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
        var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized difference
        for (int i = 0; i < aVecs.Length; i++)
        {
            rVecs[i] = aVecs[i] - bVecs[i];
        }

        // Clean up unvectorized tail
        for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = a[i] - b[i];
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.AddRange([a, b]);

            // Gradient calculation function for r = a - b -> dr/da = 1; dr/db = -1
            result._backward = () =>
            {
                if (!a.RequiresGrad && !b.RequiresGrad) return;

                // Vectorize gradients
                var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parents
                for (int i = 0; i < rgVecs.Length; i++)
                {
                    if (a.RequiresGrad) agVecs[i] += rgVecs[i];
                    if (b.RequiresGrad) bgVecs[i] -= rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    if (a.RequiresGrad) a.Grad[i] += result.Grad[i];
                    if (b.RequiresGrad) b.Grad[i] -= result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Subtracts a scalar from every element in a tensor.
    /// </summary>
    /// <param name="a">Tensor to subtract from.</param>
    /// <param name="b">Scalar to subtract.</param>
    /// <returns>Tensor containing the element-wise difference of the input tensor and input scalar.</returns>
    public static Tensor operator -(Tensor a, double b)
    {
        Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

        // Vectorize inputs and results
        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
        var vb = new Vector<double>(b); // splat scalar into a vector
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized difference
        for (int i = 0; i < aVecs.Length; i++)
        {
            rVecs[i] = aVecs[i] - vb;
        }

        // Clean up unvectorized tail
        for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = a[i] - b;
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(a);

            // Gradient calculation function for r = a - b -> dr/da = 1; dr/db = -1
            result._backward = () =>
            {
                if (!a.RequiresGrad) return;

                // Vectorize gradients
                var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parent
                for (int i = 0; i < agVecs.Length; i++)
                {
                    agVecs[i] += rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                {
                    a.Grad[i] += result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Subtracts every element in a tensor from a scalar.
    /// </summary>
    /// <param name="a">Scalar to subtract from.</param>
    /// <param name="b">Tensor to subtract.</param>
    /// <returns>Tensor containing the difference of the input scalar and every element in the input tensor.</returns>
    public static Tensor operator -(double a, Tensor b)
    {
        Tensor result = GetResultTensor(b, b.Dimensions, b.RequiresGrad);

        // Vectorize inputs and results
        var va = new Vector<double>(a); // splat scalar into a vector
        var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized difference
        for (int i = 0; i < bVecs.Length; i++)
        {
            rVecs[i] = va - bVecs[i];
        }

        // Clean up unvectorized tail
        for (int i = bVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = a - b[i];
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(b);

            // Gradient calculation function for r = a - b -> dr/da = 1; dr/db = -1
            result._backward = () =>
            {
                if (!b.RequiresGrad) return;

                // Vectorize gradients
                var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parent
                for (int i = 0; i < bgVecs.Length; i++)
                {
                    bgVecs[i] -= rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = bgVecs.Length * VectorSize; i < b.ElementCount; i++)
                {
                    b.Grad[i] -= result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Multiplies two tensors using element-wise multiplication.
    /// </summary>
    /// <param name="a">First tensor to multiply.</param>
    /// <param name="b">Second tensor to multiply.</param>
    /// <returns>Tensor containing the element-wise product of the two input tensors.</returns>
    public static Tensor operator *(Tensor a, Tensor b)
    {
        var owner = a.RequiresGrad ? a : b;
        Tensor result = GetResultTensor(owner, owner.Dimensions, a.RequiresGrad || b.RequiresGrad);

        // Vectorize inputs and results
        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
        var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized product
        for (int i = 0; i < aVecs.Length; i++)
        {
            rVecs[i] = aVecs[i] * bVecs[i];
        }

        // Clean up unvectorized tail
        for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = a[i] * b[i];
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.AddRange([a, b]);

            // Gradient calculation function for r = a * b -> dr/da = b; dr/db = a
            result._backward = () =>
            {
                if (!a.RequiresGrad && !b.RequiresGrad) return;

                // Vectorize inputs and gradients
                var avVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
                var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                var bvVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
                var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parents
                for (int i = 0; i < rgVecs.Length; i++)
                {
                    if (a.RequiresGrad) agVecs[i] += bvVecs[i] * rgVecs[i];
                    if (b.RequiresGrad) bgVecs[i] += avVecs[i] * rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    if (a.RequiresGrad) a.Grad[i] += b[i] * result.Grad[i];
                    if (b.RequiresGrad) b.Grad[i] += a[i] * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Mutliplies every element in a tensor by a scalar.
    /// </summary>
    /// <param name="a">Tensor to multiply.</param>
    /// <param name="b">Scalar to multiply by.</param>
    /// <returns>Tensor containing the product of every element in the input tensor and the input scalar.</returns>
    public static Tensor operator *(Tensor a, double b)
    {
        Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

        // Vectorize inputs and results
        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
        var vb = new Vector<double>(b); // splat constant into vector
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized product
        for (int i = 0; i < aVecs.Length; i++)
        {
            rVecs[i] = aVecs[i] * vb;
        }

        // Clean up unvectorized tail
        for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = a[i] * b;
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(a);

            // Gradient calculation function for r = a * b -> dr/da = b; dr/db = a
            result._backward = () =>
            {
                if (!a.RequiresGrad) return;

                // Vectorize inputs and gradients
                var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                var vb = new Vector<double>(b); // splat scalar into vector
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parent
                for (int i = 0; i < agVecs.Length; i++)
                {
                    agVecs[i] += vb * rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                {
                    a.Grad[i] += b * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Multiplies every element in a tensor by a scalar.
    /// </summary>
    /// <param name="a">Scalar to multiply by.</param>
    /// <param name="b">Tensor to multiply.</param>
    /// <returns>Tensor containing the product of every element in the input tensor and the input scalar.</returns>
    public static Tensor operator *(double a, Tensor b) => b * a; // commutative operation -> a * b = b * a

    /// <summary>
    /// Divides a tensor by another tensor using element-wise division.
    /// </summary>
    /// <param name="a">Tensor to divide.</param>
    /// <param name="b">Tensor to divide by.</param>
    /// <returns>Tensor containing the element-wise quotient of the two input tensors.</returns>
    public static Tensor operator /(Tensor a, Tensor b)
    {
        var owner = a.RequiresGrad ? a : b;
        Tensor result = GetResultTensor(owner, owner.Dimensions, a.RequiresGrad || b.RequiresGrad);

        // Vectorize inputs and results
        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
        var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized quotient
        for (int i = 0; i < aVecs.Length; i++)
        {
            rVecs[i] = aVecs[i] / bVecs[i];
        }

        // Clean up unvectorized tail
        for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = a[i] / b[i];
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.AddRange([a, b]);

            // Gradient calculation function for r = a / b -> dr/da = 1 / b; dr/db = a * (1 / b^2)
            result._backward = () =>
            {
                if (!a.RequiresGrad && !b.RequiresGrad) return;

                // Vectorize inputs and gradients
                var avVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
                var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                var bvVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
                var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parents
                for (int i = 0; i < rgVecs.Length; i++)
                {
                    if (a.RequiresGrad) agVecs[i] += rgVecs[i] / bvVecs[i];
                    if (b.RequiresGrad) bgVecs[i] -= (avVecs[i] / (bvVecs[i] * bvVecs[i])) * rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    if (a.RequiresGrad) a.Grad[i] += result.Grad[i] / b[i];
                    if (b.RequiresGrad) b.Grad[i] -= (a[i] / (b[i] * b[i])) * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Divides every element in a tensor by a scalar.
    /// </summary>
    /// <param name="a">Tensor to divide.</param>
    /// <param name="b">Scalar to divide by.</param>
    /// <returns>Tensor containing the quotient of every element in the input tensor and the input scalar.</returns>
    public static Tensor operator /(Tensor a, double b)
    {
        Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

        // Vectorize inputs and results
        var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
        var vb = new Vector<double>(b); // splat scalar into vector
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized quotient
        for (int i = 0; i < aVecs.Length; i++)
        {
            rVecs[i] = aVecs[i] / vb;
        }

        // Clean up unvectorized tail
        for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = a[i] / b;
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(a);

            // Gradient calculation function for r = a / b -> dr/da = 1 / b; dr/db = a * (1 / b^2)
            result._backward = () =>
            {
                if (!a.RequiresGrad) return;

                // Vectorize inputs and gradients
                var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                var vb = new Vector<double>(b); // splat constant into vector
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parent
                for (int i = 0; i < agVecs.Length; i++)
                {
                    agVecs[i] += rgVecs[i] / vb;
                }

                // Clean up unvectorized tail
                for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                {
                    a.Grad[i] += (1.0 / b) * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Divides a scalar by every element in a tensor.
    /// </summary>
    /// <param name="a">Scalar to divide.</param>
    /// <param name="b">Tensor to divide by.</param>
    /// <returns>Tensor containing the quotient of the input scalar and every element in the input tensor.</returns>
    public static Tensor operator /(double a, Tensor b)
    {
        Tensor result = GetResultTensor(b, b.Dimensions, b.RequiresGrad);

        // Vectorize inputs and results
        var va = new Vector<double>(a); // splat scalar into vector
        var bVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized quotient
        for (int i = 0; i < bVecs.Length; i++)
        {
            rVecs[i] = va / bVecs[i];
        }

        // Clean up unvectorized tail
        for (int i = bVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = a / b[i];
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(b);

            // Gradient calculation function for r = a / b -> dr/da = 1 / b; dr/db = a * (1 / b^2)
            result._backward = () =>
            {
                if (!b.RequiresGrad) return;

                // Vectorize inputs and gradients
                var va = new Vector<double>(a); // splat scalar into vector
                var bvVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Data.AsSpan());
                var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(b.Grad.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorize gradients for parent
                for (int i = 0; i < bgVecs.Length; i++)
                {
                    bgVecs[i] -= (va / (bvVecs[i] * bvVecs[i])) * rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = bgVecs.Length * VectorSize; i < b.ElementCount; i++)
                {
                    b.Grad[i] -= (a / (b[i] * b[i])) * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Raises a tensor to the power of another tensor using element-wise exponentiation.
    /// </summary>
    /// <param name="a">Tensor to exponentiate.</param>
    /// <param name="exp">Tensor to raise to the power of.</param>
    /// <returns>Tensor containing the element-wise power of the two input tensors.</returns>
    public static Tensor Pow(Tensor a, Tensor exp)
    {
        var owner = a.RequiresGrad ? a : exp;
        Tensor result = GetResultTensor(owner, owner.Dimensions, a.RequiresGrad || exp.RequiresGrad);

        // Calculate power sequentially - no general exponentiation vectorization available
        for (int i = 0; i < result.ElementCount; i++)
        {
            result[i] = Math.Pow(a[i], exp[i]);
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.AddRange([a, exp]);

            // Gradient calculation function for r = a^b -> dr/da = b * a^(b - 1); dr/db = a^b * ln(a)
            result._backward = () =>
            {
                if (!a.RequiresGrad && !exp.RequiresGrad) return;

                // Calculate gradients for parents sequentially - no general exponentiation vectorization available
                for (int i = 0; i < result.ElementCount; i++)
                {
                    if (a.RequiresGrad) a.Grad[i] += exp[i] * Math.Pow(a[i], exp[i] - 1.0) * result.Grad[i];
                    if (exp.RequiresGrad) exp.Grad[i] += result[i] * Math.Log(a[i]) * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Raises every element in a tensor to the power of a scalar.
    /// </summary>
    /// <param name="a">Tensor to exponentiate.</param>
    /// <param name="exp">Scalar to raise to the power of.</param>
    /// <returns>Tensor containing the power of every element in the input tensor and the input scalar.</returns>
    public static Tensor Pow(Tensor a, double exp)
    {
        Tensor result = GetResultTensor(a, a.Dimensions, a.RequiresGrad);

        // Special vectorization cases - power of 2 and square root
        if (exp == 2.0 || exp == 0.5)
        {
            // Vectorize inputs and results
            var aVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
            var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

            // Calculate vectorized power
            for (int i = 0; i < aVecs.Length; i++)
            {
                rVecs[i] = exp == 2.0 ? aVecs[i] * aVecs[i] : Vector.SquareRoot(aVecs[i]);
            }

            // Clean up unvectorized tail
            for (int i = aVecs.Length * VectorSize; i < result.ElementCount; i++)
            {
                result[i] = exp == 2.0 ? a[i] * a[i] : Math.Sqrt(a[i]);
            }
        }
        else // Calculate power sequentially - no general exponentiation vectorization available
        {
            for (int i = 0; i < result.ElementCount; i++)
            {
                result[i] = Math.Pow(a[i], exp);
            }
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(a);

            // Gradient calculation function for r = a^b -> dr/da = b * a^(b - 1); dr/db = a^b * ln(a)
            result._backward = () =>
            {
                if (!a.RequiresGrad) return;

                // Special vectorization case - power of 2 and square root
                if (exp == 2.0 || exp == 0.5)
                {
                    // Vectorize inputs and gradients
                    var avVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Data.AsSpan());
                    var agVecs = MemoryMarshal.Cast<double, Vector<double>>(a.Grad.AsSpan());
                    var vexp = new Vector<double>(exp); // splat scalar into vector
                    var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                    // Calculate vectorized gradients of parent
                    for (int i = 0; i < agVecs.Length; i++)
                    {
                        agVecs[i] += exp == 2.0 ? vexp * avVecs[i] * rgVecs[i] : (vexp / Vector.SquareRoot(avVecs[i])) * rgVecs[i];
                    }

                    // Clean up unvectorized tail
                    for (int i = agVecs.Length * VectorSize; i < a.ElementCount; i++)
                    {
                        a.Grad[i] += exp * Math.Pow(a[i], exp - 1.0) * result.Grad[i];
                    }
                }
                else // calculate gradients of parent sequentially - no general exponentiation vectorization available
                {
                    for (int i = 0; i < a.ElementCount; i++)
                    {
                        a.Grad[i] += exp * Math.Pow(a[i], exp - 1.0) * result.Grad[i];
                    }
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Raises a scalar to the power of every element in a tensor.
    /// </summary>
    /// <param name="a">Scalar to exponentiate.</param>
    /// <param name="exp">Tensor to raise to the power of.</param>
    /// <returns>Tensor containing the power of the input scalar and every element in the input tensor.</returns>
    public static Tensor Pow(double a, Tensor exp)
    {
        Tensor result = GetResultTensor(exp, exp.Dimensions, exp.RequiresGrad);

        // Calculate power sequentially - no general exponentiation vectorization available
        for (int i = 0; i < result.ElementCount; i++)
        {
            result[i] = Math.Pow(a, exp[i]);
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(exp);

            // Gradient calculation function for r = a^b -> dr/da = b * a^(b - 1); dr/db = a^b * ln(a)
            result._backward = () =>
            {
                if (!exp.RequiresGrad) return;

                // Vectorize inputs, results, and gradients - possible due to derivative only involving a^b (already calculated) and ln(a) (scalar)
                var expgVecs = MemoryMarshal.Cast<double, Vector<double>>(exp.Grad.AsSpan());
                double lna = Math.Log(a); // precalculate ln(a)
                var vlna = new Vector<double>(lna); // splat ln(a) into vector
                var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parent
                for (int i = 0; i < expgVecs.Length; i++)
                {
                    expgVecs[i] += rvVecs[i] * vlna * rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = expgVecs.Length * VectorSize; i < exp.ElementCount; i++)
                {
                    exp.Grad[i] += result[i] * lna * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Raises e to the power of every element in the tensor.
    /// </summary>
    /// <param name="t">Tensor to raise e to the power of.</param>
    /// <returns>Tensor containing the power of e and every element in the input tensor.</returns>
    public static Tensor Exp(Tensor t)
    {
        Tensor result = GetResultTensor(t, t.Dimensions, t.RequiresGrad);

        // Vectorize inputs and results
        var tVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Data.AsSpan());
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized powers of e
        for (int i = 0; i < tVecs.Length; i++)
        {
            rVecs[i] = Vector.Exp(tVecs[i]);
        }

        // Clean up unvectorized tail
        for (int i = tVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = Math.Exp(t[i]);
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(t);

            // Gradient calculation function for r = e^t -> dr/dt = e^t
            result._backward = () =>
            {
                if (!t.RequiresGrad) return;

                // Vectorize results and gradients
                var tgVecs = MemoryMarshal.Cast<double, Vector<double>>(t.Grad.AsSpan());
                var rvVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parent
                for (int i = 0; i < rgVecs.Length; i++)
                {
                    tgVecs[i] += rvVecs[i] * rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    t.Grad[i] += result[i] * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Takes the element-wise logarithm of two tensors.
    /// </summary>
    /// <param name="logBase">Tensor to use as logarithm base.</param>
    /// <param name="arg">Tensor to take logarithm of.</param>
    /// <returns>Tensor containing the element-wise logarithm of the two input tensors.</returns>
    public static Tensor Log(Tensor logBase, Tensor arg)
    {
        var owner = logBase.RequiresGrad ? logBase : arg;
        Tensor result = GetResultTensor(owner, owner.Dimensions, logBase.RequiresGrad || arg.RequiresGrad);

        // Vectorize inputs and results
        var lbVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Data.AsSpan());
        var argVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Data.AsSpan());
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized logarithms - using log base change formula -> log_lb(arg) = ln(arg) / ln(lb)
        for (int i = 0; i < lbVecs.Length; i++)
        {
            rVecs[i] = Vector.Log(argVecs[i]) / Vector.Log(lbVecs[i]);
        }

        // Clean up unvectorized tail
        for (int i = lbVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = Math.Log(arg[i], logBase[i]);
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.AddRange([logBase, arg]);

            // Gradient calculation function for r = log_lb(arg) -> dr/dlb = -ln(arg) / (lb * ln^2(lb)); dr/darg = 1 / (arg * ln(lb))
            result._backward = () =>
            {
                if (!logBase.RequiresGrad && !arg.RequiresGrad) return;

                // Vectorize inputs and gradients
                var lbvVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Data.AsSpan());
                var lbgVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Grad.AsSpan());
                var argvVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Data.AsSpan());
                var arggVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Grad.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parents
                for (int i = 0; i < rgVecs.Length; i++)
                {
                    var lnb = Vector.Log(lbvVecs[i]);
                    if (logBase.RequiresGrad)
                    {
                        lbgVecs[i] -= (Vector.Log(argvVecs[i]) / (lbvVecs[i] * lnb * lnb)) * rgVecs[i];
                    }
                    if (arg.RequiresGrad) arggVecs[i] += rgVecs[i] / (argvVecs[i] * lnb);
                }

                // Clean up unvectorized tail
                for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    double lnb = Math.Log(logBase[i]);
                    if (logBase.RequiresGrad) logBase.Grad[i] -= (Math.Log(arg[i]) / (logBase[i] * (lnb * lnb))) * result.Grad[i];
                    if (arg.RequiresGrad) arg.Grad[i] += (1.0 / (arg[i] * lnb)) * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Takes the logarithm of a scalar with the base of every element in a tensor.
    /// </summary>
    /// <param name="logBase">Tensor to use as logarithm base.</param>
    /// <param name="arg">Scalar to take logarithm of.</param>
    /// <returns>Tensor containing the logarithm of the input scalar with the base of every element in the input tensor.</returns>
    public static Tensor Log(Tensor logBase, double arg)
    {
        Tensor result = GetResultTensor(logBase, logBase.Dimensions, logBase.RequiresGrad);

        // Vectorize inputs and results
        var lbVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Data.AsSpan());
        var lnvarg = Vector.Log(new Vector<double>(arg));
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized logarithm
        for (int i = 0; i < lbVecs.Length; i++)
        {
            rVecs[i] = lnvarg / Vector.Log(lbVecs[i]);
        }

        // Clean up unvectorized tail
        for (int i = lbVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = Math.Log(arg, logBase[i]);
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(logBase);

            // Gradient calculation function for r = log_lb(arg) -> dr/dlb = -ln(arg) / (lb * ln^2(lb)); dr/darg = 1 / (arg * ln(lb))
            result._backward = () =>
            {
                if (!logBase.RequiresGrad) return;

                // Vectorize inputs and gradients
                var lbvVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Data.AsSpan());
                var lbgVecs = MemoryMarshal.Cast<double, Vector<double>>(logBase.Grad.AsSpan());
                double lnarg = Math.Log(arg); // precalculate ln(arg)
                var lnvarg = new Vector<double>(lnarg); // splat scalar into vector
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parent
                for (int i = 0; i < rgVecs.Length; i++)
                {
                    var lnb = Vector.Log(lbvVecs[i]);
                    lbgVecs[i] -= (lnvarg / (lbvVecs[i] * lnb * lnb)) * rgVecs[i];
                }

                // Clean up unvectorized tail
                for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    double lnb = Math.Log(logBase[i]);
                    logBase.Grad[i] -= (lnarg / (logBase[i] * (lnb * lnb))) * result.Grad[i];
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Takes the element-wise logarithm with a scalar base of a tensor.
    /// </summary>
    /// <param name="logBase">Scalar to use as logarithm base.</param>
    /// <param name="arg">Tensor to take logarithm of.</param>
    /// <returns>Tensor containing the element-wise logarithm with the input scalar base of the input tensor.</returns>
    public static Tensor Log(double logBase, Tensor arg)
    {
        Tensor result = GetResultTensor(arg, arg.Dimensions, arg.RequiresGrad);

        // Vectorize inputs and results
        var lnvlb = Vector.Log(new Vector<double>(logBase)); // splat scalar into vector
        var argVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Data.AsSpan());
        var rVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Data.AsSpan());

        // Calculate vectorized logarithms
        for (int i = 0; i < argVecs.Length; i++)
        {
            rVecs[i] = Vector.Log(argVecs[i]) / lnvlb;
        }

        // Clean up unvectorized tail
        for (int i = argVecs.Length * VectorSize; i < result.ElementCount; i++)
        {
            result[i] = Math.Log(arg[i], logBase);
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.Add(arg);

            // Gradient calculation function for r = log_lb(arg) -> dr/dlb = -ln(arg) / (lb * ln^2(lb)); dr/darg = 1 / (arg * ln(lb))
            result._backward = () =>
            {
                if (!arg.RequiresGrad) return;

                // Vectorize inputs and gradients
                double lnb = Math.Log(logBase); // precalculate ln(lb)
                var lnvlb = new Vector<double>(lnb); // splat scalar into vector
                var argvVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Data.AsSpan());
                var arggVecs = MemoryMarshal.Cast<double, Vector<double>>(arg.Grad.AsSpan());
                var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan());

                // Calculate vectorized gradients for parent
                for (int i = 0; i < rgVecs.Length; i++)
                {
                    arggVecs[i] += rgVecs[i] / (argvVecs[i] * lnvlb);
                }

                // Clean up unvectorized tail
                for (int i = rgVecs.Length * VectorSize; i < result.ElementCount; i++)
                {
                    arg.Grad[i] += (1.0 / (arg[i] * lnb)) * result.Grad[i];
                }
            };
        }

        return result;
    }

    // Matrix multiplication + convolution

    /// <summary>
    /// Multiplies two tensors using 2D matrix multiplication.
    /// </summary>
    /// <param name="a">First tensor to multiply.</param>
    /// <param name="b">Second tensor to multiply.</param>
    /// <returns>Tensor containing the matrix multiplication product of the two input tensors.</returns>
    public static Tensor operator ^(Tensor a, Tensor b)
    {
        // Extract "outer" and "inner" 2D matrix dimensions - preceding dimensions represent "batch" dimensions
        int rank = a.Rank;
        int m = a.Dimensions[^2]; // "outer" dimension of A
        int n = a.Dimensions[^1]; // shared "inner" dimension
        int p = b.Dimensions[^1]; // "outer" dimension of B

        bool bBatched = b.Rank == a.Rank; // whether B contains a separate 2D matrix per batch of A

        // Compute number of batches and linear sizes of inputs and outputs
        int batchSize = 1;
        for (int i = 0; i < rank - 2; i++) batchSize *= a.Dimensions[i];
        int aMatSize = m * n;
        int bMatSize = n * p;
        int rMatSize = m * p;

        int totalRows = batchSize * m;

        // Build result dimensions -> batch dimensions + outer 2D matrix dimensions
        var resultDims = (int[])a.Dimensions.Clone();
        resultDims[^1] = p;

        var owner = a.RequiresGrad ? a : b;
        Tensor result = GetResultTensor(owner, resultDims, a.RequiresGrad || b.RequiresGrad);

        // Calculate matrix multiplication product
        bool useParallel = (long)totalRows * n * p > ParallelThreshold; // whether inputs are large enough to warrant parallelizing (multithreading)
        double[] bT = ArrayPool<double>.Shared.Rent(bMatSize * batchSize); // Rent buffer for transposition of B - avoid garbage collector
        try
        {
            // Transpose B - columns -> rows - dot product calculated along contiguous rows of A and contiguous rows of B - reduces cache misses
            for (int batch = 0; batch < batchSize; batch++)
            {
                int bSrcOff = bBatched ? batch * bMatSize : 0; // duplicates B's data if same 2D matrix should be used for each batch
                TransposeMatrix(b.Data, bT, bSrcOff, batch * bMatSize, n, p);
            }

            // Parallelize if inputs large enough
            if (useParallel)
            {
                Parallel.For(0, batchSize * m, row =>
                {
                    int batch = row / m;
                    int i = row % m;
                    ComputeRow(i, n, p, a.Data, bT, result.Data, batch * aMatSize, batch * bMatSize, batch * rMatSize);
                });
            }
            else // calculate product sequentially
            {
                for (int batch = 0; batch < batchSize; batch++)
                {
                    int aOff = batch * aMatSize;
                    int bTOff = batch * bMatSize;
                    int rOff = batch * rMatSize;

                    for (int i = 0; i < m; i++)
                    {
                        ComputeRow(i, n, p, a.Data, bT, result.Data, aOff, bTOff, rOff);
                    }
                }
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(bT); // release B transposition buffer
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.AddRange([a, b]);

            // Gradient calculation function for R = A ^ B (2D matmul) -> grad_A = grad_C ^ B^T; grad_B = A^T ^ grad_C
            result._backward = () =>
            {
                if (!a.RequiresGrad && !b.RequiresGrad) return;

                for (int batch = 0; batch < batchSize; batch++)
                {
                    int aOff = batch * aMatSize;
                    int bOff = bBatched ? batch * bMatSize : 0; // ensure B gradients accumulate correctly if B was broadcasted during forward pass
                    int rOff = batch * rMatSize;
                    bool par = (long)m * n * p > ParallelThreshold;

                    if (a.RequiresGrad && b.RequiresGrad) // branch if both parent gradients needed
                    {
                        // Rent buffers for transpositions - avoid garbage collector
                        double[] aT = ArrayPool<double>.Shared.Rent(aMatSize);
                        double[] dOutT = ArrayPool<double>.Shared.Rent(rMatSize);
                        try
                        {
                            TransposeMatrix(a.Data, aT, aOff, 0, m, n);
                            TransposeMatrix(result.Grad, dOutT, rOff, 0, m, p); // transpose grad_C for contiguous memory access

                            // Parallelize if inputs large enough
                            if (par)
                            {
                                // Calculate grad_A in parallel
                                Parallel.For(0, m, i =>
                                {
                                    for (int k = 0; k < n; k++)
                                    {
                                        // Use untransposed B for contiguous memory access - reduce cache misses
                                        a.Grad[aOff + i * n + k] += DotProduct(result.Grad, b.Data, rOff + i * p, bOff + k * p, p);
                                    }
                                });

                                // Calculate grad_B in parallel
                                Parallel.For(0, n, k =>
                                {
                                    for (int j = 0; j < p; j++)
                                    {
                                        // Use transposed grad_C (dOutT) for contiguous memory access - reduce cache misses
                                        b.Grad[bOff + k * p + j] += DotProduct(aT, dOutT, k * m, j * m, m);
                                    }
                                });
                            }
                            else // calculate gradients sequentially
                            {
                                // Calculate grad_A sequentially
                                for (int i = 0; i < m; i++)
                                {
                                    for (int k = 0; k < n; k++)
                                    {
                                        // Use untransposed B for contiguous memory access - reduce cache misses
                                        a.Grad[aOff + i * n + k] += DotProduct(result.Grad, b.Data, rOff + i * p, bOff + k * p, p);
                                    }
                                }

                                // Calculate grad_B sequentially
                                for (int k = 0; k < n; k++)
                                {
                                    for (int j = 0; j < p; j++)
                                    {
                                        // Use transposed grad_C (dOutT) for contiguous memory access - reduce cache misses
                                        b.Grad[bOff + k * p + j] += DotProduct(aT, dOutT, k * m, j * m, m);
                                    }
                                }
                            }
                        }
                        finally
                        {
                            // Release transposition buffers
                            ArrayPool<double>.Shared.Return(aT);
                            ArrayPool<double>.Shared.Return(dOutT);
                        }
                    }
                    else if (a.RequiresGrad) // branch if only grad_A is needed
                    {
                        // Parallelize if inputs large enough
                        if (par)
                        {
                            Parallel.For(0, m, i =>
                            {
                                for (int k = 0; k < n; k++)
                                {
                                    // Use untransposed B for contiguous memory access - reduce cache misses
                                    a.Grad[aOff + i * n + k] += DotProduct(result.Grad, b.Data, rOff + i * p, bOff + k * p, p);
                                }
                            });
                        }
                        else // calculate gradients sequentially
                        {
                            for (int i = 0; i < m; i++)
                            {
                                for (int k = 0; k < n; k++)
                                {
                                    // Use untransposed B for contiguous memory access - reduce cache misses
                                    a.Grad[aOff + i * n + k] += DotProduct(result.Grad, b.Data, rOff + i * p, bOff + k * p, p);
                                }
                            }
                        }
                    }
                    else if (b.RequiresGrad) // branch if only grad_B is needed
                    {
                        // Rent buffers for transpositions - avoid garbage collector
                        double[] aT = ArrayPool<double>.Shared.Rent(aMatSize);
                        double[] dOutT = ArrayPool<double>.Shared.Rent(rMatSize);
                        try
                        {
                            TransposeMatrix(a.Data, aT, aOff, 0, m, n);
                            TransposeMatrix(result.Grad, dOutT, rOff, 0, m, p); // transpose grad_C for contiguous memory access

                            // Parallelize if inputs large enough
                            if (par)
                            {
                                Parallel.For(0, n, k =>
                                {
                                    for (int j = 0; j < p; j++)
                                    {
                                        // Use transposed grad_C (dOutT) for contiguous memory access - reduce cache misses
                                        b.Grad[bOff + k * p + j] += DotProduct(aT, dOutT, k * m, j * m, m);
                                    }
                                });
                            }
                            else // calculate gradients sequentially
                            {
                                for (int k = 0; k < n; k++)
                                {
                                    for (int j = 0; j < p; j++)
                                    {
                                        // Use transposed grad_C (dOutT) for contiguous memory access - reduce cache misses
                                        b.Grad[bOff + k * p + j] += DotProduct(aT, dOutT, k * m, j * m, m);
                                    }
                                }
                            }
                        }
                        finally
                        {
                            // Release transposition buffers
                            ArrayPool<double>.Shared.Return(aT);
                            ArrayPool<double>.Shared.Return(dOutT);
                        }
                    }
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Transposes a 2D matrix, switching its rows and columns.
    /// </summary>
    /// <param name="src">Linear array buffer of matrix to transpose.</param>
    /// <param name="dst">Linear array buffer to write transposed matrix to.</param>
    /// <param name="srcOff">Offset of the first element to transpose in the source buffer.</param>
    /// <param name="dstOff">Offset of the first element to write to in the destination buffer.</param>
    /// <param name="rows">Number of rows of the source matrix.</param>
    /// <param name="cols">Number of columns of the source matrix.</param>
    static void TransposeMatrix(double[] src, double[] dst, int srcOff, int dstOff, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                dst[dstOff + c * rows + r] = src[srcOff + r * cols + c]; // switch row and column indices of data
            }
        }
    }

    /// <summary>
    /// Computes a single row of the result matrix of a 2D matrix multiplication.
    /// </summary>
    /// <param name="i">Index of the row to compute.</param>
    /// <param name="n">Inner dimension of the input matrices.</param>
    /// <param name="p">Outer dimension of the second input matrix.</param>
    /// <param name="a">Linear array buffer of the data of the first input matrix</param>
    /// <param name="bT">Linear array buffer of the data of the transposition of the second input matrix.</param>
    /// <param name="r">Linear array buffer of the result matrix to write to.</param>
    /// <param name="aOff">Linear offset of the current row in the first input matrix.</param>
    /// <param name="bTOff">Linear offset of the current row in the second input matrix.</param>
    /// <param name="rOff">Linear offset of the current row in the result matrix.</param>
    static void ComputeRow(int i, int n, int p, double[] a, double[] bT, double[] r, int aOff, int bTOff, int rOff)
    {
        for (int j = 0; j < p; j++) // iterate across row
        {
            // Calculate dot product at current position
            r[rOff + i * p + j] = DotProduct(a, bT, aOff + i * n, bTOff + j * n, n);
        }
    }

    /// <summary>
    /// Performs a convolution of a tensor.
    /// </summary>
    /// <param name="input">Tensor to convolve.</param>
    /// <param name="kernels">Kernels tensor to convolve with.</param>
    /// <param name="biases">Bias tensor to add.</param>
    /// <returns>Result tensor of the convolution of the input and kernels tensors with the bias tensor added.</returns>
    public static Tensor Convolve(Tensor input, Tensor kernels, Tensor biases)
    {
        // Extract and compute spatial data
        int batch = input.Dimensions[0];
        int spatialRank = kernels.Rank - 2;
        int filterCount = kernels.Dimensions[0];
        int inputChannels = kernels.Dimensions[^1];

        var outSpatialDims = new int[spatialRank];
        for (int i = 0; i < spatialRank; i++)
        {
            outSpatialDims[i] = input.Dimensions[i + 1] - kernels.Dimensions[i + 1] + 1;
        }

        int inSpatialSize = 1;
        for (int i = 1; i <= spatialRank; i++)
        {
            inSpatialSize *= input.Dimensions[i];
        }

        int outSpatialSize = 1;
        foreach (var dim in outSpatialDims)
        {
            outSpatialSize *= dim;
        }

        int kernelSpatialSize = 1;
        for (int i = 1; i <= spatialRank; i++)
        {
            kernelSpatialSize *= kernels.Dimensions[i];
        }

        int kernelVolumeSize = kernelSpatialSize * inputChannels;

        var inSpatialStrides = new int[spatialRank];
        inSpatialStrides[^1] = 1;
        for (int i = spatialRank - 2; i >= 0; i--)
        {
            inSpatialStrides[i] = inSpatialStrides[i + 1] * input.Dimensions[i + 2];
        }

        var outSpatialStrides = new int[spatialRank];
        outSpatialStrides[^1] = 1;
        for (int i = spatialRank - 2; i >= 0; i--)
        {
            outSpatialStrides[i] = outSpatialStrides[i + 1] * outSpatialDims[i + 1];
        }

        var kernelSpatialStrides = new int[spatialRank];
        kernelSpatialStrides[^1] = 1;
        for (int i = spatialRank - 2; i >= 0; i--)
        {
            kernelSpatialStrides[i] = kernelSpatialStrides[i + 1] * kernels.Dimensions[i + 2];
        }

        // Compute result dimensions
        var resultDims = new int[input.Rank];
        resultDims[0] = batch;
        Array.Copy(outSpatialDims, 0, resultDims, 1, outSpatialDims.Length);
        resultDims[^1] = filterCount;

        var owner = input.RequiresGrad ? input : (kernels.RequiresGrad ? kernels : biases);
        Tensor result = GetResultTensor(owner, resultDims, input.RequiresGrad || kernels.RequiresGrad || biases.RequiresGrad);

        // Calculate convolution result
        bool useParallel = (long)batch * outSpatialSize * filterCount * kernelVolumeSize > ParallelThreshold; // whether the inputs are large enough to warrant parallelizations
        if (useParallel) // calculate result in parallel
        {
            Parallel.For(0, batch * outSpatialSize, i => ComputeOutputPosition(i, spatialRank, outSpatialSize, filterCount,
                kernelSpatialSize, inputChannels, outSpatialStrides, kernelSpatialStrides, input.Strides, kernels.Strides, result.Strides,
                input.Data, kernels.Data, biases.Data, result.Data));
        }
        else // calculate result sequentially
        {
            for (int i = 0; i < batch * outSpatialSize; i++)
            {
                ComputeOutputPosition(i, spatialRank, outSpatialSize, filterCount, kernelSpatialSize, inputChannels, outSpatialStrides,
                    kernelSpatialStrides, input.Strides, kernels.Strides, result.Strides, input.Data, kernels.Data, biases.Data,
                    result.Data);
            }
        }

        // Connect result tensor to autograd graph if needed
        if (!Inference)
        {
            result._parents.AddRange([input, kernels, biases]);

            // Gradient calculation function for convolution
            result._backward = () =>
            {
                bool par = (long)batch * outSpatialSize * filterCount * kernelVolumeSize > ParallelThreshold; // whether the inputs are large enough to warrant parallelization

                // Calculate gradients for bias tensor
                if (biases.RequiresGrad)
                {
                    // No need to parallelize - only filterCount gradients to calculate

                    if (filterCount % VectorSize == 0) // number of filters is an exact multiple of vector size
                    {
                        // Can safely vectorize gradient accumulation

                        int vecsPerChunk = filterCount / VectorSize;
                        int numChunks = result.Grad.Length / filterCount;

                        // Preallocate vectors to accumulate gradients into per filter
                        Span<Vector<double>> accVecs = stackalloc Vector<double>[vecsPerChunk];

                        // Accumulate gradients across all result gradient chunks
                        for (int chunk = 0; chunk < numChunks; chunk++)
                        {
                            // Vectorize result gradients
                            var rgVecs = MemoryMarshal.Cast<double, Vector<double>>(result.Grad.AsSpan(chunk * filterCount, filterCount));

                            // Accumulate vectorized gradients for chunk
                            for (int v = 0; v < vecsPerChunk; v++)
                            {
                                accVecs[v] += rgVecs[v];
                            }
                        }

                        // Vectorize bias gradients
                        var bgVecs = MemoryMarshal.Cast<double, Vector<double>>(biases.Grad.AsSpan(0, filterCount));

                        // Add accumulated gradients to bias gradient
                        for (int v = 0; v < vecsPerChunk; v++)
                        {
                            bgVecs[v] += accVecs[v];
                        }
                    }
                    else // filterCount does not align with vector size - fall back to scalar-based sequential
                    {
                        // Accumulate result gradients into corresponding bias gradients
                        int f = 0;
                        for (int i = 0; i < result.Grad.Length; i++)
                        {
                            biases.Grad[f] += result.Grad[i];
                            if (++f == filterCount) f = 0;
                        }
                    }
                }

                // Calculate gradients for kernels tensor
                if (kernels.RequiresGrad)
                {
                    if (par) // calculate gradients in parallel
                    {
                        Parallel.For(0, filterCount * kernelSpatialSize, fkp => ComputeKernelGrad(fkp, spatialRank,
                            batch, outSpatialSize, filterCount, kernelSpatialSize, inputChannels, outSpatialStrides,
                            kernelSpatialStrides, input.Strides, kernels.Strides, result.Strides, input.Data, kernels.Grad,
                            result.Grad));
                    }
                    else // calculate gradients sequentially
                    {
                        for (int fkp = 0; fkp < filterCount * kernelSpatialSize; fkp++)
                        {
                            ComputeKernelGrad(fkp, spatialRank, batch, outSpatialSize, filterCount, kernelSpatialSize,
                                inputChannels, outSpatialStrides, kernelSpatialStrides, input.Strides, kernels.Strides,
                                result.Strides, input.Data, kernels.Grad, result.Grad);
                        }
                    }
                }

                // Calculate gradients for input tensor
                if (input.RequiresGrad)
                {
                    if (par) // calculate gradients in parallel
                    {
                        Parallel.For(0, batch * inSpatialSize, batchInPos => ComputeInputGrad(batchInPos, spatialRank,
                            inSpatialSize, filterCount, kernelSpatialSize, inputChannels, inSpatialStrides, kernelSpatialStrides,
                            outSpatialDims, outSpatialStrides, input.Strides, kernels.Strides, result.Strides, input.Grad, kernels.Data,
                            result.Grad));
                    }
                    else // calculate gradients sequentially
                    {
                        for (int batchInPos = 0; batchInPos < batch * inSpatialSize; batchInPos++)
                        {
                            ComputeInputGrad(batchInPos, spatialRank, inSpatialSize, filterCount, kernelSpatialSize, inputChannels,
                                inSpatialStrides, kernelSpatialStrides, outSpatialDims, outSpatialStrides, input.Strides, kernels.Strides,
                                result.Strides, input.Grad, kernels.Data, result.Grad);
                        }
                    }
                }
            };
        }

        return result;
    }

    /// <summary>
    /// Calculates all filter outputs at a single output spatial position of a tensor convolution.
    /// </summary>
    /// <param name="batchOutPos">Spatial position in the result tensor to calculate.</param>
    /// <param name="spatialRank">Spatial rank of the convolution.</param>
    /// <param name="outSpatialSize">Spatial size of the result tensor.</param>
    /// <param name="filterCount">Number of filters used in the convolution.</param>
    /// <param name="kernelSpatialSize">Spatial size of the kernels tensor.</param>
    /// <param name="inputChannels">Number of input channels in the convolution.</param>
    /// <param name="outSpatialStrides">Spatial strides of the result tensor.</param>
    /// <param name="kernelSpatialStrides">Spatial strides of the kernels tensor.</param>
    /// <param name="inputStrides">Dimensional strides of the input tensor.</param>
    /// <param name="kernelStrides">Dimensional strides of the kernels tensor.</param>
    /// <param name="resultStrides">Dimensional strides of the result tensor.</param>
    /// <param name="inputData">Data array of the input tensor.</param>
    /// <param name="kernelData">Data array of the kernels tensor.</param>
    /// <param name="biasData">Data array of the bias tensor.</param>
    /// <param name="resultData">Data array of the result tensor.</param>
    static void ComputeOutputPosition(int batchOutPos, int spatialRank, int outSpatialSize, int filterCount, int kernelSpatialSize,
        int inputChannels, int[] outSpatialStrides, int[] kernelSpatialStrides, int[] inputStrides, int[] kernelStrides, int[] resultStrides,
        double[] inputData, double[] kernelData, double[] biasData, double[] resultData)
    {
        int b = batchOutPos / outSpatialSize;

        // Compute spatial coordinates
        Span<int> outCoords = stackalloc int[spatialRank];
        int rem = batchOutPos % outSpatialSize;
        for (int i = 0; i < spatialRank; i++)
        {
            outCoords[i] = rem / outSpatialStrides[i];
            rem %= outSpatialStrides[i];
        }

        // Preallocate arrays on the stack - avoid garbage collector
        Span<int> kernelCoords = stackalloc int[spatialRank];
        Span<double> sums = stackalloc double[filterCount];
        biasData.AsSpan(0, filterCount).CopyTo(sums); // add bias tensor to all sum results

        // Precompute constant components of offsets
        int inputOffsetBase = b * inputStrides[0];
        int kernelOffsetBaseCoeff = kernelSpatialSize * inputChannels;
        int kernelOffsetBase = 0;
        for (int i = 0; i < spatialRank; i++)
        {
            kernelOffsetBase += kernelCoords[i] * kernelStrides[i + 1];
        }

        // Calculate filter values for each kernel spatial position
        for (int kp = 0; kp < kernelSpatialSize; kp++)
        {
            // Compute spatial coordinates and linear offsets
            rem = kp;
            for (int i = 0; i < spatialRank; i++)
            {
                kernelCoords[i] = rem / kernelSpatialStrides[i];
                rem %= kernelSpatialStrides[i];
            }

            int inputOffset = inputOffsetBase;
            for (int i = 0; i < spatialRank; i++)
            {
                inputOffset += (outCoords[i] + kernelCoords[i]) * inputStrides[i + 1];
            }

            // Calculate value for each filter at the current position
            for (int f = 0; f < filterCount; f++)
            {
                // Compute kernel spatial offset
                int kernelOffset = kernelOffsetBaseCoeff * f + kernelOffsetBase;

                // Add dot product of current kernel position to the sums of the current filter
                sums[f] += DotProduct(inputData, kernelData, inputOffset, kernelOffset, inputChannels);
            }
        }

        // Compute linear offset of the result
        int resultOffset = b * resultStrides[0];
        for (int i = 0; i < spatialRank; i++)
        {
            resultOffset += outCoords[i] * resultStrides[i + 1];
        }

        // Fill result data array with calculated sums
        sums.CopyTo(resultData.AsSpan(resultOffset, filterCount));
    }

    /// <summary>
    /// Calculates the gradient of the kernels tensor used in a convolution.
    /// </summary>
    /// <param name="fkp">Linear index of the kernel position and filter to use.</param>
    /// <param name="spatialRank">Spatial rank of the convolution.</param>
    /// <param name="batch">Size of the batch used in the convolution.</param>
    /// <param name="outSpatialSize">Spatial size of the result tensor.</param>
    /// <param name="filterCount">Number of filters used in the convolution.</param>
    /// <param name="kernelSpatialSize">Spatial size of the kernels tensor.</param>
    /// <param name="inputChannels">Number of input channels used in the convolution.</param>
    /// <param name="outSpatialStrides">Spatial strides of the result tensor.</param>
    /// <param name="kernelSpatialStrides">Spatial strides of the kernels tensor.</param>
    /// <param name="inputStrides">Dimensional strides of the input tensor.</param>
    /// <param name="kernelStrides">Dimensional strides of the kernels tensor.</param>
    /// <param name="resultStrides">Dimensional strides of the result tensor.</param>
    /// <param name="inputData">Data array of the input tensor.</param>
    /// <param name="kernelGrad">Gradient array of the kernels tensor.</param>
    /// <param name="resultGrad">Gradient array of the result tensor.</param>
    static void ComputeKernelGrad(int fkp, int spatialRank, int batch, int outSpatialSize, int filterCount,
        int kernelSpatialSize, int inputChannels, int[] outSpatialStrides, int[] kernelSpatialStrides,
        int[] inputStrides, int[] kernelStrides, int[] resultStrides, double[] inputData, double[] kernelGrad,
        double[] resultGrad)
    {
        // Compute filter index
        int f = fkp / kernelSpatialSize;

        // Compute kernels spatial coordinates and linear offset
        Span<int> kernelCoords = stackalloc int[spatialRank]; // allocate on stack - avoid garbage collector
        int rem = fkp % kernelSpatialSize;
        for (int i = 0; i < spatialRank; i++)
        {
            kernelCoords[i] = rem / kernelSpatialStrides[i];
            rem %= kernelSpatialStrides[i];
        }

        int kernelOffset = f * kernelSpatialSize * inputChannels;
        for (int i = 0; i < spatialRank; i++)
        {
            kernelOffset += kernelCoords[i] * kernelStrides[i + 1];
        }

        // Preallocate array on the stack - avoid garbage collector
        Span<int> outCoords = stackalloc int[spatialRank];

        // Calculate gradients at the given position in each batch
        for (int b = 0; b < batch; b++)
        {
            // Precompute constant components of offsets
            int inputOffsetBase = b * inputStrides[0];
            int resultOffsetBase = b * resultStrides[0] + f;

            // Calculate gradients for each output position
            for (int op = 0; op < outSpatialSize; op++)
            {
                // Compute spatial coordinates and offsets
                rem = op;
                for (int i = 0; i < spatialRank; i++)
                {
                    outCoords[i] = rem / outSpatialStrides[i];
                    rem %= outSpatialStrides[i];
                }

                int inputOffset = inputOffsetBase;
                for (int i = 0; i < spatialRank; i++)
                {
                    inputOffset += (outCoords[i] + kernelCoords[i]) * inputStrides[i + 1];
                }

                int resultOffset = resultOffsetBase;
                for (int i = 0; i < spatialRank; i++)
                {
                    resultOffset += outCoords[i] * resultStrides[i + 1];
                }
                double dOut = resultGrad[resultOffset];

                // Vectorize gradients and inputs
                var kgVecs = MemoryMarshal.Cast<double, Vector<double>>(kernelGrad.AsSpan(kernelOffset, inputChannels));
                var inVecs = MemoryMarshal.Cast<double, Vector<double>>(inputData.AsSpan(inputOffset, inputChannels));
                var vdOut = new Vector<double>(dOut); // splat scalar into vector

                // Calculate vectorized gradients
                for (int i = 0; i < kgVecs.Length; i++)
                {
                    kgVecs[i] += inVecs[i] * vdOut;
                }

                // Clean up unvectorized tail
                for (int i = kgVecs.Length * VectorSize; i < inputChannels; i++)
                {
                    kernelGrad[kernelOffset + i] += inputData[inputOffset + i] * dOut;
                }
            }
        }
    }

    /// <summary>
    /// Calculates the gradient of the input tensor used in a convolution.
    /// </summary>
    /// <param name="batchInPos">Linear index of the batch and input position to use.</param>
    /// <param name="spatialRank">Spatial rank of the convolution.</param>
    /// <param name="inSpatialSize">Spatial size of the input tensor.</param>
    /// <param name="filterCount">Number of filters used in the convolution.</param>
    /// <param name="kernelSpatialSize">Spatial size of the kernels tensor.</param>
    /// <param name="inputChannels">Number of input channels used in the convolution.</param>
    /// <param name="inSpatialStrides">Spatial strides of the input tensor.</param>
    /// <param name="kernelSpatialStrides">Spatial strides of the kernels tensor.</param>
    /// <param name="outSpatialDims">Spatial dimensions of the result tensor.</param>
    /// <param name="outSpatialStrides">Spatial strides of the result tensor.</param>
    /// <param name="inputStrides">Dimensional strides of the input tensor.</param>
    /// <param name="kernelStrides">Dimensional strides of the kernels tensor.</param>
    /// <param name="resultStrides">Dimensional strides of the result tensor.</param>
    /// <param name="inputGrad">Gradient array of the input tensor.</param>
    /// <param name="kernelData">Data array of the kernels tensor.</param>
    /// <param name="resultGrad">Gradient array of the result tensor.</param>
    static void ComputeInputGrad(int batchInPos, int spatialRank, int inSpatialSize, int filterCount, int kernelSpatialSize,
        int inputChannels, int[] inSpatialStrides, int[] kernelSpatialStrides, int[] outSpatialDims, int[] outSpatialStrides,
        int[] inputStrides, int[] kernelStrides, int[] resultStrides, double[] inputGrad, double[] kernelData, double[] resultGrad)
    {
        // Compute batch index
        int b = batchInPos / inSpatialSize;

        // Compute input spatial coordinates and offsets
        Span<int> inCoords = stackalloc int[spatialRank];
        int rem = batchInPos % inSpatialSize;
        for (int i = 0; i < spatialRank; i++)
        {
            inCoords[i] = rem / inSpatialStrides[i];
            rem %= inSpatialStrides[i];
        }

        int inputOffset = b * inputStrides[0];
        for (int i = 0; i < spatialRank; i++)
        {
            inputOffset += inCoords[i] * inputStrides[i + 1];
        }

        // Preallocate array on stack - avoid garbage collector
        Span<int> kernelCoords = stackalloc int[spatialRank];

        // Precompute offset constant
        int resultOffsetBase = b * resultStrides[0];

        // Calculate gradients for each filter
        for (int f = 0; f < filterCount; f++)
        {
            // Precompute offset constant
            int kernelOffsetBase = f * kernelSpatialSize * inputChannels;

            // Calculate gradients at each kernel spatial position
            for (int kp = 0; kp < kernelSpatialSize; kp++)
            {
                // Compute kernel spatial coordinates
                rem = kp;
                for (int i = 0; i < spatialRank; i++)
                {
                    kernelCoords[i] = rem / kernelSpatialStrides[i];
                    rem %= kernelSpatialStrides[i];
                }

                // Ensure position is valid a valid output position
                bool valid = true;
                int resultOffset = resultOffsetBase + f;
                for (int i = 0; i < spatialRank; i++)
                {
                    int outCoord = inCoords[i] - kernelCoords[i];
                    if (outCoord < 0 || outCoord >= outSpatialDims[i])
                    {
                        valid = false;
                        break;
                    }
                    resultOffset += outCoord * resultStrides[i + 1];
                }
                if (!valid) continue; // skip if invalid

                double dOut = resultGrad[resultOffset];

                // Compute kernel linear offset
                int kernelOffset = kernelOffsetBase;
                for (int i = 0; i < spatialRank; i++)
                {
                    kernelOffset += kernelCoords[i] * kernelStrides[i + 1];
                }

                // Vectorize gradients and kernels
                var igVecs = MemoryMarshal.Cast<double, Vector<double>>(inputGrad.AsSpan(inputOffset, inputChannels));
                var kVecs = MemoryMarshal.Cast<double, Vector<double>>(kernelData.AsSpan(kernelOffset, inputChannels));
                var vdOut = new Vector<double>(dOut); // splat scalar into vector

                // Calculate vectorized gradients
                for (int i = 0; i < igVecs.Length; i++)
                {
                    igVecs[i] += kVecs[i] * vdOut;
                }

                // Clean up unvectorized tail
                for (int i = igVecs.Length * VectorSize; i < inputChannels; i++)
                {
                    inputGrad[inputOffset + i] += kernelData[kernelOffset + i] * dOut;
                }
            }
        }
    }

    /// <summary>
    /// Calculates the dot product of a subrange of two vectors.
    /// </summary>
    /// <param name="x">First input vector.</param>
    /// <param name="y">Second input vector.</param>
    /// <param name="xOff">Offset of the subrange in the first input vector.</param>
    /// <param name="yOff">Offset of the subrange in the second input vector.</param>
    /// <param name="len">Length of the subrange.</param>
    /// <returns>Dot product of the given subrange of the two input vectors.</returns>
    static double DotProduct(double[] x, double[] y, int xOff, int yOff, int len)
    {
        // Vectorize inputs
        var xVecs = MemoryMarshal.Cast<double, Vector<double>>(x.AsSpan(xOff, len));
        var yVecs = MemoryMarshal.Cast<double, Vector<double>>(y.AsSpan(yOff, len));

        var acc = Vector<double>.Zero; // product/sum vector

        // Calculate vectorized product/sum
        for (int i = 0; i < xVecs.Length; i++)
        {
            acc += xVecs[i] * yVecs[i];
        }
        double sum = Vector.Sum(acc);

        // Clean up unvectorized tail
        for (int i = xVecs.Length * VectorSize; i < len; i++)
        {
            sum += x[xOff + i] * y[yOff + i];
        }

        return sum;
    }
}
