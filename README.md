# VecMath

High-performance vector math for Delphi.

Modern CPUs can apply a single arithmetic instruction to several numbers at
once - a capability known as SIMD (Single Instruction, Multiple Data). One
instruction can add multiple pairs of doubles in the same time an ordinary
add handles one pair. A plain Delphi loop uses none of this: each iteration
processes one value, and the compiler does not vectorize loops on its own.
For code that works through large arrays of doubles, most of the machine
sits idle.

VecMath provides array-oriented math - element-wise arithmetic, `Exp`, `Log`,
sums and dot products over `TArray<Double>` - built on hand-written SIMD
routines. The best implementation for the actual CPU is selected automatically
at startup, and every operation has a pure-Pascal fallback.

Missing vectorization is the first of three bottlenecks that make
straightforwardly written array code 10x or more slower than the hardware
allows. VecMath addresses all three:

1. **One value at a time.** `System.Exp` and `System.Ln` compute a single
   value per call through a general-purpose routine. VecMath evaluates the
   same functions four doubles at a time, with the same accuracy.
2. **One trip through memory per operation.** Computing utilities, then
   exponentials, then a sum as separate steps drags a large array across the
   memory bus three times. VecMath offers fused operations that split the
   work into blocks small enough to stay in the CPU's fastest cache, and
   runs all steps on a block before moving to the next.
3. **Waiting for the previous result.** Even vectorized code runs at a
   fraction of peak speed if each instruction needs the previous one's
   result. VecMath's kernels keep several independent calculations in
   flight so the floating-point units never wait.

## The engine: TVectorMath

`TVectorMath` is the compute engine: element-wise operations (`Add`, `Subtract`,
`Multiply`, `Divide`, `Sqrt`, `Exp`, `Log`, `Sum`, ...) plus accumulation
primitives (dot product, scaled accumulate). At program start the global
`Engine` variable is initialised with the best implementation the CPU supports:

```pascal
uses VecMath;

Engine.Multiply(2.5, A, B);     // B[i] := 2.5 * A[i]
Engine.Exp(U, E, SumExp);       // E[i] := exp(U[i]),  SumExp = Sum E[i]  (fused)
```

The engine is **stateless and thread-safe**: any number of threads may share
the one instance. Operations that need scratch memory have overloads taking a
`TScratchBuffer` so repeated calls reuse allocations - pass one scratch buffer
per thread.

## Platforms and the pure-Pascal fallback

The principle: **every operation has a pure-Pascal implementation; assembly is
an override, never a requirement.** Selection happens at two levels:

- **Compile time** decides which implementations exist in the binary. The
  assembly units are guarded on `MSWINDOWS AND CPUX64`; on every other target
  the SIMD classes compile to empty shells that inherit the pure-Pascal base
  class.
- **Run time** (Windows 64-bit only) picks the best compiled implementation
  via CPUID: AVX2+FMA where available, otherwise AVX, otherwise SSE2 -
  including a check that the OS actually saves the YMM registers.

| Compile target       | Implementation                                  |
| -------------------- | ----------------------------------------------- |
| Windows 64-bit       | AVX2+FMA / AVX / SSE2, selected by CPUID        |
| Windows 32-bit       | Pure Pascal                                     |
| Linux, macOS, mobile | Pure Pascal                                     |

Note that the *compile target* decides, not the machine the program runs on:
a Win32 program running on a 64-bit OS with an AVX2 CPU still uses pure
Pascal, because the x64 assembly (Win64 calling convention, 16 vector
registers) was never compiled into it. If you need the performance, build the
host application as Win64.

Results are identical across implementations up to floating-point rounding
order, and the same unit tests run against every implementation the CPU
supports. The Pascal fallbacks are deliberately written as unrolled loops, so
"fallback" still means reasonable scalar code rather than a naive reference
implementation.

## Accuracy

- `Exp`: degree-12 polynomial after range reduction, ~1-2 ulp for |x| < 709
  (the full finite-result range of double). No special handling of NaN/+/-Inf
  inputs.
- `Log`: polynomial after mantissa/exponent split, ~1 ulp for positive finite
  inputs. Zero or negative inputs produce garbage rather than +/-Inf/NaN - the
  caller is responsible for keeping inputs positive.
- Sums use multiple independent accumulators, which changes the summation
  order versus a naive running sum (and is typically slightly *more*
  accurate).

## Layout

| Path                        | Contents                                         |
| --------------------------- | ------------------------------------------------ |
| `Source\VecMath.pas`        | `TVectorMath` base class, pure-Pascal implementations, engine selection |
| `Source\VecMath.CPU.pas`    | CPUID feature detection (SSE2/AVX/AVX2+OS support) |
| `Source\VecMath.SSE2.pas`   | SSE2 assembly overrides                          |
| `Source\VecMath.AVX.pas`    | AVX and AVX2+FMA assembly overrides              |
| `Tests\TestVecMath.dpr`     | DUnitX test suite - accuracy vs `System.Exp`/`Ln`, cross-engine consistency |
| `Benchmark\Benchmark.dpr`   | Benchmark suite - measures scalar/SSE2/AVX/AVX2 side by side on your hardware |

## License

MIT - see the LICENSE file in the repository root.
