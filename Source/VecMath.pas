unit VecMath;

////////////////////////////////////////////////////////////////////////////////
//
// Author: Jaap Baak (initial implementation developed with Claude Fable 5)
// https://github.com/transportmodelling/VecMath
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
interface
////////////////////////////////////////////////////////////////////////////////

Type
  TScratchBuffer = record
  private
    FDouble: array[0..3] of TArray<Double>;
    FInt64:  TArray<Int64>;
    Procedure EnsureSize(Len: Integer);
  end;

  // The first and second moment of X weighted by W*exp(Scale*X), see ExpMoments
  TExpMoments = record
    First,Second: Double;
    Procedure Add(const Moments: TExpMoments);
  end;

  TVectorMath = class abstract
  strict protected
    // Range reduction constants of the scalar code. ln2 = LN2HI + LN2LO, where
    // LN2HI has trailing mantissa bits zero so that n*LN2HI is exact (fdlibm).
    // The literals round to the intended doubles, so the warning on that
    // rounding is off.
    {$WARN LOST_EXTENDED_PRECISION OFF}
    const
      LOG2E = 1.4426950408889634;
      LN2HI = 6.93147180369123816490e-1;
      LN2LO = 1.90821492927058770002e-10;
      LN2   = 6.93147180559945309417e-1;
      SQRT2 = 1.41421356237309504880e0;
    {$WARN LOST_EXTENDED_PRECISION ON}
    // ExpCore is accurate for |x| <= EXP_LIMIT. Beyond it, 2^n leaves the
    // exponent range of a double and the reconstruction yields garbage.
    const
      EXP_LIMIT = 708.0;
  strict private
    Procedure ExpBlock(Count: Integer; V, Res: PDouble; var Scratch: TScratchBuffer);
  strict protected
    // The scalar steps of exp and ln for one element, which every scalar path
    // and the scalar tails of the SIMD subclasses share, so they round alike:
    // exp(x) = 2^n * p(r) with x = n*ln2 + r, and ln(x) = n*ln2 + s*g(s^2)
    // with x = 2^n * m and s = (m-1)/(m+1).
    Function ExpReduction(const X: Double; out N: Int64): Double; inline;
    Function ExpPolynomial(const R: Double): Double; inline;
    Function PowerOfTwo(const N: Int64): Double; inline;
    Function LogReduction(const X: Double; out NLn2: Double): Double; inline;
    Function LogPolynomial(const S: Double): Double; inline;
    // Hardware-dispatch hooks - overridden by SIMD subclasses.
    Procedure ExpReduce (QuadCount: Integer; V, R: PDouble; N: PInt64); virtual;
    Procedure ExpRecon  (QuadCount: Integer; P: PDouble; N: PInt64; Res: PDouble); virtual;
    Procedure LogReduce (QuadCount: Integer; V, S, NLn2: PDouble); virtual;
    Procedure ExpHorner(Count: Integer; R, P: PDouble); virtual;
    Procedure LogHorner(Count: Integer; S, NLn2, Res: PDouble); virtual;
    // Full exp pipeline for one block: Res[i] := exp(V[i]) for i < Count.
    // Base implementation runs reduce -> Horner -> reconstruct through the
    // virtuals above, using Scratch.FDouble[1,2] and FInt64 as workspace.
    // TVectorMathAVX2 overrides with a fused single-pass kernel that keeps
    // all intermediates in registers and does not touch scratch memory at all.
    Procedure ExpCore(Count: Integer; V, Res: PDouble;
      var Scratch: TScratchBuffer); virtual;
    // Whether every argument of a block lies within +-EXP_LIMIT; a NaN fails.
    // TVectorMathAVX overrides with a SIMD scan.
    Function ExpInRange(Count: Integer; V: PDouble): Boolean; virtual;
    // Res[i] := Scale*X[i] where Mask[i] <> 0, else 0. X is masked before the
    // multiplication, so any value of X where Mask[i] = 0 is harmless.
    Procedure MaskedScaleBlock(const Scale: Double; Count: Integer; X, Mask, Res: PDouble); virtual;
    // The moments of ExpMoments for one block, with E[i] = exp(Scale*X[i])
    Function ExpMomentsBlock(Count: Integer; X, W, E: PDouble): TExpMoments; virtual;
    // Full ln pipeline for one block: Res[i] := ln(X[i]) for i < Count.
    // Base implementation runs reduce -> Horner through the virtuals above;
    // S and NLn2 are caller-provided workspace of at least Count elements.
    // TVectorMathAVX2 overrides with a fused single-pass kernel that keeps
    // all intermediates in registers and ignores the workspace.
    Procedure LogCore(Count: Integer; X, Res, S, NLn2: PDouble); virtual;
  public
    // Element-wise math operations.
    Procedure Add     (const Scalar: Double; const Vector, Result: TArray<Double>); overload; virtual;
    Procedure Add     (const Left, Right, Result: TArray<Double>);                  overload; virtual;
    Procedure Subtract(const Scalar: Double; const Vector, Result: TArray<Double>); overload; virtual;
    Procedure Subtract(const Left, Right, Result: TArray<Double>);                  overload; virtual;
    Procedure Multiply(const Scalar: Double; const Vector, Result: TArray<Double>); overload; virtual;
    Procedure Multiply(const Left, Right, Result: TArray<Double>);                  overload; virtual;
    Procedure Divide  (const Scalar: Double; const Vector, Result: TArray<Double>); overload; virtual;
    Procedure Divide  (const Left, Right, Result: TArray<Double>);                  overload; virtual;
    Procedure Sqrt    (const Vector, Result: TArray<Double>); virtual;
    function  Sum     (const Vector: TArray<Double>): Double; virtual;
    Procedure Exp     (const Vector, Result: TArray<Double>); overload; virtual;
    Procedure Exp     (const Vector, Result: TArray<Double>; var Scratch: TScratchBuffer); overload; virtual;
    Procedure Exp     (const Vector, Result: TArray<Double>; out Sum: Double); overload; virtual;
    Procedure Exp     (const Vector, Result: TArray<Double>; out Sum: Double; var Scratch: TScratchBuffer); overload; virtual;
    Procedure Log     (const Vector, Result: TArray<Double>); overload; virtual;
    Procedure Log     (const Vector, Result: TArray<Double>; var Scratch: TScratchBuffer); overload; virtual;
    // Moments of X weighted by W*exp(Scale*X), over the elements with W[i] <> 0:
    //   First  = Sum W[i]*X[i]*exp(Scale*X[i])
    //   Second = Sum W[i]*X[i]^2*exp(Scale*X[i])
    // which are the function and derivative of a Newton-Raphson step on
    // Sum W[i]*X[i]*exp(Scale*X[i]) = Target. X and W have equal lengths; the X
    // of an element with W[i] = 0 may hold any value, such as a missing value code.
    Function ExpMoments(const Scale: Double; const X, W: TArray<Double>): TExpMoments; overload;
    Function ExpMoments(const Scale: Double; const X, W: TArray<Double>; var Scratch: TScratchBuffer): TExpMoments; overload;
    // InOut[i] := InOut[i]*exp(Scale*X[i]) where InOut[i] <> 0. X and InOut have
    // equal lengths; the X of an element with InOut[i] = 0 may hold any value.
    Procedure MultiplyByExp(const Scale: Double; const X, InOut: TArray<Double>); overload;
    Procedure MultiplyByExp(const Scale: Double; const X, InOut: TArray<Double>; var Scratch: TScratchBuffer); overload;
    // Accumulation primitives.
    function SumBlock(Data: PDouble; Count: Integer): Double; virtual;
    // Dot product: Sum Data[i]*Weights[i].  Weights may contain exact zeros;
    // 0*finite contributes exactly 0.
    function WeightedSumBlock(Data, Weights: PDouble; Count: Integer): Double; virtual;
    // Dest[i] += Scale * Src[i]  (SAXPY)
    Procedure ScaleAddBlock(Scale: Double; Count: Integer; Src, Dest: PDouble); virtual;
    // InOut[i] *= Src[i]
    Procedure MultiplyBlock(Count: Integer; Src, InOut: PDouble); virtual;
    // Dest[i] += Scale * Src[i].
    // Thin TArray wrapper over ScaleAddBlock - the virtual lives there.
    Procedure ScaleAdd(Scale: Double; Len: Integer;
      const Src, Dest: TArray<Double>);
    class function CreateOptimal: TVectorMath;
  end;

Var
  Engine: TVectorMath;

////////////////////////////////////////////////////////////////////////////////
implementation
////////////////////////////////////////////////////////////////////////////////

{$OPTIMIZATION ON}
{$POINTERMATH ON}

uses
  VecMath.CPU,
  VecMath.SSE2,
  VecMath.AVX;

// Block size for cache-blocked operations.
// Working arrays (V, R, P, NLn2, N) fit in L1 cache (typically 32 KB):
//   4 x Double arrays x 512 x 8 B = 16 KB  +  1 x Int64 array x 512 x 8 B = 4 KB
Const
  EXP_LOG_BLOCK = 512;

Type
  // Steps through an array of Length elements in blocks of EXP_LOG_BLOCK
  TBlocks = record
    Start,Count,Length: Integer;
    Constructor Create(ArrayLength: Integer);
    Function Next: Boolean;
  end;

Constructor TBlocks.Create(ArrayLength: Integer);
begin
  Start := 0;
  Count := 0;
  Length := ArrayLength;
end;

Function TBlocks.Next: Boolean;
begin
  Inc(Start, Count);
  Count := Length - Start;
  if Count > EXP_LOG_BLOCK then Count := EXP_LOG_BLOCK;
  Result := Count > 0;
end;

// ---------------------------------------------------------------------------
// TScratchBuffer
// ---------------------------------------------------------------------------

Procedure TScratchBuffer.EnsureSize(Len: Integer);
var I: Integer;
begin
  if Length(FInt64) < Len then SetLength(FInt64, Len);
  for I := 0 to 3 do
    if Length(FDouble[I]) < Len then SetLength(FDouble[I], Len);
end;

// ---------------------------------------------------------------------------
// TExpMoments
// ---------------------------------------------------------------------------

Procedure TExpMoments.Add(const Moments: TExpMoments);
begin
  First := First + Moments.First;
  Second := Second + Moments.Second;
end;

// ---------------------------------------------------------------------------
// TVectorMath - the scalar steps of exp and ln for one element
// ---------------------------------------------------------------------------

Function TVectorMath.ExpReduction(const X: Double; out N: Int64): Double;
// r = x - n*ln2 with n = round(x/ln2), so |r| <= ln2/2
begin
  N := System.Round(X * LOG2E);
  Result := (X - N * LN2HI) - N * LN2LO;
end;

Function TVectorMath.ExpPolynomial(const R: Double): Double;
// Degree-12 Taylor: truncation error r^13/13! ~ 1.7e-16 at |r| = ln2/2, so the
// polynomial itself contributes ~1 ulp.  (Degree 10 looked sufficient but its
// truncation error is ~2e-13 - caught by Exp_MatchesSystemExp.)
begin
  Result := R * (1.0/479001600.0) + (1.0/39916800.0);
  Result := R * Result + (1.0/3628800.0);
  Result := R * Result + (1.0/362880.0);
  Result := R * Result + (1.0/40320.0);
  Result := R * Result + (1.0/5040.0);
  Result := R * Result + (1.0/720.0);
  Result := R * Result + (1.0/120.0);
  Result := R * Result + (1.0/24.0);
  Result := R * Result + (1.0/6.0);
  Result := R * Result + 0.5;
  Result := R * Result + 1.0;
  Result := R * Result + 1.0;
end;

Function TVectorMath.PowerOfTwo(const N: Int64): Double;
// 2^n, written into the exponent field
begin
  PInt64(@Result)^ := (N + 1023) shl 52;
end;

Function TVectorMath.LogReduction(const X: Double; out NLn2: Double): Double;
// s = (m-1)/(m+1) and n*ln2 for x = 2^n * m, m in [sqrt(2)/2, sqrt(2))
var
  Bits: Int64;
  NI: Integer;
  M: Double;
begin
  Bits := PInt64(@X)^;
  NI   := Integer((Bits shr 52) and $7FF) - 1023;
  PInt64(@M)^ := (Bits and $000FFFFFFFFFFFFF) or $3FF0000000000000;
  if M >= SQRT2 then begin M := M * 0.5; Inc(NI); end;
  Result := (M - 1.0) / (M + 1.0);
  NLn2 := NI * LN2;
end;

Function TVectorMath.LogPolynomial(const S: Double): Double;
// s*g(s^2) with g(t) = 2 + 2t/3 + 2t^2/5 + ... + 2t^7/15
var T: Double;
begin
  T := S * S;
  Result := T * (2.0/15.0) + (2.0/13.0);
  Result := T * Result + (2.0/11.0);
  Result := T * Result + (2.0/9.0);
  Result := T * Result + (2.0/7.0);
  Result := T * Result + (2.0/5.0);
  Result := T * Result + (2.0/3.0);
  Result := T * Result + 2.0;
  Result := S * Result;
end;

// ---------------------------------------------------------------------------
// TVectorMath - scalar reduce/recon hooks (base-class defaults)
// ---------------------------------------------------------------------------

Procedure TVectorMath.ExpReduce(QuadCount: Integer; V, R: PDouble; N: PInt64);
var I: Integer;
begin
  for I := 0 to QuadCount * 4 - 1 do R[I] := ExpReduction(V[I], N[I]);
end;

Procedure TVectorMath.ExpRecon(QuadCount: Integer; P: PDouble; N: PInt64; Res: PDouble);
var I: Integer;
begin
  for I := 0 to QuadCount * 4 - 1 do Res[I] := P[I] * PowerOfTwo(N[I]);
end;

Procedure TVectorMath.LogReduce(QuadCount: Integer; V, S, NLn2: PDouble);
var I: Integer;
begin
  for I := 0 to QuadCount * 4 - 1 do S[I] := LogReduction(V[I], NLn2[I]);
end;

// ---------------------------------------------------------------------------
// TVectorMath - scalar element-wise defaults (subclasses override with SIMD)
// ---------------------------------------------------------------------------

Procedure TVectorMath.Add(const Scalar: Double; const Vector, Result: TArray<Double>);
var V, R: PDouble; N: Integer;
begin
  N := Length(Vector); V := Pointer(Vector); R := Pointer(Result);
  while N > 0 do begin R^ := V^ + Scalar; Inc(V); Inc(R); Dec(N); end;
end;

Procedure TVectorMath.Add(const Left, Right, Result: TArray<Double>);
var L, Ri, R: PDouble; N: Integer;
begin
  N := Length(Left); L := Pointer(Left); Ri := Pointer(Right); R := Pointer(Result);
  while N > 0 do begin R^ := L^ + Ri^; Inc(L); Inc(Ri); Inc(R); Dec(N); end;
end;

Procedure TVectorMath.Subtract(const Scalar: Double; const Vector, Result: TArray<Double>);
var V, R: PDouble; N: Integer;
begin
  N := Length(Vector); V := Pointer(Vector); R := Pointer(Result);
  while N > 0 do begin R^ := V^ - Scalar; Inc(V); Inc(R); Dec(N); end;
end;

Procedure TVectorMath.Subtract(const Left, Right, Result: TArray<Double>);
var L, Ri, R: PDouble; N: Integer;
begin
  N := Length(Left); L := Pointer(Left); Ri := Pointer(Right); R := Pointer(Result);
  while N > 0 do begin R^ := L^ - Ri^; Inc(L); Inc(Ri); Inc(R); Dec(N); end;
end;

Procedure TVectorMath.Multiply(const Scalar: Double; const Vector, Result: TArray<Double>);
var V, R: PDouble; N: Integer;
begin
  N := Length(Vector); V := Pointer(Vector); R := Pointer(Result);
  while N > 0 do begin R^ := V^ * Scalar; Inc(V); Inc(R); Dec(N); end;
end;

Procedure TVectorMath.Multiply(const Left, Right, Result: TArray<Double>);
var L, Ri, R: PDouble; N: Integer;
begin
  N := Length(Left); L := Pointer(Left); Ri := Pointer(Right); R := Pointer(Result);
  while N > 0 do begin R^ := L^ * Ri^; Inc(L); Inc(Ri); Inc(R); Dec(N); end;
end;

Procedure TVectorMath.Divide(const Scalar: Double; const Vector, Result: TArray<Double>);
var V, R: PDouble; N: Integer; Recip: Double;
begin
  Recip := 1.0 / Scalar;
  N := Length(Vector); V := Pointer(Vector); R := Pointer(Result);
  while N > 0 do begin R^ := V^ * Recip; Inc(V); Inc(R); Dec(N); end;
end;

Procedure TVectorMath.Divide(const Left, Right, Result: TArray<Double>);
var L, Ri, R: PDouble; N: Integer;
begin
  N := Length(Left); L := Pointer(Left); Ri := Pointer(Right); R := Pointer(Result);
  while N > 0 do begin R^ := L^ / Ri^; Inc(L); Inc(Ri); Inc(R); Dec(N); end;
end;

Procedure TVectorMath.Sqrt(const Vector, Result: TArray<Double>);
var V, R: PDouble; N: Integer;
begin
  N := Length(Vector); V := Pointer(Vector); R := Pointer(Result);
  while N > 0 do begin R^ := System.Sqrt(V^); Inc(V); Inc(R); Dec(N); end;
end;

function TVectorMath.Sum(const Vector: TArray<Double>): Double;
var V: PDouble; N: Integer;
begin
  Result := 0.0;
  N := Length(Vector); V := Pointer(Vector);
  while N > 0 do begin Result := Result + V^; Inc(V); Dec(N); end;
end;

Procedure TVectorMath.ExpHorner(Count: Integer; R, P: PDouble);
var I: Integer;
begin
  for I := 0 to Count - 1 do P[I] := ExpPolynomial(R[I]);
end;

Procedure TVectorMath.LogHorner(Count: Integer; S, NLn2, Res: PDouble);
var I: Integer;
begin
  for I := 0 to Count - 1 do Res[I] := LogPolynomial(S[I]) + NLn2[I];
end;

function TVectorMath.SumBlock(Data: PDouble; Count: Integer): Double;
var S0, S1, S2, S3: Double; I: Integer;
begin
  S0 := 0.0; S1 := 0.0; S2 := 0.0; S3 := 0.0; I := 0;
  while I + 3 < Count do
  begin
    S0 := S0 + Data[I]; S1 := S1 + Data[I+1];
    S2 := S2 + Data[I+2]; S3 := S3 + Data[I+3];
    Inc(I, 4);
  end;
  while I < Count do begin S0 := S0 + Data[I]; Inc(I); end;
  Result := (S0 + S1) + (S2 + S3);
end;

function TVectorMath.WeightedSumBlock(Data, Weights: PDouble; Count: Integer): Double;
// Four-way unrolled dot product: Sum Data[i] * Weights[i].
// Weights[i] = 0 -> contributes exactly 0 (IEEE 754 guarantees 0 * finite = 0).
var S0, S1, S2, S3: Double; I: Integer;
begin
  S0 := 0.0; S1 := 0.0; S2 := 0.0; S3 := 0.0; I := 0;
  while I + 3 < Count do
  begin
    S0 := S0 + Data[I]   * Weights[I];
    S1 := S1 + Data[I+1] * Weights[I+1];
    S2 := S2 + Data[I+2] * Weights[I+2];
    S3 := S3 + Data[I+3] * Weights[I+3];
    Inc(I, 4);
  end;
  while I < Count do begin S0 := S0 + Data[I] * Weights[I]; Inc(I); end;
  Result := (S0 + S1) + (S2 + S3);
end;

Procedure TVectorMath.ScaleAddBlock(Scale: Double; Count: Integer; Src, Dest: PDouble);
// Four-way unrolled SAXPY: Dest[i] += Scale * Src[i].
var S: Double; I: Integer;
begin
  S := Scale; I := 0;
  while I + 3 < Count do
  begin
    Dest[I]   := Dest[I]   + S * Src[I];
    Dest[I+1] := Dest[I+1] + S * Src[I+1];
    Dest[I+2] := Dest[I+2] + S * Src[I+2];
    Dest[I+3] := Dest[I+3] + S * Src[I+3];
    Inc(I, 4);
  end;
  while I < Count do begin Dest[I] := Dest[I] + S * Src[I]; Inc(I); end;
end;

Procedure TVectorMath.MultiplyBlock(Count: Integer; Src, InOut: PDouble);
// Four-way unrolled in-place multiply: InOut[i] *= Src[i].
var I: Integer;
begin
  I := 0;
  while I + 3 < Count do
  begin
    InOut[I]   := InOut[I]   * Src[I];
    InOut[I+1] := InOut[I+1] * Src[I+1];
    InOut[I+2] := InOut[I+2] * Src[I+2];
    InOut[I+3] := InOut[I+3] * Src[I+3];
    Inc(I, 4);
  end;
  while I < Count do begin InOut[I] := InOut[I] * Src[I]; Inc(I); end;
end;

Procedure TVectorMath.ScaleAdd(Scale: Double; Len: Integer;
  const Src, Dest: TArray<Double>);
begin
  ScaleAddBlock(Scale, Len, Pointer(Src), Pointer(Dest));
end;

// ---------------------------------------------------------------------------
// TVectorMath.Exponentiate
// ---------------------------------------------------------------------------

Procedure TVectorMath.Exp(const Vector, Result: TArray<Double>);
var Scratch: TScratchBuffer;
begin
  Exp(Vector, Result, Scratch);
end;

Procedure TVectorMath.Exp(const Vector, Result: TArray<Double>; var Scratch: TScratchBuffer);
// Cache-blocked polynomial exp(x) = 2^n * exp(r),  n = round(x/ln2),  |r| <= ln2/2
//
// Each block fits in L1 cache, so all pipeline stages hit L1 rather than
// making separate passes through main memory.  Accuracy: ~1-2 ULP.
//
// ExpBlock dispatches to the subclass (fused SIMD or scalar three-pass), or to
// System.Exp for a block with an argument beyond +-EXP_LIMIT.
var Blocks: TBlocks;
begin
  Scratch.EnsureSize(EXP_LOG_BLOCK);
  Blocks := TBlocks.Create(System.Length(Vector));
  while Blocks.Next do
    ExpBlock(Blocks.Count, @Vector[Blocks.Start], @Result[Blocks.Start], Scratch);
end;

Procedure TVectorMath.Exp(const Vector, Result: TArray<Double>; out Sum: Double);
var Scratch: TScratchBuffer;
begin
  Exp(Vector, Result, Sum, Scratch);
end;

Procedure TVectorMath.Exp(const Vector, Result: TArray<Double>; out Sum: Double; var Scratch: TScratchBuffer);
// Fused Exp + Sum: identical block loop to Exp(V,R,Scratch) with an extra
// accumulation step over each block while it is still L1-resident.
var Blocks: TBlocks;
begin
  Sum := 0.0;
  Scratch.EnsureSize(EXP_LOG_BLOCK);
  Blocks := TBlocks.Create(System.Length(Vector));
  while Blocks.Next do
  begin
    ExpBlock(Blocks.Count, @Vector[Blocks.Start], @Result[Blocks.Start], Scratch);
    Sum := Sum + SumBlock(@Result[Blocks.Start], Blocks.Count);
  end;
end;

// ---------------------------------------------------------------------------
// TVectorMath.Log
// ---------------------------------------------------------------------------

Procedure TVectorMath.Log(const Vector, Result: TArray<Double>);
var Scratch: TScratchBuffer;
begin
  Log(Vector, Result, Scratch);
end;

Procedure TVectorMath.Log(const Vector, Result: TArray<Double>; var Scratch: TScratchBuffer);
// Cache-blocked polynomial log(x) = n*ln2 + log(m),  x = 2^n * m,
// m in [sqrt(2)/2, sqrt(2)).  log(m) = 2*s*f(t),  s=(m-1)/(m+1),  t=s^2.
// f(t) = 1 + t/3 + t^2/5 + t^3/7 + t^4/9 + t^5/11 + t^6/13 + t^7/15
// |s| <= 0.172  ->  ~15 digit accuracy with 8 terms.
//
// LogCore dispatches to the subclass (fused SIMD or scalar two-pass).
var Blocks: TBlocks;
begin
  Scratch.EnsureSize(EXP_LOG_BLOCK);
  Blocks := TBlocks.Create(System.Length(Vector));
  while Blocks.Next do
    LogCore(Blocks.Count, @Vector[Blocks.Start], @Result[Blocks.Start],
      @Scratch.FDouble[0][0], @Scratch.FDouble[3][0]);
end;

Procedure TVectorMath.LogCore(Count: Integer; X, Res, S, NLn2: PDouble);
// Two-pass ln pipeline: argument reduction then Horner, each dispatching to
// the subclass virtuals.  S and NLn2 are workspace of at least Count elements.
var FullQuads, I: Integer;
begin
  FullQuads := Count shr 2;
  // Vectorised argument-reduction for full groups of 4, then the scalar tail
  if FullQuads > 0 then
    LogReduce(FullQuads, X, S, NLn2);
  for I := FullQuads shl 2 to Count - 1 do S[I] := LogReduction(X[I], NLn2[I]);
  // Single-pass Horner: t=s^2, g(t) polynomial, result = s*g(t) + nln2.
  LogHorner(Count, S, NLn2, Res);
end;

Procedure TVectorMath.ExpCore(Count: Integer; V, Res: PDouble;
  var Scratch: TScratchBuffer);
// Three-pass exp pipeline: reduce -> Horner -> reconstruct, each dispatching to
// the subclass virtuals, with a scalar tail after the full groups of 4.
// Uses FDouble[1] as R, FDouble[2] as P, FInt64 as N.
var
  FullQuads, I: Integer;
  R, P: TArray<Double>;
begin
  FullQuads := Count shr 2;
  R := Scratch.FDouble[1];
  P := Scratch.FDouble[2];
  if FullQuads > 0 then ExpReduce(FullQuads, V, @R[0], @Scratch.FInt64[0]);
  for I := FullQuads shl 2 to Count - 1 do R[I] := ExpReduction(V[I], Scratch.FInt64[I]);
  ExpHorner(Count, @R[0], @P[0]);
  if FullQuads > 0 then ExpRecon(FullQuads, @P[0], @Scratch.FInt64[0], Res);
  for I := FullQuads shl 2 to Count - 1 do Res[I] := P[I] * PowerOfTwo(Scratch.FInt64[I]);
end;

Function TVectorMath.ExpInRange(Count: Integer; V: PDouble): Boolean;
var I: Integer;
begin
  for I := 0 to Count - 1 do
    if not (Abs(V[I]) <= EXP_LIMIT) then Exit(false);
  Result := true;
end;

Procedure TVectorMath.ExpBlock(Count: Integer; V, Res: PDouble; var Scratch: TScratchBuffer);
// Beyond the range of ExpCore, System.Exp overflows, underflows to a denormal
// or zero, or passes a NaN, as the RTL does
var I: Integer;
begin
  if ExpInRange(Count, V) then
    ExpCore(Count, V, Res, Scratch)
  else
    for I := 0 to Count - 1 do Res[I] := System.Exp(V[I]);
end;

Procedure TVectorMath.MaskedScaleBlock(const Scale: Double; Count: Integer; X, Mask, Res: PDouble);
var I: Integer;
begin
  for I := 0 to Count - 1 do
    if Mask[I] <> 0.0 then Res[I] := Scale * X[I] else Res[I] := 0.0;
end;

Function TVectorMath.ExpMomentsBlock(Count: Integer; X, W, E: PDouble): TExpMoments;
var I: Integer; T: Double;
begin
  Result := Default(TExpMoments);
  for I := 0 to Count - 1 do
    if W[I] <> 0.0 then
    begin
      T := W[I] * X[I] * E[I];
      Result.First := Result.First + T;
      Result.Second := Result.Second + T * X[I];
    end;
end;

// ---------------------------------------------------------------------------
// TVectorMath.ExpMoments and MultiplyByExp: per block, the masked arguments
// Scale*X go to Scratch.FDouble[0] and their exponentials to FDouble[3]
// ---------------------------------------------------------------------------

Function TVectorMath.ExpMoments(const Scale: Double; const X, W: TArray<Double>): TExpMoments;
var Scratch: TScratchBuffer;
begin
  Result := ExpMoments(Scale, X, W, Scratch);
end;

Function TVectorMath.ExpMoments(const Scale: Double; const X, W: TArray<Double>; var Scratch: TScratchBuffer): TExpMoments;
var Blocks: TBlocks;
begin
  Result := Default(TExpMoments);
  Scratch.EnsureSize(EXP_LOG_BLOCK);
  Blocks := TBlocks.Create(System.Length(X));
  while Blocks.Next do
  begin
    MaskedScaleBlock(Scale, Blocks.Count, @X[Blocks.Start], @W[Blocks.Start], @Scratch.FDouble[0][0]);
    ExpBlock(Blocks.Count, @Scratch.FDouble[0][0], @Scratch.FDouble[3][0], Scratch);
    Result.Add(ExpMomentsBlock(Blocks.Count, @X[Blocks.Start], @W[Blocks.Start], @Scratch.FDouble[3][0]));
  end;
end;

Procedure TVectorMath.MultiplyByExp(const Scale: Double; const X, InOut: TArray<Double>);
var Scratch: TScratchBuffer;
begin
  MultiplyByExp(Scale, X, InOut, Scratch);
end;

Procedure TVectorMath.MultiplyByExp(const Scale: Double; const X, InOut: TArray<Double>; var Scratch: TScratchBuffer);
// An element with InOut[i] = 0 has argument 0 and factor 1
var Blocks: TBlocks;
begin
  Scratch.EnsureSize(EXP_LOG_BLOCK);
  Blocks := TBlocks.Create(System.Length(X));
  while Blocks.Next do
  begin
    MaskedScaleBlock(Scale, Blocks.Count, @X[Blocks.Start], @InOut[Blocks.Start], @Scratch.FDouble[0][0]);
    ExpBlock(Blocks.Count, @Scratch.FDouble[0][0], @Scratch.FDouble[3][0], Scratch);
    MultiplyBlock(Blocks.Count, @Scratch.FDouble[3][0], @InOut[Blocks.Start]);
  end;
end;

// ---------------------------------------------------------------------------

class function TVectorMath.CreateOptimal: TVectorMath;
var
  F: TCpuFeatures;
begin
  F := GetCpuFeatures;
  if F.HasAVX2 then Exit(TVectorMathAVX2.Create);
  if F.HasAVX  then Exit(TVectorMathAVX.Create);
  if F.HasSSE2 then Exit(TVectorMathSSE2.Create);
  Result := TVectorMath.Create;
end;

initialization
  Engine := TVectorMath.CreateOptimal;
finalization
  Engine.Free;

end.
