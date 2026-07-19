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

  TVectorMath = class abstract
  strict protected
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
// TVectorMath - scalar reduce/recon hooks (base-class defaults)
// ---------------------------------------------------------------------------

Procedure TVectorMath.ExpReduce(QuadCount: Integer; V, R: PDouble; N: PInt64);
const
  LOG2E = 1.4426950408889634;
  LN2HI = 6.93147180369123816490e-1;
  LN2LO = 1.90821492927058770002e-10;
var I, Count: Integer;
begin
  Count := QuadCount * 4;
  for I := 0 to Count - 1 do
  begin
    N[I] := System.Round(V[I] * LOG2E);
    R[I] := (V[I] - N[I] * LN2HI) - N[I] * LN2LO;
  end;
end;

Procedure TVectorMath.ExpRecon(QuadCount: Integer; P: PDouble; N: PInt64; Res: PDouble);
var
  I, Count: Integer;
  TwoN: Double;
  NI: Int64;
begin
  Count := QuadCount * 4;
  for I := 0 to Count - 1 do
  begin
    NI := N[I] + 1023;
    PInt64(@TwoN)^ := NI shl 52;
    Res[I] := P[I] * TwoN;
  end;
end;

Procedure TVectorMath.LogReduce(QuadCount: Integer; V, S, NLn2: PDouble);
const
  LN2   = 6.93147180559945309417e-1;
  SQRT2 = 1.41421356237309504880e0;
var
  I, Count, NI: Integer;
  Bits: Int64;
  M: Double;
begin
  Count := QuadCount * 4;
  for I := 0 to Count - 1 do
  begin
    Bits := PInt64(@V[I])^;
    NI   := Integer((Bits shr 52) and $7FF) - 1023;
    M    := V[I];
    PInt64(@M)^ := (Bits and $000FFFFFFFFFFFFF) or $3FF0000000000000;
    if M >= SQRT2 then begin M := M * 0.5; Inc(NI); end;
    S[I]    := (M - 1.0) / (M + 1.0);
    NLn2[I] := NI * LN2;
  end;
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
// Degree-12 Taylor: truncation error r^13/13! ~ 1.7e-16 at |r| = ln2/2, so the
// polynomial itself contributes ~1 ulp.  (Degree 10 looked sufficient but its
// truncation error is ~2e-13 - caught by Exp_MatchesSystemExp.)
var I: Integer; Ri, Pi: Double;
begin
  for I := 0 to Count - 1 do
  begin
    Ri := R[I];
    Pi := Ri * (1.0/479001600.0) + (1.0/39916800.0);
    Pi := Ri * Pi + (1.0/3628800.0);
    Pi := Ri * Pi + (1.0/362880.0);
    Pi := Ri * Pi + (1.0/40320.0);
    Pi := Ri * Pi + (1.0/5040.0);
    Pi := Ri * Pi + (1.0/720.0);
    Pi := Ri * Pi + (1.0/120.0);
    Pi := Ri * Pi + (1.0/24.0);
    Pi := Ri * Pi + (1.0/6.0);
    Pi := Ri * Pi + 0.5;
    Pi := Ri * Pi + 1.0;
    Pi := Ri * Pi + 1.0;
    P[I] := Pi;
  end;
end;

Procedure TVectorMath.LogHorner(Count: Integer; S, NLn2, Res: PDouble);
var I: Integer; Si, Ti, Pi: Double;
begin
  for I := 0 to Count - 1 do
  begin
    Si := S[I];
    Ti := Si * Si;
    Pi := Ti * (2.0/15.0) + (2.0/13.0);
    Pi := Ti * Pi + (2.0/11.0);
    Pi := Ti * Pi + (2.0/9.0);
    Pi := Ti * Pi + (2.0/7.0);
    Pi := Ti * Pi + (2.0/5.0);
    Pi := Ti * Pi + (2.0/3.0);
    Pi := Ti * Pi + 2.0;
    Res[I] := Si * Pi + NLn2[I];
  end;
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
// making separate passes through main memory.  Accuracy: ~1-2 ULP for |x| < 710.
//
// ExpCore dispatches to the subclass (fused SIMD or scalar three-pass).
var
  Len, BlockStart, BlockLen: Integer;
begin
  Len := Length(Vector);
  if Len = 0 then Exit;
  Scratch.EnsureSize(EXP_LOG_BLOCK);

  BlockStart := 0;
  while BlockStart < Len do
  begin
    BlockLen := Len - BlockStart;
    if BlockLen > EXP_LOG_BLOCK then BlockLen := EXP_LOG_BLOCK;
    ExpCore(BlockLen, @Vector[BlockStart], @Result[BlockStart], Scratch);
    Inc(BlockStart, EXP_LOG_BLOCK);
  end;
end;

Procedure TVectorMath.Exp(const Vector, Result: TArray<Double>; out Sum: Double);
var Scratch: TScratchBuffer;
begin
  Exp(Vector, Result, Sum, Scratch);
end;

Procedure TVectorMath.Exp(const Vector, Result: TArray<Double>; out Sum: Double; var Scratch: TScratchBuffer);
// Fused Exp + Sum: identical block loop to Exp(V,R,Scratch) with an extra
// accumulation step over each block while it is still L1-resident.
var
  Len, BlockStart, BlockLen: Integer;
begin
  Len := Length(Vector);
  Sum := 0.0;
  if Len = 0 then Exit;
  Scratch.EnsureSize(EXP_LOG_BLOCK);

  BlockStart := 0;
  while BlockStart < Len do
  begin
    BlockLen := Len - BlockStart;
    if BlockLen > EXP_LOG_BLOCK then BlockLen := EXP_LOG_BLOCK;
    ExpCore(BlockLen, @Vector[BlockStart], @Result[BlockStart], Scratch);
    Sum := Sum + SumBlock(@Result[BlockStart], BlockLen);
    Inc(BlockStart, EXP_LOG_BLOCK);
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
var
  Len, BlockStart, BlockLen: Integer;
begin
  Len := Length(Vector);
  if Len = 0 then Exit;
  Scratch.EnsureSize(EXP_LOG_BLOCK);

  BlockStart := 0;
  while BlockStart < Len do
  begin
    BlockLen := Len - BlockStart;
    if BlockLen > EXP_LOG_BLOCK then BlockLen := EXP_LOG_BLOCK;
    LogCore(BlockLen, @Vector[BlockStart], @Result[BlockStart],
      @Scratch.FDouble[0][0], @Scratch.FDouble[3][0]);
    Inc(BlockStart, EXP_LOG_BLOCK);
  end;
end;

Procedure TVectorMath.LogCore(Count: Integer; X, Res, S, NLn2: PDouble);
// Two-pass ln pipeline: argument reduction then Horner, each dispatching to
// the subclass virtuals.  S and NLn2 are workspace of at least Count elements.
const
  LN2   = 6.93147180559945309417e-1;
  SQRT2 = 1.41421356237309504880e0;
var
  FullQuads, TailStart, I, NI: Integer;
  Bits: Int64;
  M: Double;
begin
  FullQuads := Count shr 2;
  TailStart := FullQuads shl 2;

  // Vectorised argument-reduction for full groups of 4
  if FullQuads > 0 then
    LogReduce(FullQuads, X, S, NLn2);

  // Scalar tail
  for I := TailStart to Count - 1 do
  begin
    Bits := PInt64(@X[I])^;
    NI   := Integer((Bits shr 52) and $7FF) - 1023;
    M    := X[I];
    PInt64(@M)^ := (Bits and $000FFFFFFFFFFFFF) or $3FF0000000000000;
    if M >= SQRT2 then begin M := M * 0.5; Inc(NI); end;
    S[I]    := (M - 1.0) / (M + 1.0);
    NLn2[I] := NI * LN2;
  end;

  // Single-pass Horner: t=s^2, g(t) polynomial, result = s*g(t) + nln2.
  LogHorner(Count, S, NLn2, Res);
end;

Procedure TVectorMath.ExpCore(Count: Integer; V, Res: PDouble;
  var Scratch: TScratchBuffer);
// Three-pass exp pipeline: reduce -> Horner -> reconstruct, each dispatching to
// the subclass virtuals.  Uses FDouble[1] as R, FDouble[2] as P, FInt64 as N.
const
  LOG2E = 1.4426950408889634;
  LN2HI = 6.93147180369123816490e-1;
  LN2LO = 1.90821492927058770002e-10;
var
  FullQuads, TailStart, I: Integer;
  R, P: TArray<Double>;
  N: TArray<Int64>;
  TwoN: Double;
  NI: Int64;
begin
  FullQuads := Count shr 2;
  TailStart := FullQuads shl 2;
  R := Scratch.FDouble[1];
  P := Scratch.FDouble[2];
  N := Scratch.FInt64;

  if FullQuads > 0 then
    ExpReduce(FullQuads, V, @R[0], @N[0]);
  for I := TailStart to Count - 1 do
  begin
    N[I] := System.Round(V[I] * LOG2E);
    R[I] := (V[I] - N[I] * LN2HI) - N[I] * LN2LO;
  end;

  ExpHorner(Count, @R[0], @P[0]);

  if FullQuads > 0 then
    ExpRecon(FullQuads, @P[0], @N[0], Res);
  for I := TailStart to Count - 1 do
  begin
    NI := N[I] + 1023;
    PInt64(@TwoN)^ := NI shl 52;
    Res[I] := P[I] * TwoN;
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
