unit VecMath.Tests;

////////////////////////////////////////////////////////////////////////////////
//
// Author: Jaap Baak (initial implementation developed with Claude Fable 5)
// https://github.com/transportmodelling/VecMath
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Unit tests for VecMath.
//
// Strategy: every SIMD path has a trustworthy reference -
//   * element-wise ops: all engine implementations must agree bit-exactly
//     with the pure-Pascal base class (same arithmetic per element);
//   * Exp/Log: compared against System.Exp / System.Ln with tight tolerances
//     (sums compare with tolerance, since fused multiply-add and
//     multi-accumulator summation legitimately change rounding).
//
// Lengths are chosen to hit every code path: scalar tails (1-3), the 4-quad
// interleaved groups (15/16/17), the L1 block boundary (511/512/513), and
// multi-block arrays.
////////////////////////////////////////////////////////////////////////////////

interface

uses
  DUnitX.TestFramework,
  VecMath,
  VecMath.CPU,
  VecMath.SSE2,
  VecMath.AVX;

type
  [TestFixture]
  TVectorMathTests = class
  private
    FEngines: TArray<TVectorMath>;
    FNames:   TArray<string>;
    Procedure AddEngine(Engine: TVectorMath; const Name: string);
    Function RandomArray(Len: Integer; Lo, Hi: Double): TArray<Double>;
    Function RelErr(Actual, Ref: Double): Double;
    Procedure MaskEveryThird(const X, W: TArray<Double>);
    Function NaiveExpMoments(const Scale: Double; const X, W: TArray<Double>): TExpMoments;
  public
    [Setup]    Procedure Setup;
    [TearDown] Procedure TearDown;

    [Test] Procedure ElementWiseOps_AllEnginesBitIdentical;
    [Test] Procedure Sum_MatchesNaiveReference;
    [Test] Procedure WeightedSumBlock_MatchesNaiveReference;
    [Test] Procedure Exp_MatchesSystemExp;
    [Test] Procedure Log_MatchesSystemLn;
    [Test] Procedure ExpWithSum_SumMatchesArray;
    [Test] Procedure Exp_ScratchOverloadIdentical;
    [Test] Procedure Exp_BeyondLimit_MatchesSystemExp;
    [Test] Procedure Exp_UnderflowAnywhere_GivesZero;
    [Test] Procedure Exp_Overflow_LikeSystemExp;
    [Test] Procedure ExpMoments_MatchesNaiveReference;
    [Test] Procedure ExpMoments_BeyondLimit_MatchesNaiveReference;
    [Test] Procedure MultiplyByExp_MatchesSystemExp;
  end;

implementation

uses
  System.SysUtils,
  System.Math;

const
  // Lengths covering scalar tails, 4-quad groups, and L1 block boundaries.
  LENS: array[0..13] of Integer =
    (1, 2, 3, 4, 5, 7, 15, 16, 17, 511, 512, 513, 1000, 2048);

Procedure TVectorMathTests.AddEngine(Engine: TVectorMath; const Name: string);
begin
  FEngines := FEngines + [Engine];
  FNames   := FNames + [Name];
end;

Function TVectorMathTests.RandomArray(Len: Integer; Lo, Hi: Double): TArray<Double>;
var I: Integer;
begin
  SetLength(Result, Len);
  for I := 0 to Len - 1 do
    Result[I] := Lo + (Hi - Lo) * Random;
end;

// Relative error with an absolute floor: near-zero references are compared
// absolutely (relevant for Log results near ln(1) = 0).
Function TVectorMathTests.RelErr(Actual, Ref: Double): Double;
begin
  Result := Abs(Actual - Ref) / Max(1.0, Abs(Ref));
end;

// Weights with exact zeros at every third element, where X holds values that
// must not be used: a huge value, an infinity and a NaN
Procedure TVectorMathTests.MaskEveryThird(const X, W: TArray<Double>);
var I: Integer;
begin
  for I := 0 to High(W) do
    if I mod 3 = 0 then
    begin
      W[I] := 0.0;
      case (I div 3) mod 3 of
        0: X[I] := MaxDouble;
        1: X[I] := Infinity;
        else X[I] := NaN;
      end;
    end;
end;

// Sum W*X*exp(Scale*X) and Sum W*X^2*exp(Scale*X) over W <> 0, with System.Exp
Function TVectorMathTests.NaiveExpMoments(const Scale: Double; const X, W: TArray<Double>): TExpMoments;
var I: Integer; T: Double;
begin
  Result := Default(TExpMoments);
  for I := 0 to High(X) do
    if W[I] <> 0.0 then
    begin
      T := W[I] * X[I] * System.Exp(Scale * X[I]);
      Result.First := Result.First + T;
      Result.Second := Result.Second + T * X[I];
    end;
end;

Procedure TVectorMathTests.Setup;
var F: TCpuFeatures;
begin
  RandSeed := 20260611;   // fixed seed: failures must be reproducible
  AddEngine(TVectorMath.Create, 'Scalar');
  F := GetCpuFeatures;
  if F.HasSSE2 then AddEngine(TVectorMathSSE2.Create, 'SSE2');
  if F.HasAVX  then AddEngine(TVectorMathAVX.Create,  'AVX');
  if F.HasAVX2 then AddEngine(TVectorMathAVX2.Create, 'AVX2');
end;

Procedure TVectorMathTests.TearDown;
var I: Integer;
begin
  for I := 0 to High(FEngines) do FEngines[I].Free;
  FEngines := nil;
  FNames   := nil;
end;

Procedure TVectorMathTests.ElementWiseOps_AllEnginesBitIdentical;
// Add/Subtract/Multiply/Divide/Sqrt perform the same IEEE 754 operation per
// element in every implementation, so results must be bit-identical to the
// scalar base class - any difference indicates an indexing or tail bug.
var
  E, L, I, Op: Integer;
  A, B, RefR, R: TArray<Double>;
  Len: Integer;
const
  OPS = 9;
  OP_NAMES: array[0..OPS-1] of string = (
    'Add(s,V)', 'Add(V,V)', 'Subtract(s,V)', 'Subtract(V,V)',
    'Multiply(s,V)', 'Multiply(V,V)', 'Divide(s,V)', 'Divide(V,V)', 'Sqrt');

  Procedure Run(Eng: TVectorMath; const Res: TArray<Double>);
  begin
    case Op of
      0: Eng.Add(2.5, A, Res);
      1: Eng.Add(A, B, Res);
      2: Eng.Subtract(0.25, A, Res);
      3: Eng.Subtract(A, B, Res);
      4: Eng.Multiply(3.5, A, Res);
      5: Eng.Multiply(A, B, Res);
      6: Eng.Divide(3.0, A, Res);
      7: Eng.Divide(A, B, Res);
      8: Eng.Sqrt(A, Res);
    end;
  end;

begin
  for L := 0 to High(LENS) do
  begin
    Len := LENS[L];
    A := RandomArray(Len, 0.1, 100.0);   // positive: valid for Sqrt/Divide
    B := RandomArray(Len, 0.1, 100.0);
    SetLength(RefR, Len);
    SetLength(R, Len);
    for Op := 0 to OPS - 1 do
    begin
      Run(FEngines[0], RefR);            // scalar base = reference
      for E := 1 to High(FEngines) do
      begin
        Run(FEngines[E], R);
        for I := 0 to Len - 1 do
          Assert.IsTrue(R[I] = RefR[I], Format(
            '%s/%s Len=%d i=%d: %.17g <> %.17g',
            [FNames[E], OP_NAMES[Op], Len, I, R[I], RefR[I]]));
      end;
    end;
  end;
end;

Procedure TVectorMathTests.Sum_MatchesNaiveReference;
// Summation order differs between implementations (multi-accumulator SIMD),
// so compare with a tolerance instead of exactly.
var
  E, L, I: Integer;
  A: TArray<Double>;
  Ref, S: Double;
begin
  for L := 0 to High(LENS) do
  begin
    A := RandomArray(LENS[L], -1.0, 1.0);
    Ref := 0.0;
    for I := 0 to High(A) do Ref := Ref + A[I];
    for E := 0 to High(FEngines) do
    begin
      S := FEngines[E].Sum(A);
      Assert.IsTrue(Abs(S - Ref) <= 1e-12 * Max(1.0, Abs(Ref)), Format(
        '%s Sum Len=%d: %.17g <> ref %.17g',
        [FNames[E], LENS[L], S, Ref]));
    end;
  end;
end;

Procedure TVectorMathTests.WeightedSumBlock_MatchesNaiveReference;
// Dot product Sum Data[i]*Weights[i].  Accumulation order differs between
// implementations, so compare with a tolerance.  Weights include exact zeros:
// 0*finite must contribute exactly 0.
var
  E, L, I: Integer;
  A, W: TArray<Double>;
  Ref, S: Double;
begin
  for L := 0 to High(LENS) do
  begin
    A := RandomArray(LENS[L], -1.0, 1.0);
    W := RandomArray(LENS[L], 0.0, 2.0);
    for I := 0 to High(W) do
      if I mod 3 = 0 then W[I] := 0.0;
    Ref := 0.0;
    for I := 0 to High(A) do Ref := Ref + A[I] * W[I];
    for E := 0 to High(FEngines) do
    begin
      S := FEngines[E].WeightedSumBlock(Pointer(A), Pointer(W), LENS[L]);
      Assert.IsTrue(Abs(S - Ref) <= 1e-12 * Max(1.0, Abs(Ref)), Format(
        '%s WeightedSumBlock Len=%d: %.17g <> ref %.17g',
        [FNames[E], LENS[L], S, Ref]));
    end;
  end;
end;

Procedure TVectorMathTests.Exp_MatchesSystemExp;
var
  E, L, I: Integer;
  A, R: TArray<Double>;
  Ref, MaxErr, Err: Double;
begin
  for L := 0 to High(LENS) do
  begin
    A := RandomArray(LENS[L], -700.0, 700.0);   // full finite double range
    SetLength(R, LENS[L]);
    for E := 0 to High(FEngines) do
    begin
      FEngines[E].Exp(A, R);
      MaxErr := 0.0;
      for I := 0 to High(A) do
      begin
        Ref := System.Exp(A[I]);
        Err := Abs(R[I] - Ref) / Ref;           // exp > 0: pure relative
        if Err > MaxErr then MaxErr := Err;
      end;
      Assert.IsTrue(MaxErr < 2e-14, Format(
        '%s Exp Len=%d: max rel err %.3g exceeds 2e-14',
        [FNames[E], LENS[L], MaxErr]));
    end;
  end;
end;

Procedure TVectorMathTests.Log_MatchesSystemLn;
var
  E, L, I: Integer;
  A, R: TArray<Double>;
  MaxErr, Err: Double;
begin
  for L := 0 to High(LENS) do
  begin
    // Positive values across many magnitudes, including the ln(x)~0 region.
    SetLength(A, LENS[L]);
    for I := 0 to High(A) do
      A[I] := Power(10.0, -300.0 + 600.0 * Random);
    SetLength(R, LENS[L]);
    for E := 0 to High(FEngines) do
    begin
      FEngines[E].Log(A, R);
      MaxErr := 0.0;
      for I := 0 to High(A) do
      begin
        Err := RelErr(R[I], Ln(A[I]));
        if Err > MaxErr then MaxErr := Err;
      end;
      Assert.IsTrue(MaxErr < 2e-14, Format(
        '%s Log Len=%d: max err %.3g exceeds 2e-14',
        [FNames[E], LENS[L], MaxErr]));
    end;
  end;
end;

Procedure TVectorMathTests.ExpWithSum_SumMatchesArray;
// The fused Exp+Sum must return the sum of exactly the array it produced.
var
  E, L, I: Integer;
  A, R: TArray<Double>;
  S, Ref: Double;
begin
  for L := 0 to High(LENS) do
  begin
    A := RandomArray(LENS[L], -5.0, 5.0);
    SetLength(R, LENS[L]);
    for E := 0 to High(FEngines) do
    begin
      FEngines[E].Exp(A, R, S);
      Ref := 0.0;
      for I := 0 to High(R) do Ref := Ref + R[I];
      Assert.IsTrue(Abs(S - Ref) <= 1e-12 * Ref, Format(
        '%s Exp+Sum Len=%d: %.17g <> array sum %.17g',
        [FNames[E], LENS[L], S, Ref]));
    end;
  end;
end;

Procedure TVectorMathTests.Exp_ScratchOverloadIdentical;
// With/without explicit scratch must produce bit-identical results.
var
  E, I: Integer;
  A, R1, R2: TArray<Double>;
  Scratch: TScratchBuffer;
begin
  A := RandomArray(1000, -5.0, 5.0);
  SetLength(R1, 1000);
  SetLength(R2, 1000);
  for E := 0 to High(FEngines) do
  begin
    FEngines[E].Exp(A, R1);
    FEngines[E].Exp(A, R2, Scratch);
    for I := 0 to High(A) do
      Assert.IsTrue(R1[I] = R2[I], Format(
        '%s scratch overload differs at i=%d', [FNames[E], I]));
  end;
end;

Procedure TVectorMathTests.Exp_BeyondLimit_MatchesSystemExp;
// Arguments down to the denormal range and up to the largest finite result.
// A normal result matches System.Exp within the usual tolerance; a block with
// an argument beyond the limit takes System.Exp itself, so a denormal or zero
// result matches it exactly.
const
  MIN_NORMAL = 2.2250738585072014e-308;
var
  E, L, I: Integer;
  A, R: TArray<Double>;
  Ref, MaxErr, Err: Double;
begin
  for L := 0 to High(LENS) do
  begin
    A := RandomArray(LENS[L], -745.0, 709.7);
    SetLength(R, LENS[L]);
    for E := 0 to High(FEngines) do
    begin
      FEngines[E].Exp(A, R);
      MaxErr := 0.0;
      for I := 0 to High(A) do
      begin
        Ref := System.Exp(A[I]);
        if Ref >= MIN_NORMAL then
        begin
          Err := Abs(R[I] - Ref) / Ref;
          if Err > MaxErr then MaxErr := Err;
        end else
          Assert.IsTrue(R[I] = Ref, Format('%s Exp Len=%d i=%d: %.17g <> %.17g',
                                           [FNames[E], LENS[L], I, R[I], Ref]));
      end;
      Assert.IsTrue(MaxErr < 2e-14, Format(
        '%s Exp Len=%d: max rel err %.3g exceeds 2e-14',
        [FNames[E], LENS[L], MaxErr]));
    end;
  end;
end;

Procedure TVectorMathTests.Exp_UnderflowAnywhere_GivesZero;
// One argument far below the limit, at the start, in the middle or in the
// scalar tail of an array: the range check must see it wherever it is.
var
  E, L, P, Pos: Integer;
  A, R: TArray<Double>;
begin
  for L := 0 to High(LENS) do
  for P := 0 to 2 do
  begin
    A := RandomArray(LENS[L], -5.0, 5.0);
    case P of
      0: Pos := 0;
      1: Pos := LENS[L] div 2;
      else Pos := LENS[L] - 1;
    end;
    A[Pos] := -800.0;
    SetLength(R, LENS[L]);
    for E := 0 to High(FEngines) do
    begin
      FEngines[E].Exp(A, R);
      Assert.IsTrue(R[Pos] = 0.0, Format('%s Exp Len=%d i=%d: %.17g <> 0',
                                         [FNames[E], LENS[L], Pos, R[Pos]]));
    end;
  end;
end;

Procedure TVectorMathTests.Exp_Overflow_LikeSystemExp;
// Exp(710) overflows as System.Exp does: +Inf with the overflow exception
// masked, the default, and EOverflow with it unmasked
var
  E: Integer;
  A, R: TArray<Double>;
  Mask: TArithmeticExceptionMask;
begin
  A := RandomArray(100, -5.0, 5.0);
  A[37] := 710.0;
  SetLength(R, 100);
  Mask := GetExceptionMask;
  try
    SetExceptionMask(Mask + [exOverflow]);
    for E := 0 to High(FEngines) do
    begin
      FEngines[E].Exp(A, R);
      Assert.IsTrue(IsInfinite(R[37]) and (R[37] > 0), Format('%s Exp(710): %.17g', [FNames[E], R[37]]));
    end;
    SetExceptionMask(Mask - [exOverflow]);
    for E := 0 to High(FEngines) do
      Assert.WillRaise(
        Procedure
        begin
          FEngines[E].Exp(A, R);
        end, EOverflow, FNames[E]);
  finally
    SetExceptionMask(Mask);
  end;
end;

Procedure TVectorMathTests.ExpMoments_MatchesNaiveReference;
// Accumulation order differs between implementations, so compare with a tolerance
var
  E, L: Integer;
  X, W: TArray<Double>;
  Ref, M: TExpMoments;
begin
  for L := 0 to High(LENS) do
  begin
    X := RandomArray(LENS[L], -5.0, 5.0);
    W := RandomArray(LENS[L], 0.5, 2.0);
    MaskEveryThird(X, W);
    Ref := NaiveExpMoments(0.37, X, W);
    for E := 0 to High(FEngines) do
    begin
      M := FEngines[E].ExpMoments(0.37, X, W);
      Assert.IsTrue(RelErr(M.First, Ref.First) <= 1e-12, Format('%s ExpMoments.First Len=%d: %.17g <> ref %.17g',
                                                                [FNames[E], LENS[L], M.First, Ref.First]));
      Assert.IsTrue(RelErr(M.Second, Ref.Second) <= 1e-12, Format('%s ExpMoments.Second Len=%d: %.17g <> ref %.17g',
                                                                  [FNames[E], LENS[L], M.Second, Ref.Second]));
    end;
  end;
end;

Procedure TVectorMathTests.ExpMoments_BeyondLimit_MatchesNaiveReference;
// Arguments down to -1000, where System.Exp underflows to a denormal or zero
var
  E, L: Integer;
  X, W: TArray<Double>;
  Ref, M: TExpMoments;
begin
  for L := 0 to High(LENS) do
  begin
    X := RandomArray(LENS[L], -5.0, 3.0);
    W := RandomArray(LENS[L], 0.5, 2.0);
    MaskEveryThird(X, W);
    Ref := NaiveExpMoments(200.0, X, W);
    for E := 0 to High(FEngines) do
    begin
      M := FEngines[E].ExpMoments(200.0, X, W);
      Assert.IsTrue(Abs(M.First - Ref.First) <= 1e-12 * Max(1.0, Abs(Ref.First)), Format(
        '%s ExpMoments.First Len=%d: %.17g <> ref %.17g', [FNames[E], LENS[L], M.First, Ref.First]));
      Assert.IsTrue(Abs(M.Second - Ref.Second) <= 1e-12 * Max(1.0, Abs(Ref.Second)), Format(
        '%s ExpMoments.Second Len=%d: %.17g <> ref %.17g', [FNames[E], LENS[L], M.Second, Ref.Second]));
    end;
  end;
end;

Procedure TVectorMathTests.MultiplyByExp_MatchesSystemExp;
// A zero element stays exactly zero, whatever its X
var
  E, L, I: Integer;
  X, W, R: TArray<Double>;
  Ref: Double;
begin
  for L := 0 to High(LENS) do
  begin
    X := RandomArray(LENS[L], -5.0, 5.0);
    W := RandomArray(LENS[L], 0.5, 2.0);
    MaskEveryThird(X, W);
    for E := 0 to High(FEngines) do
    begin
      R := Copy(W);
      FEngines[E].MultiplyByExp(0.37, X, R);
      for I := 0 to High(R) do
      begin
        if W[I] <> 0.0 then Ref := W[I] * System.Exp(0.37 * X[I]) else Ref := 0.0;
        Assert.IsTrue((R[I] = Ref) or (Abs(R[I] - Ref) <= 2e-14 * Ref), Format(
          '%s MultiplyByExp Len=%d i=%d: %.17g <> %.17g', [FNames[E], LENS[L], I, R[I], Ref]));
      end;
    end;
  end;
end;

initialization
  TDUnitX.RegisterTestFixture(TVectorMathTests);

end.
