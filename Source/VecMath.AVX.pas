unit VecMath.AVX;

////////////////////////////////////////////////////////////////////////////////
//
// Author: Jaap Baak (initial implementation developed with Claude Fable 5)
// https://github.com/transportmodelling/VecMath
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
interface
////////////////////////////////////////////////////////////////////////////////

uses
  VecMath;

type
  TVectorMathAVX = class(TVectorMath)
  {$IF DEFINED(MSWINDOWS) AND DEFINED(CPUX64)}
  strict protected
    Function ExpInRange(Count: Integer; V: PDouble): Boolean; override;
    Procedure MaskedScaleBlock(const Scale: Double; Count: Integer; X, Mask, Res: PDouble); override;
    Function ExpMomentsBlock(Count: Integer; X, W, E: PDouble): TExpMoments; override;
  public
    Procedure Add     (const Scalar: Double; const Vector, Result: TArray<Double>); overload; override;
    Procedure Add     (const Left, Right, Result: TArray<Double>);                  overload; override;
    Procedure Subtract(const Scalar: Double; const Vector, Result: TArray<Double>); overload; override;
    Procedure Subtract(const Left, Right, Result: TArray<Double>);                  overload; override;
    Procedure Multiply(const Scalar: Double; const Vector, Result: TArray<Double>); overload; override;
    Procedure Multiply(const Left, Right, Result: TArray<Double>);                  overload; override;
    Procedure Divide  (const Scalar: Double; const Vector, Result: TArray<Double>); overload; override;
    Procedure Divide  (const Left, Right, Result: TArray<Double>);                  overload; override;
    Procedure Sqrt    (const Vector, Result: TArray<Double>); override;
    function  Sum            (const Vector: TArray<Double>): Double; override;
    function  SumBlock       (Data: PDouble; Count: Integer): Double; override;
    function  WeightedSumBlock(Data, Weights: PDouble; Count: Integer): Double; override;
    Procedure MultiplyBlock  (Count: Integer; Src, InOut: PDouble); override;
  {$ENDIF}
  end;

  // AVX2 adds fully-fused exp/ln pipelines and FMA-based accumulation.
  // All basic ops are inherited from TVectorMathAVX.
  TVectorMathAVX2 = class(TVectorMathAVX)
  {$IF DEFINED(MSWINDOWS) AND DEFINED(CPUX64)}
  strict protected
    // Fused exp/ln pipelines: reduce -> Horner (-> reconstruct) in a single asm
    // loop with all intermediates register-resident.
    Procedure ExpCore(Count: Integer; V, Res: PDouble;
      var Scratch: TScratchBuffer); override;
    Procedure LogCore(Count: Integer; X, Res, S, NLn2: PDouble); override;
  public
    function  WeightedSumBlock(Data, Weights: PDouble; Count: Integer): Double; override;
    Procedure ScaleAddBlock(Scale: Double; Count: Integer; Src, Dest: PDouble); override;
  {$ENDIF}
  end;

////////////////////////////////////////////////////////////////////////////////
implementation
////////////////////////////////////////////////////////////////////////////////

{$POINTERMATH ON}
{$IF DEFINED(MSWINDOWS) AND DEFINED(CPUX64)}

// Win64 calling convention:
//   Integer/pointer params : RCX, RDX, R8, R9
//   Double params          : XMM0, XMM1, XMM2, XMM3
// Each helper processes QuadCount groups of 4 doubles (one YMM register).
// VZEROUPPER is called at the end of every AVX helper to avoid the SSE/AVX
// transition penalty when returning to non-VEX code.
//
// Scalar broadcast (AVX without AVX2):
//   VUNPCKLPD XMM0, XMM0, XMM0   -> XMM0 = [s, s]
//   VINSERTF128 YMM0, YMM0, XMM0, 1  -> YMM0 = [s, s, s, s]

// ---- vector op vector -------------------------------------------------------

Procedure AVXVAddV(QuadCount: Integer; L, R, Res: PDouble);
// RCX=QuadCount  RDX=L  R8=R  R9=Res
asm
  TEST  ECX, ECX
  JLE   @done
@loop:
  VMOVUPD   YMM0, [RDX]
  VMOVUPD   YMM1, [R8]
  VADDPD    YMM0, YMM0, YMM1
  VMOVUPD   [R9], YMM0
  ADD       RDX, 32
  ADD       R8,  32
  ADD       R9,  32
  DEC       ECX
  JNZ       @loop
@done:
  VZEROUPPER
end;

Procedure AVXVSubV(QuadCount: Integer; L, R, Res: PDouble);
asm
  TEST  ECX, ECX
  JLE   @done
@loop:
  VMOVUPD   YMM0, [RDX]
  VMOVUPD   YMM1, [R8]
  VSUBPD    YMM0, YMM0, YMM1
  VMOVUPD   [R9], YMM0
  ADD       RDX, 32
  ADD       R8,  32
  ADD       R9,  32
  DEC       ECX
  JNZ       @loop
@done:
  VZEROUPPER
end;

Procedure AVXVMulV(QuadCount: Integer; L, R, Res: PDouble);
asm
  TEST  ECX, ECX
  JLE   @done
@loop:
  VMOVUPD   YMM0, [RDX]
  VMOVUPD   YMM1, [R8]
  VMULPD    YMM0, YMM0, YMM1
  VMOVUPD   [R9], YMM0
  ADD       RDX, 32
  ADD       R8,  32
  ADD       R9,  32
  DEC       ECX
  JNZ       @loop
@done:
  VZEROUPPER
end;

Procedure AVXVDivV(QuadCount: Integer; L, R, Res: PDouble);
asm
  TEST  ECX, ECX
  JLE   @done
@loop:
  VMOVUPD   YMM0, [RDX]
  VMOVUPD   YMM1, [R8]
  VDIVPD    YMM0, YMM0, YMM1
  VMOVUPD   [R9], YMM0
  ADD       RDX, 32
  ADD       R8,  32
  ADD       R9,  32
  DEC       ECX
  JNZ       @loop
@done:
  VZEROUPPER
end;

Procedure AVXVSqrt(QuadCount: Integer; V, Res: PDouble);
// RCX=QuadCount  RDX=V  R8=Res
asm
  TEST  ECX, ECX
  JLE   @done
@loop:
  VMOVUPD   YMM0, [RDX]
  VSQRTPD   YMM0, YMM0
  VMOVUPD   [R8], YMM0
  ADD       RDX, 32
  ADD       R8,  32
  DEC       ECX
  JNZ       @loop
@done:
  VZEROUPPER
end;

// ---- scalar op vector -------------------------------------------------------
// Scalar in XMM0 (position 0), QuadCount in RDX, V in R8, Res in R9.

Procedure AVXSAddV(Scalar: Double; QuadCount: Integer; V, Res: PDouble);
// XMM0=Scalar  RDX=QuadCount  R8=V  R9=Res
asm
  TEST        EDX, EDX
  JLE         @done
  VUNPCKLPD   XMM0, XMM0, XMM0     // XMM0 = [s, s]
  VINSERTF128 YMM0, YMM0, XMM0, 1  // YMM0 = [s, s, s, s]
@loop:
  VMOVUPD     YMM1, [R8]
  VADDPD      YMM1, YMM1, YMM0
  VMOVUPD     [R9], YMM1
  ADD         R8, 32
  ADD         R9, 32
  DEC         EDX
  JNZ         @loop
@done:
  VZEROUPPER
end;

Procedure AVXSSubV(Scalar: Double; QuadCount: Integer; V, Res: PDouble);
asm
  TEST        EDX, EDX
  JLE         @done
  VUNPCKLPD   XMM0, XMM0, XMM0
  VINSERTF128 YMM0, YMM0, XMM0, 1
@loop:
  VMOVUPD     YMM1, [R8]
  VSUBPD      YMM1, YMM1, YMM0
  VMOVUPD     [R9], YMM1
  ADD         R8, 32
  ADD         R9, 32
  DEC         EDX
  JNZ         @loop
@done:
  VZEROUPPER
end;

Procedure AVXSMulV(Scalar: Double; QuadCount: Integer; V, Res: PDouble);
asm
  TEST        EDX, EDX
  JLE         @done
  VUNPCKLPD   XMM0, XMM0, XMM0
  VINSERTF128 YMM0, YMM0, XMM0, 1
@loop:
  VMOVUPD     YMM1, [R8]
  VMULPD      YMM1, YMM1, YMM0
  VMOVUPD     [R9], YMM1
  ADD         R8, 32
  ADD         R9, 32
  DEC         EDX
  JNZ         @loop
@done:
  VZEROUPPER
end;

// ---- horizontal sum ---------------------------------------------------------

function AVXSumV(QuadCount: Integer; V: PDouble): Double;
// RCX=QuadCount  RDX=V  -> result in XMM0
//
// Four accumulators (YMM0-YMM3) break the VADDPD latency chain: with a
// single accumulator each add must wait ~4 cycles for the previous one,
// capping throughput at 1 double/cycle; four independent chains keep the
// add ports busy and let L1 loads run at full speed.
asm
  VXORPD       YMM0, YMM0, YMM0
  VXORPD       YMM1, YMM1, YMM1
  VXORPD       YMM2, YMM2, YMM2
  VXORPD       YMM3, YMM3, YMM3
  // 4-quad groups (16 doubles per iteration)
  MOV          EAX, ECX
  SHR          EAX, 2
  JZ           @rest
@loop4:
  VADDPD       YMM0, YMM0, [RDX]
  VADDPD       YMM1, YMM1, [RDX+32]
  VADDPD       YMM2, YMM2, [RDX+64]
  VADDPD       YMM3, YMM3, [RDX+96]
  ADD          RDX, 128
  DEC          EAX
  JNZ          @loop4
@rest:
  // Remaining 0-3 quads
  AND          ECX, 3
  JZ           @reduce
@loop1:
  VADDPD       YMM0, YMM0, [RDX]
  ADD          RDX, 32
  DEC          ECX
  JNZ          @loop1
@reduce:
  VADDPD       YMM0, YMM0, YMM1
  VADDPD       YMM2, YMM2, YMM3
  VADDPD       YMM0, YMM0, YMM2
  VEXTRACTF128 XMM1, YMM0, 1      // XMM1 = [e2, e3]
  VADDPD       XMM0, XMM0, XMM1  // XMM0 = [e0+e2, e1+e3]
  VUNPCKHPD    XMM1, XMM0, XMM0  // XMM1[0] = e1+e3
  VADDSD       XMM0, XMM0, XMM1  // XMM0[0] = e0+e1+e2+e3
  VZEROUPPER
end;

// ---- method wrappers --------------------------------------------------------

Procedure TVectorMathAVX.Add(const Scalar: Double; const Vector, Result: TArray<Double>);
var Len, Tail: Integer;
begin
  Len  := Length(Vector);
  Tail := Len and 3;
  if Len >= 4 then AVXSAddV(Scalar, Len shr 2, @Vector[0], @Result[0]);
  while Tail > 0 do
  begin
    Result[Len - Tail] := Vector[Len - Tail] + Scalar;
    Dec(Tail);
  end;
end;

Procedure TVectorMathAVX.Add(const Left, Right, Result: TArray<Double>);
var Len, Tail: Integer;
begin
  Len  := Length(Left);
  Tail := Len and 3;
  if Len >= 4 then AVXVAddV(Len shr 2, @Left[0], @Right[0], @Result[0]);
  while Tail > 0 do
  begin
    Result[Len - Tail] := Left[Len - Tail] + Right[Len - Tail];
    Dec(Tail);
  end;
end;

Procedure TVectorMathAVX.Subtract(const Scalar: Double; const Vector, Result: TArray<Double>);
var Len, Tail: Integer;
begin
  Len  := Length(Vector);
  Tail := Len and 3;
  if Len >= 4 then AVXSSubV(Scalar, Len shr 2, @Vector[0], @Result[0]);
  while Tail > 0 do
  begin
    Result[Len - Tail] := Vector[Len - Tail] - Scalar;
    Dec(Tail);
  end;
end;

Procedure TVectorMathAVX.Subtract(const Left, Right, Result: TArray<Double>);
var Len, Tail: Integer;
begin
  Len  := Length(Left);
  Tail := Len and 3;
  if Len >= 4 then AVXVSubV(Len shr 2, @Left[0], @Right[0], @Result[0]);
  while Tail > 0 do
  begin
    Result[Len - Tail] := Left[Len - Tail] - Right[Len - Tail];
    Dec(Tail);
  end;
end;

Procedure TVectorMathAVX.Multiply(const Scalar: Double; const Vector, Result: TArray<Double>);
var Len, Tail: Integer;
begin
  Len  := Length(Vector);
  Tail := Len and 3;
  if Len >= 4 then AVXSMulV(Scalar, Len shr 2, @Vector[0], @Result[0]);
  while Tail > 0 do
  begin
    Result[Len - Tail] := Vector[Len - Tail] * Scalar;
    Dec(Tail);
  end;
end;

Procedure TVectorMathAVX.Multiply(const Left, Right, Result: TArray<Double>);
var Len, Tail: Integer;
begin
  Len  := Length(Left);
  Tail := Len and 3;
  if Len >= 4 then AVXVMulV(Len shr 2, @Left[0], @Right[0], @Result[0]);
  while Tail > 0 do
  begin
    Result[Len - Tail] := Left[Len - Tail] * Right[Len - Tail];
    Dec(Tail);
  end;
end;

Procedure TVectorMathAVX.Divide(const Scalar: Double; const Vector, Result: TArray<Double>);
var Len, Tail: Integer; Recip: Double;
begin
  Len  := Length(Vector);
  Tail := Len and 3;
  Recip := 1.0 / Scalar;
  if Len >= 4 then AVXSMulV(Recip, Len shr 2, @Vector[0], @Result[0]);
  // Tail multiplies by the reciprocal like the SIMD body and the scalar base
  // class, so all elements (and all engines) round identically.
  while Tail > 0 do
  begin
    Result[Len - Tail] := Vector[Len - Tail] * Recip;
    Dec(Tail);
  end;
end;

Procedure TVectorMathAVX.Divide(const Left, Right, Result: TArray<Double>);
var Len, Tail: Integer;
begin
  Len  := Length(Left);
  Tail := Len and 3;
  if Len >= 4 then AVXVDivV(Len shr 2, @Left[0], @Right[0], @Result[0]);
  while Tail > 0 do
  begin
    Result[Len - Tail] := Left[Len - Tail] / Right[Len - Tail];
    Dec(Tail);
  end;
end;

Procedure TVectorMathAVX.Sqrt(const Vector, Result: TArray<Double>);
var Len, Tail: Integer;
begin
  Len  := Length(Vector);
  Tail := Len and 3;
  if Len >= 4 then AVXVSqrt(Len shr 2, @Vector[0], @Result[0]);
  while Tail > 0 do
  begin
    Result[Len - Tail] := System.Sqrt(Vector[Len - Tail]);
    Dec(Tail);
  end;
end;

function TVectorMathAVX.Sum(const Vector: TArray<Double>): Double;
var Len, Tail, I: Integer;
begin
  Len  := Length(Vector);
  Tail := Len and 3;
  if Len >= 4 then
    Result := AVXSumV(Len shr 2, @Vector[0])
  else
    Result := 0.0;
  for I := Len - Tail to Len - 1 do
    Result := Result + Vector[I];
end;

function TVectorMathAVX.SumBlock(Data: PDouble; Count: Integer): Double;
var Tail, I: Integer;
begin
  Tail := Count and 3;
  if Count >= 4 then Result := AVXSumV(Count shr 2, Data)
  else Result := 0.0;
  for I := Count - Tail to Count - 1 do Result := Result + Data[I];
end;

// Weighted sum: Sum Data[i] * Weights[i].
// Four accumulators break the VADDPD latency chain (see AVXSumV).
// Win64: RCX=QuadCount  RDX=Data  R8=Weights  -> result in XMM0
function AVXWeightedSumV(QuadCount: Integer; Data, Weights: PDouble): Double;
asm
  VXORPD    YMM0, YMM0, YMM0
  VXORPD    YMM1, YMM1, YMM1
  VXORPD    YMM2, YMM2, YMM2
  VXORPD    YMM3, YMM3, YMM3
  // 4-quad groups (16 doubles per iteration)
  MOV       EAX, ECX
  SHR       EAX, 2
  JZ        @rest
@loop4:
  VMOVUPD   YMM4, [RDX]
  VMULPD    YMM4, YMM4, [R8]
  VADDPD    YMM0, YMM0, YMM4
  VMOVUPD   YMM5, [RDX+32]
  VMULPD    YMM5, YMM5, [R8+32]
  VADDPD    YMM1, YMM1, YMM5
  VMOVUPD   YMM4, [RDX+64]
  VMULPD    YMM4, YMM4, [R8+64]
  VADDPD    YMM2, YMM2, YMM4
  VMOVUPD   YMM5, [RDX+96]
  VMULPD    YMM5, YMM5, [R8+96]
  VADDPD    YMM3, YMM3, YMM5
  ADD       RDX, 128
  ADD       R8,  128
  DEC       EAX
  JNZ       @loop4
@rest:
  // Remaining 0-3 quads
  AND       ECX, 3
  JZ        @reduce
@loop1:
  VMOVUPD   YMM4, [RDX]
  VMULPD    YMM4, YMM4, [R8]
  VADDPD    YMM0, YMM0, YMM4
  ADD       RDX, 32
  ADD       R8,  32
  DEC       ECX
  JNZ       @loop1
@reduce:
  VADDPD    YMM0, YMM0, YMM1
  VADDPD    YMM2, YMM2, YMM3
  VADDPD    YMM0, YMM0, YMM2
  VEXTRACTF128 XMM1, YMM0, 1
  VADDPD    XMM0, XMM0, XMM1
  VUNPCKHPD XMM1, XMM0, XMM0
  VADDSD    XMM0, XMM0, XMM1
  VZEROUPPER
end;

function TVectorMathAVX.WeightedSumBlock(Data, Weights: PDouble; Count: Integer): Double;
var Tail, I: Integer;
begin
  Tail := Count and 3;
  if Count >= 4 then Result := AVXWeightedSumV(Count shr 2, Data, Weights)
  else Result := 0.0;
  for I := Count - Tail to Count - 1 do
    Result := Result + Data[I] * Weights[I];
end;

// Whether |v| <= 708 (TVectorMath.EXP_LIMIT) for QuadCount groups of 4 doubles;
// a NaN fails. A lane fails when |v| is not <= 708 (NLE_UQ, true for a NaN),
// and the failures are ORed, a one-cycle chain, so the scan runs at load speed.
// Win64: RCX=QuadCount  RDX=V  -> AL
Function AVXExpInRangeV(QuadCount: Integer; V: PDouble): Boolean;
asm
  MOV       RAX, $7FFFFFFFFFFFFFFF   // clears the sign bit: |v|
  VMOVQ     XMM2, RAX
  VUNPCKLPD XMM2, XMM2, XMM2
  VINSERTF128 YMM2, YMM2, XMM2, 1
  MOV       RAX, $4086200000000000   // 708.0
  VMOVQ     XMM3, RAX
  VUNPCKLPD XMM3, XMM3, XMM3
  VINSERTF128 YMM3, YMM3, XMM3, 1
  VXORPD    YMM0, YMM0, YMM0          // failed lanes
  TEST      ECX, ECX
  JLE       @check
@loop:
  VANDPD    YMM1, YMM2, [RDX]
  VCMPPD    YMM1, YMM1, YMM3, 22      // NLE_UQ
  VORPD     YMM0, YMM0, YMM1
  ADD       RDX, 32
  DEC       ECX
  JNZ       @loop
@check:
  VTESTPD   YMM0, YMM0                // ZF = 1 when no lane failed
  SETZ      AL
  VZEROUPPER
end;

Function TVectorMathAVX.ExpInRange(Count: Integer; V: PDouble): Boolean;
var Tail: Integer;
begin
  Tail := Count and 3;
  Result := AVXExpInRangeV(Count shr 2, V) and inherited ExpInRange(Tail, @V[Count - Tail]);
end;

// Res[i] := Scale*X[i] where Mask[i] <> 0, else 0, for QuadCount groups of 4.
// X is ANDed with the mask before the multiplication, so a masked X, however
// large or NaN, raises no floating point exception.
// Win64: XMM0=Scale  EDX=QuadCount  R8=X  R9=Mask  stack=Res
Procedure AVXMaskedScaleV(const Scale: Double; QuadCount: Integer; X, Mask, Res: PDouble);
asm
  TEST      EDX, EDX
  JLE       @done
  MOV       RAX, Res
  VUNPCKLPD XMM0, XMM0, XMM0
  VINSERTF128 YMM0, YMM0, XMM0, 1     // Scale in all lanes
  VXORPD    YMM1, YMM1, YMM1
@loop:
  VCMPPD    YMM2, YMM1, [R9], 4       // NEQ_UQ: Mask <> 0
  VANDPD    YMM2, YMM2, [R8]
  VMULPD    YMM2, YMM2, YMM0
  VMOVUPD   [RAX], YMM2
  ADD       R8,  32
  ADD       R9,  32
  ADD       RAX, 32
  DEC       EDX
  JNZ       @loop
@done:
  VZEROUPPER
end;

Procedure TVectorMathAVX.MaskedScaleBlock(const Scale: Double; Count: Integer; X, Mask, Res: PDouble);
var Tail: Integer;
begin
  Tail := Count and 3;
  AVXMaskedScaleV(Scale, Count shr 2, X, Mask, Res);
  inherited MaskedScaleBlock(Scale, Tail, @X[Count - Tail], @Mask[Count - Tail], @Res[Count - Tail]);
end;

// Moments over QuadCount groups of 4: Sums[0] := Sum W*X*E, Sums[1] := Sum W*X^2*E
// for W <> 0. X is ANDed with the mask W <> 0 first, so a masked X contributes
// an exact zero. Two streams of two accumulators hide the VADDPD latency.
// Win64: ECX=QuadCount  RDX=X  R8=W  R9=E  stack=Sums (2 doubles)
Procedure AVXExpMomentsV(QuadCount: Integer; X, W, E, Sums: PDouble);
asm
  VXORPD    YMM0, YMM0, YMM0          // first moment, stream a
  VXORPD    YMM1, YMM1, YMM1          // first moment, stream b
  VXORPD    YMM2, YMM2, YMM2          // second moment, stream a
  VXORPD    YMM3, YMM3, YMM3          // second moment, stream b
  MOV       EAX, ECX
  SHR       EAX, 1
  JZ        @rest
@loop2:
  VXORPD    YMM5, YMM5, YMM5
  VCMPPD    YMM5, YMM5, [R8], 4       // NEQ_UQ: W <> 0
  VANDPD    YMM5, YMM5, [RDX]         // masked X
  VMULPD    YMM4, YMM5, [R8]
  VMULPD    YMM4, YMM4, [R9]          // T = X*W*E
  VADDPD    YMM0, YMM0, YMM4
  VMULPD    YMM4, YMM4, YMM5
  VADDPD    YMM2, YMM2, YMM4
  VXORPD    YMM5, YMM5, YMM5
  VCMPPD    YMM5, YMM5, [R8+32], 4
  VANDPD    YMM5, YMM5, [RDX+32]
  VMULPD    YMM4, YMM5, [R8+32]
  VMULPD    YMM4, YMM4, [R9+32]
  VADDPD    YMM1, YMM1, YMM4
  VMULPD    YMM4, YMM4, YMM5
  VADDPD    YMM3, YMM3, YMM4
  ADD       RDX, 64
  ADD       R8,  64
  ADD       R9,  64
  DEC       EAX
  JNZ       @loop2
@rest:
  AND       ECX, 1
  JZ        @reduce
  VXORPD    YMM5, YMM5, YMM5
  VCMPPD    YMM5, YMM5, [R8], 4
  VANDPD    YMM5, YMM5, [RDX]
  VMULPD    YMM4, YMM5, [R8]
  VMULPD    YMM4, YMM4, [R9]
  VADDPD    YMM0, YMM0, YMM4
  VMULPD    YMM4, YMM4, YMM5
  VADDPD    YMM2, YMM2, YMM4
@reduce:
  VADDPD    YMM0, YMM0, YMM1
  VEXTRACTF128 XMM1, YMM0, 1
  VADDPD    XMM0, XMM0, XMM1
  VUNPCKHPD XMM1, XMM0, XMM0
  VADDSD    XMM0, XMM0, XMM1
  VADDPD    YMM2, YMM2, YMM3
  VEXTRACTF128 XMM3, YMM2, 1
  VADDPD    XMM2, XMM2, XMM3
  VUNPCKHPD XMM3, XMM2, XMM2
  VADDSD    XMM2, XMM2, XMM3
  MOV       RAX, Sums
  VMOVSD    [RAX],   XMM0
  VMOVSD    [RAX+8], XMM2
  VZEROUPPER
end;

Function TVectorMathAVX.ExpMomentsBlock(Count: Integer; X, W, E: PDouble): TExpMoments;
var Tail: Integer;
begin
  Tail := Count and 3;
  AVXExpMomentsV(Count shr 2, X, W, E, @Result.First);
  Result.Add(inherited ExpMomentsBlock(Tail, @X[Count - Tail], @W[Count - Tail], @E[Count - Tail]));
end;

// In-place multiply: InOut[i] *= Src[i], 4 doubles per iteration.
// Streaming (no loop-carried dependency), so one accumulator-free pass suffices.
// Win64: RCX=QuadCount  RDX=Src  R8=InOut
Procedure AVXMulVIP(QuadCount: Integer; Src, InOut: PDouble);
asm
  TEST  ECX, ECX
  JLE   @done
@loop:
  VMOVUPD  YMM0, [R8]
  VMULPD   YMM0, YMM0, [RDX]
  VMOVUPD  [R8], YMM0
  ADD   RDX, 32
  ADD   R8,  32
  DEC   ECX
  JNZ   @loop
@done:
  VZEROUPPER
end;

Procedure TVectorMathAVX.MultiplyBlock(Count: Integer; Src, InOut: PDouble);
var Tail, I: Integer;
begin
  Tail := Count and 3;
  if Count >= 4 then AVXMulVIP(Count shr 2, Src, InOut);
  for I := Count - Tail to Count - 1 do
    InOut[I] := InOut[I] * Src[I];
end;

// ---------------------------------------------------------------------------
// Horner coefficients for exp(r) = 1 + r + r^2/2! + ... + r^12/12!, those of
// TVectorMath.ExpPolynomial, evaluated outermost to innermost:
//   p = r*c[0]+c[1]; p = r*p+c[2]; ... ; p = r*p+c[11]; p = r*p+c[11]
// c[11] = 1.0 is reused for the final two steps (x^1 and x^0 terms).
// Pre-broadcast: row k holds c[k] in all four lanes so a row can be used
// directly as a 256-bit FMA memory operand (AVX2 has no embedded broadcast -
// that arrived with AVX-512). Row k lives at byte offset k*32.
// ---------------------------------------------------------------------------

const
  EXP_HORNER_C4: array[0..47] of Double = (
    1.0/479001600.0, 1.0/479001600.0, 1.0/479001600.0, 1.0/479001600.0, // c0
    1.0/39916800.0,  1.0/39916800.0,  1.0/39916800.0,  1.0/39916800.0,  // c1
    1.0/3628800.0,   1.0/3628800.0,   1.0/3628800.0,   1.0/3628800.0,   // c2
    1.0/362880.0,    1.0/362880.0,    1.0/362880.0,    1.0/362880.0,    // c3
    1.0/40320.0,     1.0/40320.0,     1.0/40320.0,     1.0/40320.0,     // c4
    1.0/5040.0,      1.0/5040.0,      1.0/5040.0,      1.0/5040.0,      // c5
    1.0/720.0,       1.0/720.0,       1.0/720.0,       1.0/720.0,       // c6
    1.0/120.0,       1.0/120.0,       1.0/120.0,       1.0/120.0,       // c7
    1.0/24.0,        1.0/24.0,        1.0/24.0,        1.0/24.0,        // c8
    1.0/6.0,         1.0/6.0,         1.0/6.0,         1.0/6.0,         // c9
    0.5,             0.5,             0.5,             0.5,             // c10
    1.0,             1.0,             1.0,             1.0              // c11
  );

// Fused AVX2+FMA exp pipeline: range reduction, Horner polynomial and
// 2^n reconstruction in one loop, 4 vectors (16 doubles) per iteration.
// r, n and p never leave registers - the only memory traffic is the input
// load, the result store, and the L1-resident coefficient table.
//
// The 4-way interleave overlaps four dependent Horner chains so the two FMA
// ports stay busy (a single chain waits the full FMA latency every step).
//
// Per vector: n = round(v*log2e);  r = v - n*ln2hi - n*ln2lo  (FNMA);
//             p = Horner(r) from C4 rows;  Res = p * 2^n via exponent
//             injection (int32 convert -> +1023 -> shift into exponent field).
//
// Win64: RCX=QuadCount  RDX=V  R8=Res  R9=C4 (pointer to EXP_HORNER_C4)
// Registers: YMM0-YMM3 = v/r, YMM4-YMM7 = p, YMM8-YMM11 = n,
//            YMM12 = LOG2E, YMM13 = LN2HI, YMM14 = LN2LO, YMM15 = 1023.
Procedure AVX2ExpCore4(QuadCount: Integer; V, Res, C4: PDouble);
asm
  TEST  ECX, ECX
  JLE   @done
  // Save non-volatile XMM6-XMM15 (Win64 ABI: lower 128-bit halves)
  SUB   RSP, 168
  MOVDQU [RSP+8],   XMM6
  MOVDQU [RSP+24],  XMM7
  MOVDQU [RSP+40],  XMM8
  MOVDQU [RSP+56],  XMM9
  MOVDQU [RSP+72],  XMM10
  MOVDQU [RSP+88],  XMM11
  MOVDQU [RSP+104], XMM12
  MOVDQU [RSP+120], XMM13
  MOVDQU [RSP+136], XMM14
  MOVDQU [RSP+152], XMM15
  // Loop-invariant constants
  MOV    RAX, $3FF71547652B82FE   // LOG2E = 1/ln(2)
  VMOVQ  XMM12, RAX
  VBROADCASTSD YMM12, XMM12
  MOV    RAX, $3FE62E42FEFA3800   // LN2HI (trailing mantissa bits zeroed)
  VMOVQ  XMM13, RAX
  VBROADCASTSD YMM13, XMM13
  MOV    RAX, $3D2EF35793C76730   // LN2LO (LN2HI+LN2LO = ln2 to full precision)
  VMOVQ  XMM14, RAX
  VBROADCASTSD YMM14, XMM14
  MOV    RAX, 1023                // IEEE 754 exponent bias (int64)
  VMOVQ  XMM15, RAX
  VPBROADCASTQ YMM15, XMM15
  // 4-quad groups (16 doubles per iteration)
  MOV   EAX, ECX
  SHR   EAX, 2
  JZ    @rest
@loop4:
  VMOVUPD      YMM0, [RDX]            // v0..v3
  VMOVUPD      YMM1, [RDX+32]
  VMOVUPD      YMM2, [RDX+64]
  VMOVUPD      YMM3, [RDX+96]
  // n = round(v * log2e)
  VMULPD       YMM8,  YMM0, YMM12
  VMULPD       YMM9,  YMM1, YMM12
  VMULPD       YMM10, YMM2, YMM12
  VMULPD       YMM11, YMM3, YMM12
  VROUNDPD     YMM8,  YMM8,  0
  VROUNDPD     YMM9,  YMM9,  0
  VROUNDPD     YMM10, YMM10, 0
  VROUNDPD     YMM11, YMM11, 0
  // r = (v - n*LN2HI) - n*LN2LO   (in place: v becomes r)
  VFNMADD231PD YMM0, YMM8,  YMM13
  VFNMADD231PD YMM1, YMM9,  YMM13
  VFNMADD231PD YMM2, YMM10, YMM13
  VFNMADD231PD YMM3, YMM11, YMM13
  VFNMADD231PD YMM0, YMM8,  YMM14
  VFNMADD231PD YMM1, YMM9,  YMM14
  VFNMADD231PD YMM2, YMM10, YMM14
  VFNMADD231PD YMM3, YMM11, YMM14
  // p = r*c0 + c1
  VMULPD       YMM4, YMM0, [R9]
  VMULPD       YMM5, YMM1, [R9]
  VMULPD       YMM6, YMM2, [R9]
  VMULPD       YMM7, YMM3, [R9]
  VADDPD       YMM4, YMM4, [R9+32]
  VADDPD       YMM5, YMM5, [R9+32]
  VADDPD       YMM6, YMM6, [R9+32]
  VADDPD       YMM7, YMM7, [R9+32]
  // p = p*r + c2 .. c11, then +1.0 once more (c11 row reused)
  VFMADD213PD  YMM4, YMM0, [R9+64]
  VFMADD213PD  YMM5, YMM1, [R9+64]
  VFMADD213PD  YMM6, YMM2, [R9+64]
  VFMADD213PD  YMM7, YMM3, [R9+64]
  VFMADD213PD  YMM4, YMM0, [R9+96]
  VFMADD213PD  YMM5, YMM1, [R9+96]
  VFMADD213PD  YMM6, YMM2, [R9+96]
  VFMADD213PD  YMM7, YMM3, [R9+96]
  VFMADD213PD  YMM4, YMM0, [R9+128]
  VFMADD213PD  YMM5, YMM1, [R9+128]
  VFMADD213PD  YMM6, YMM2, [R9+128]
  VFMADD213PD  YMM7, YMM3, [R9+128]
  VFMADD213PD  YMM4, YMM0, [R9+160]
  VFMADD213PD  YMM5, YMM1, [R9+160]
  VFMADD213PD  YMM6, YMM2, [R9+160]
  VFMADD213PD  YMM7, YMM3, [R9+160]
  VFMADD213PD  YMM4, YMM0, [R9+192]
  VFMADD213PD  YMM5, YMM1, [R9+192]
  VFMADD213PD  YMM6, YMM2, [R9+192]
  VFMADD213PD  YMM7, YMM3, [R9+192]
  VFMADD213PD  YMM4, YMM0, [R9+224]
  VFMADD213PD  YMM5, YMM1, [R9+224]
  VFMADD213PD  YMM6, YMM2, [R9+224]
  VFMADD213PD  YMM7, YMM3, [R9+224]
  VFMADD213PD  YMM4, YMM0, [R9+256]
  VFMADD213PD  YMM5, YMM1, [R9+256]
  VFMADD213PD  YMM6, YMM2, [R9+256]
  VFMADD213PD  YMM7, YMM3, [R9+256]
  VFMADD213PD  YMM4, YMM0, [R9+288]
  VFMADD213PD  YMM5, YMM1, [R9+288]
  VFMADD213PD  YMM6, YMM2, [R9+288]
  VFMADD213PD  YMM7, YMM3, [R9+288]
  VFMADD213PD  YMM4, YMM0, [R9+320]
  VFMADD213PD  YMM5, YMM1, [R9+320]
  VFMADD213PD  YMM6, YMM2, [R9+320]
  VFMADD213PD  YMM7, YMM3, [R9+320]
  VFMADD213PD  YMM4, YMM0, [R9+352]
  VFMADD213PD  YMM5, YMM1, [R9+352]
  VFMADD213PD  YMM6, YMM2, [R9+352]
  VFMADD213PD  YMM7, YMM3, [R9+352]
  VFMADD213PD  YMM4, YMM0, [R9+352]
  VFMADD213PD  YMM5, YMM1, [R9+352]
  VFMADD213PD  YMM6, YMM2, [R9+352]
  VFMADD213PD  YMM7, YMM3, [R9+352]
  // 2^n: n -> int32 -> int64, +1023, shift into exponent field (in place)
  VCVTTPD2DQ   XMM8,  YMM8
  VCVTTPD2DQ   XMM9,  YMM9
  VCVTTPD2DQ   XMM10, YMM10
  VCVTTPD2DQ   XMM11, YMM11
  VPMOVSXDQ    YMM8,  XMM8
  VPMOVSXDQ    YMM9,  XMM9
  VPMOVSXDQ    YMM10, XMM10
  VPMOVSXDQ    YMM11, XMM11
  VPADDQ       YMM8,  YMM8,  YMM15
  VPADDQ       YMM9,  YMM9,  YMM15
  VPADDQ       YMM10, YMM10, YMM15
  VPADDQ       YMM11, YMM11, YMM15
  VPSLLQ       YMM8,  YMM8,  52
  VPSLLQ       YMM9,  YMM9,  52
  VPSLLQ       YMM10, YMM10, 52
  VPSLLQ       YMM11, YMM11, 52
  // Res = p * 2^n
  VMULPD       YMM4, YMM4, YMM8
  VMULPD       YMM5, YMM5, YMM9
  VMULPD       YMM6, YMM6, YMM10
  VMULPD       YMM7, YMM7, YMM11
  VMOVUPD      [R8],    YMM4
  VMOVUPD      [R8+32], YMM5
  VMOVUPD      [R8+64], YMM6
  VMOVUPD      [R8+96], YMM7
  ADD   RDX, 128
  ADD   R8,  128
  DEC   EAX
  JNZ   @loop4
@rest:
  // Remaining 0-3 quads, one vector at a time (same pipeline)
  AND   ECX, 3
  JZ    @restore
@loop1:
  VMOVUPD      YMM0, [RDX]
  VMULPD       YMM8, YMM0, YMM12
  VROUNDPD     YMM8, YMM8, 0
  VFNMADD231PD YMM0, YMM8, YMM13
  VFNMADD231PD YMM0, YMM8, YMM14
  VMULPD       YMM4, YMM0, [R9]
  VADDPD       YMM4, YMM4, [R9+32]
  VFMADD213PD  YMM4, YMM0, [R9+64]
  VFMADD213PD  YMM4, YMM0, [R9+96]
  VFMADD213PD  YMM4, YMM0, [R9+128]
  VFMADD213PD  YMM4, YMM0, [R9+160]
  VFMADD213PD  YMM4, YMM0, [R9+192]
  VFMADD213PD  YMM4, YMM0, [R9+224]
  VFMADD213PD  YMM4, YMM0, [R9+256]
  VFMADD213PD  YMM4, YMM0, [R9+288]
  VFMADD213PD  YMM4, YMM0, [R9+320]
  VFMADD213PD  YMM4, YMM0, [R9+352]
  VFMADD213PD  YMM4, YMM0, [R9+352]
  VCVTTPD2DQ   XMM8, YMM8
  VPMOVSXDQ    YMM8, XMM8
  VPADDQ       YMM8, YMM8, YMM15
  VPSLLQ       YMM8, YMM8, 52
  VMULPD       YMM4, YMM4, YMM8
  VMOVUPD      [R8], YMM4
  ADD   RDX, 32
  ADD   R8,  32
  DEC   ECX
  JNZ   @loop1
@restore:
  MOVDQU XMM6,  [RSP+8]
  MOVDQU XMM7,  [RSP+24]
  MOVDQU XMM8,  [RSP+40]
  MOVDQU XMM9,  [RSP+56]
  MOVDQU XMM10, [RSP+72]
  MOVDQU XMM11, [RSP+88]
  MOVDQU XMM12, [RSP+104]
  MOVDQU XMM13, [RSP+120]
  MOVDQU XMM14, [RSP+136]
  MOVDQU XMM15, [RSP+152]
  ADD   RSP, 168
@done:
  VZEROUPPER
end;

// ---------------------------------------------------------------------------
// TVectorMathAVX2 - reduce/recon hooks delegating to the AVX2 asm helpers
// ---------------------------------------------------------------------------

Procedure TVectorMathAVX2.ExpCore(Count: Integer; V, Res: PDouble;
  var Scratch: TScratchBuffer);
// Scratch is unused here: the fused kernel keeps all intermediates in
// registers and the scalar tail (0-3 elements) uses locals only.
var
  I: Integer;
  N: Int64;
begin
  if Count >= 4 then
    AVX2ExpCore4(Count shr 2, V, Res, @EXP_HORNER_C4[0]);
  for I := Count - (Count and 3) to Count - 1 do
  begin
    var R := ExpReduction(V[I], N);
    Res[I] := ExpPolynomial(R) * PowerOfTwo(N);
  end;
end;

// Fused AVX2+FMA ln pipeline: argument reduction, Horner polynomial and the
// nln2 recombination in one loop, 4 vectors (16 doubles) per iteration.
// s, t, nln2 and p never leave registers - the only memory traffic is the
// input load, the result store, and the L1-resident constant table.
//
// Phase 1 (per vector): split x = 2^n*m with m in [sqrt2/2, sqrt2);
//   s = (m-1)/(m+1) -> YMM0-YMM3,  nln2 = n*ln2 -> YMM4-YMM7.
// Phase 2 (4-way interleaved): t = s^2; p = g(t) via Horner from the constant
//   table; result = p*s + nln2 as one FMA - overlapping four dependency
//   chains so the FMA ports stay busy (see AVX2ExpCore4 for the rationale).
//
// Constants and coefficients are hardcoded (IEEE 754 hex - no extra pointer
// parameter, so this works correctly in both Debug and Release builds) and
// broadcast into a stack-built table of 14 rows at entry.
//
// Win64: RCX=QuadCount  RDX=X  R8=Res
// Stack: [RSP+8..167] XMM6-XMM15 saves, [RSP+168..615] constant table.
Procedure AVX2LogCore4(QuadCount: Integer; X, Res: PDouble);
asm
  TEST  ECX, ECX
  JLE   @done
  SUB   RSP, 616
  // Save non-volatile XMM6-XMM15 (Win64 ABI: lower 128-bit halves)
  MOVDQU [RSP+8],   XMM6
  MOVDQU [RSP+24],  XMM7
  MOVDQU [RSP+40],  XMM8
  MOVDQU [RSP+56],  XMM9
  MOVDQU [RSP+72],  XMM10
  MOVDQU [RSP+88],  XMM11
  MOVDQU [RSP+104], XMM12
  MOVDQU [RSP+120], XMM13
  MOVDQU [RSP+136], XMM14
  MOVDQU [RSP+152], XMM15
  // Build the broadcast constant table.
  MOV    RAX, $000FFFFFFFFFFFFF   // [RSP+168] mantissa mask
  VMOVQ  XMM0, RAX
  VPBROADCASTQ YMM0, XMM0
  VMOVUPD [RSP+168], YMM0
  MOV    RAX, $3FF0000000000000   // [RSP+200] 1.0 (exponent OR mask and float)
  VMOVQ  XMM0, RAX
  VPBROADCASTQ YMM0, XMM0
  VMOVUPD [RSP+200], YMM0
  MOV    RAX, 1023                // [RSP+232] exponent bias (int64)
  VMOVQ  XMM0, RAX
  VPBROADCASTQ YMM0, XMM0
  VMOVUPD [RSP+232], YMM0
  MOV    RAX, $3FF6A09E667F3BCD   // [RSP+264] sqrt(2)
  VMOVQ  XMM0, RAX
  VBROADCASTSD YMM0, XMM0
  VMOVUPD [RSP+264], YMM0
  MOV    RAX, $3FE0000000000000   // [RSP+296] 0.5
  VMOVQ  XMM0, RAX
  VBROADCASTSD YMM0, XMM0
  VMOVUPD [RSP+296], YMM0
  MOV    RAX, $3FE62E42FEFA39EF   // [RSP+328] ln(2)
  VMOVQ  XMM0, RAX
  VBROADCASTSD YMM0, XMM0
  VMOVUPD [RSP+328], YMM0
  MOV    RAX, $3FC1111111111111   // [RSP+360] c0 = 2/15
  VMOVQ  XMM0, RAX
  VBROADCASTSD YMM0, XMM0
  VMOVUPD [RSP+360], YMM0
  MOV    RAX, $3FC3B13B13B13B14   // [RSP+392] c1 = 2/13
  VMOVQ  XMM0, RAX
  VBROADCASTSD YMM0, XMM0
  VMOVUPD [RSP+392], YMM0
  MOV    RAX, $3FC745D1745D1746   // [RSP+424] c2 = 2/11
  VMOVQ  XMM0, RAX
  VBROADCASTSD YMM0, XMM0
  VMOVUPD [RSP+424], YMM0
  MOV    RAX, $3FCC71C71C71C71C   // [RSP+456] c3 = 2/9
  VMOVQ  XMM0, RAX
  VBROADCASTSD YMM0, XMM0
  VMOVUPD [RSP+456], YMM0
  MOV    RAX, $3FD2492492492492   // [RSP+488] c4 = 2/7
  VMOVQ  XMM0, RAX
  VBROADCASTSD YMM0, XMM0
  VMOVUPD [RSP+488], YMM0
  MOV    RAX, $3FD999999999999A   // [RSP+520] c5 = 2/5
  VMOVQ  XMM0, RAX
  VBROADCASTSD YMM0, XMM0
  VMOVUPD [RSP+520], YMM0
  MOV    RAX, $3FE5555555555555   // [RSP+552] c6 = 2/3
  VMOVQ  XMM0, RAX
  VBROADCASTSD YMM0, XMM0
  VMOVUPD [RSP+552], YMM0
  MOV    RAX, $4000000000000000   // [RSP+584] c7 = 2.0
  VMOVQ  XMM0, RAX
  VBROADCASTSD YMM0, XMM0
  VMOVUPD [RSP+584], YMM0
  // 4-quad groups (16 doubles per iteration)
  MOV   EAX, ECX
  SHR   EAX, 2
  JZ    @rest
@loop4:
  // ---- Phase 1: reduce vector 0 -> s in YMM0, nln2 in YMM4 (temps YMM12-15)
  VMOVUPD      YMM12, [RDX]
  VPSRLQ       YMM13, YMM12, 52
  VPSUBQ       YMM13, YMM13, [RSP+232]
  VPAND        YMM12, YMM12, [RSP+168]
  VPOR         YMM12, YMM12, [RSP+200]
  VCMPPD       YMM14, YMM12, [RSP+264], 5
  VMULPD       YMM15, YMM12, [RSP+296]
  VBLENDVPD    YMM12, YMM12, YMM15, YMM14
  VPSUBQ       YMM13, YMM13, YMM14
  VSUBPD       YMM15, YMM12, [RSP+200]
  VADDPD       YMM12, YMM12, [RSP+200]
  VDIVPD       YMM0,  YMM15, YMM12
  VEXTRACTF128 XMM14, YMM13, 1
  VSHUFPS      XMM14, XMM13, XMM14, $88
  VCVTDQ2PD    YMM13, XMM14
  VMULPD       YMM4,  YMM13, [RSP+328]
  // ---- vector 1 -> YMM1 / YMM5
  VMOVUPD      YMM12, [RDX+32]
  VPSRLQ       YMM13, YMM12, 52
  VPSUBQ       YMM13, YMM13, [RSP+232]
  VPAND        YMM12, YMM12, [RSP+168]
  VPOR         YMM12, YMM12, [RSP+200]
  VCMPPD       YMM14, YMM12, [RSP+264], 5
  VMULPD       YMM15, YMM12, [RSP+296]
  VBLENDVPD    YMM12, YMM12, YMM15, YMM14
  VPSUBQ       YMM13, YMM13, YMM14
  VSUBPD       YMM15, YMM12, [RSP+200]
  VADDPD       YMM12, YMM12, [RSP+200]
  VDIVPD       YMM1,  YMM15, YMM12
  VEXTRACTF128 XMM14, YMM13, 1
  VSHUFPS      XMM14, XMM13, XMM14, $88
  VCVTDQ2PD    YMM13, XMM14
  VMULPD       YMM5,  YMM13, [RSP+328]
  // ---- vector 2 -> YMM2 / YMM6
  VMOVUPD      YMM12, [RDX+64]
  VPSRLQ       YMM13, YMM12, 52
  VPSUBQ       YMM13, YMM13, [RSP+232]
  VPAND        YMM12, YMM12, [RSP+168]
  VPOR         YMM12, YMM12, [RSP+200]
  VCMPPD       YMM14, YMM12, [RSP+264], 5
  VMULPD       YMM15, YMM12, [RSP+296]
  VBLENDVPD    YMM12, YMM12, YMM15, YMM14
  VPSUBQ       YMM13, YMM13, YMM14
  VSUBPD       YMM15, YMM12, [RSP+200]
  VADDPD       YMM12, YMM12, [RSP+200]
  VDIVPD       YMM2,  YMM15, YMM12
  VEXTRACTF128 XMM14, YMM13, 1
  VSHUFPS      XMM14, XMM13, XMM14, $88
  VCVTDQ2PD    YMM13, XMM14
  VMULPD       YMM6,  YMM13, [RSP+328]
  // ---- vector 3 -> YMM3 / YMM7
  VMOVUPD      YMM12, [RDX+96]
  VPSRLQ       YMM13, YMM12, 52
  VPSUBQ       YMM13, YMM13, [RSP+232]
  VPAND        YMM12, YMM12, [RSP+168]
  VPOR         YMM12, YMM12, [RSP+200]
  VCMPPD       YMM14, YMM12, [RSP+264], 5
  VMULPD       YMM15, YMM12, [RSP+296]
  VBLENDVPD    YMM12, YMM12, YMM15, YMM14
  VPSUBQ       YMM13, YMM13, YMM14
  VSUBPD       YMM15, YMM12, [RSP+200]
  VADDPD       YMM12, YMM12, [RSP+200]
  VDIVPD       YMM3,  YMM15, YMM12
  VEXTRACTF128 XMM14, YMM13, 1
  VSHUFPS      XMM14, XMM13, XMM14, $88
  VCVTDQ2PD    YMM13, XMM14
  VMULPD       YMM7,  YMM13, [RSP+328]
  // ---- Phase 2: t = s^2 (YMM8-11), p = Horner(t) (YMM12-15), 4-way
  VMULPD       YMM8,  YMM0, YMM0
  VMULPD       YMM9,  YMM1, YMM1
  VMULPD       YMM10, YMM2, YMM2
  VMULPD       YMM11, YMM3, YMM3
  VMULPD       YMM12, YMM8,  [RSP+360]
  VMULPD       YMM13, YMM9,  [RSP+360]
  VMULPD       YMM14, YMM10, [RSP+360]
  VMULPD       YMM15, YMM11, [RSP+360]
  VADDPD       YMM12, YMM12, [RSP+392]
  VADDPD       YMM13, YMM13, [RSP+392]
  VADDPD       YMM14, YMM14, [RSP+392]
  VADDPD       YMM15, YMM15, [RSP+392]
  VFMADD213PD  YMM12, YMM8,  [RSP+424]
  VFMADD213PD  YMM13, YMM9,  [RSP+424]
  VFMADD213PD  YMM14, YMM10, [RSP+424]
  VFMADD213PD  YMM15, YMM11, [RSP+424]
  VFMADD213PD  YMM12, YMM8,  [RSP+456]
  VFMADD213PD  YMM13, YMM9,  [RSP+456]
  VFMADD213PD  YMM14, YMM10, [RSP+456]
  VFMADD213PD  YMM15, YMM11, [RSP+456]
  VFMADD213PD  YMM12, YMM8,  [RSP+488]
  VFMADD213PD  YMM13, YMM9,  [RSP+488]
  VFMADD213PD  YMM14, YMM10, [RSP+488]
  VFMADD213PD  YMM15, YMM11, [RSP+488]
  VFMADD213PD  YMM12, YMM8,  [RSP+520]
  VFMADD213PD  YMM13, YMM9,  [RSP+520]
  VFMADD213PD  YMM14, YMM10, [RSP+520]
  VFMADD213PD  YMM15, YMM11, [RSP+520]
  VFMADD213PD  YMM12, YMM8,  [RSP+552]
  VFMADD213PD  YMM13, YMM9,  [RSP+552]
  VFMADD213PD  YMM14, YMM10, [RSP+552]
  VFMADD213PD  YMM15, YMM11, [RSP+552]
  VFMADD213PD  YMM12, YMM8,  [RSP+584]
  VFMADD213PD  YMM13, YMM9,  [RSP+584]
  VFMADD213PD  YMM14, YMM10, [RSP+584]
  VFMADD213PD  YMM15, YMM11, [RSP+584]
  // result = p*s + nln2
  VFMADD213PD  YMM12, YMM0, YMM4
  VFMADD213PD  YMM13, YMM1, YMM5
  VFMADD213PD  YMM14, YMM2, YMM6
  VFMADD213PD  YMM15, YMM3, YMM7
  VMOVUPD      [R8],    YMM12
  VMOVUPD      [R8+32], YMM13
  VMOVUPD      [R8+64], YMM14
  VMOVUPD      [R8+96], YMM15
  ADD   RDX, 128
  ADD   R8,  128
  DEC   EAX
  JNZ   @loop4
@rest:
  // Remaining 0-3 quads, one vector at a time (same pipeline)
  AND   ECX, 3
  JZ    @restore
@loop1:
  VMOVUPD      YMM12, [RDX]
  VPSRLQ       YMM13, YMM12, 52
  VPSUBQ       YMM13, YMM13, [RSP+232]
  VPAND        YMM12, YMM12, [RSP+168]
  VPOR         YMM12, YMM12, [RSP+200]
  VCMPPD       YMM14, YMM12, [RSP+264], 5
  VMULPD       YMM15, YMM12, [RSP+296]
  VBLENDVPD    YMM12, YMM12, YMM15, YMM14
  VPSUBQ       YMM13, YMM13, YMM14
  VSUBPD       YMM15, YMM12, [RSP+200]
  VADDPD       YMM12, YMM12, [RSP+200]
  VDIVPD       YMM0,  YMM15, YMM12
  VEXTRACTF128 XMM14, YMM13, 1
  VSHUFPS      XMM14, XMM13, XMM14, $88
  VCVTDQ2PD    YMM13, XMM14
  VMULPD       YMM4,  YMM13, [RSP+328]
  VMULPD       YMM8,  YMM0, YMM0
  VMULPD       YMM12, YMM8, [RSP+360]
  VADDPD       YMM12, YMM12, [RSP+392]
  VFMADD213PD  YMM12, YMM8, [RSP+424]
  VFMADD213PD  YMM12, YMM8, [RSP+456]
  VFMADD213PD  YMM12, YMM8, [RSP+488]
  VFMADD213PD  YMM12, YMM8, [RSP+520]
  VFMADD213PD  YMM12, YMM8, [RSP+552]
  VFMADD213PD  YMM12, YMM8, [RSP+584]
  VFMADD213PD  YMM12, YMM0, YMM4
  VMOVUPD      [R8], YMM12
  ADD   RDX, 32
  ADD   R8,  32
  DEC   ECX
  JNZ   @loop1
@restore:
  MOVDQU XMM6,  [RSP+8]
  MOVDQU XMM7,  [RSP+24]
  MOVDQU XMM8,  [RSP+40]
  MOVDQU XMM9,  [RSP+56]
  MOVDQU XMM10, [RSP+72]
  MOVDQU XMM11, [RSP+88]
  MOVDQU XMM12, [RSP+104]
  MOVDQU XMM13, [RSP+120]
  MOVDQU XMM14, [RSP+136]
  MOVDQU XMM15, [RSP+152]
  ADD   RSP, 616
@done:
  VZEROUPPER
end;

Procedure TVectorMathAVX2.LogCore(Count: Integer; X, Res, S, NLn2: PDouble);
// S and NLn2 are unused here: the fused kernel keeps all intermediates in
// registers and the scalar tail (0-3 elements) uses locals only.
var
  I: Integer;
  TailNLn2: Double;
begin
  if Count >= 4 then
    AVX2LogCore4(Count shr 2, X, Res);
  for I := Count - (Count and 3) to Count - 1 do
  begin
    var TailS := LogReduction(X[I], TailNLn2);
    Res[I] := LogPolynomial(TailS) + TailNLn2;
  end;
end;

// ---------------------------------------------------------------------------
// AVX2+FMA helper for ScaleAddBlock: V[i] += Scalar * Src[i]
// ---------------------------------------------------------------------------
// Uses VFMADD231PD: dst = src1*src2 + dst  ->  V = Scalar * Src + V
// Scalar is broadcast into YMM2 before the loop; YMM0/YMM1 are loop temps.
// YMM0-YMM2 are all caller-saved (volatile) - no register save needed.
//
// Win64: XMM0=Scalar  RDX=QuadCount  R8=Src  R9=V
Procedure AVX2FmaSVIP(Scalar: Double; QuadCount: Integer; Src, V: PDouble);
asm
  TEST  EDX, EDX
  JLE   @done
  VBROADCASTSD YMM2, XMM0       // YMM2 = [Scalar, Scalar, Scalar, Scalar]
@loop:
  VMOVUPD      YMM0, [R8]       // Src[0..3]
  VMOVUPD      YMM1, [R9]       // V[0..3]
  VFMADD231PD  YMM1, YMM2, YMM0 // V = Scalar * Src + V
  VMOVUPD      [R9], YMM1
  ADD   R8, 32
  ADD   R9, 32
  DEC   EDX
  JNZ   @loop
@done:
  VZEROUPPER
end;

// AVX2+FMA weighted sum: Sum Data[i] * Weights[i], using a memory operand for
// Weights so each quad is one load + one FMA rather than two loads + mul + add.
// VFMADD231PD dst,src1,mem: dst = dst + src1 * mem  ->  acc += Data * Weights
// Four accumulators break the FMA latency chain (see AVXSumV).
// Win64: RCX=QuadCount  RDX=Data  R8=Weights  -> result in XMM0
function AVX2WeightedSumV(QuadCount: Integer; Data, Weights: PDouble): Double;
asm
  VXORPD    YMM0, YMM0, YMM0
  VXORPD    YMM1, YMM1, YMM1
  VXORPD    YMM2, YMM2, YMM2
  VXORPD    YMM3, YMM3, YMM3
  // 4-quad groups (16 doubles per iteration)
  MOV       EAX, ECX
  SHR       EAX, 2
  JZ        @rest
@loop4:
  VMOVUPD      YMM4, [RDX]
  VFMADD231PD  YMM0, YMM4, [R8]
  VMOVUPD      YMM5, [RDX+32]
  VFMADD231PD  YMM1, YMM5, [R8+32]
  VMOVUPD      YMM4, [RDX+64]
  VFMADD231PD  YMM2, YMM4, [R8+64]
  VMOVUPD      YMM5, [RDX+96]
  VFMADD231PD  YMM3, YMM5, [R8+96]
  ADD       RDX, 128
  ADD       R8,  128
  DEC       EAX
  JNZ       @loop4
@rest:
  // Remaining 0-3 quads
  AND       ECX, 3
  JZ        @reduce
@loop1:
  VMOVUPD      YMM4, [RDX]
  VFMADD231PD  YMM0, YMM4, [R8]
  ADD       RDX, 32
  ADD       R8,  32
  DEC       ECX
  JNZ       @loop1
@reduce:
  VADDPD    YMM0, YMM0, YMM1
  VADDPD    YMM2, YMM2, YMM3
  VADDPD    YMM0, YMM0, YMM2
  VEXTRACTF128 XMM1, YMM0, 1
  VADDPD    XMM0, XMM0, XMM1
  VUNPCKHPD XMM1, XMM0, XMM0
  VADDSD    XMM0, XMM0, XMM1
  VZEROUPPER
end;

function TVectorMathAVX2.WeightedSumBlock(Data, Weights: PDouble; Count: Integer): Double;
var Tail, I: Integer;
begin
  Tail := Count and 3;
  if Count >= 4 then Result := AVX2WeightedSumV(Count shr 2, Data, Weights)
  else Result := 0.0;
  for I := Count - Tail to Count - 1 do
    Result := Result + Data[I] * Weights[I];
end;

Procedure TVectorMathAVX2.ScaleAddBlock(Scale: Double; Count: Integer; Src, Dest: PDouble);
// Dest[i] += Scale * Src[i] via AVX2 FMA: VFMADD231PD V, Scalar, Src
var Tail, I: Integer;
begin
  Tail := Count and 3;
  if Count >= 4 then AVX2FmaSVIP(Scale, Count shr 2, Src, Dest);
  for I := Count - Tail to Count - 1 do
    Dest[I] := Dest[I] + Scale * Src[I];
end;

{$ENDIF}

end.
