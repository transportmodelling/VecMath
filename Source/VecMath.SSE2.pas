unit VecMath.SSE2;

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
  TVectorMathSSE2 = class(TVectorMath)
  {$IF DEFINED(MSWINDOWS) AND DEFINED(CPUX64)}
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
    function  Sum     (const Vector: TArray<Double>): Double; override;
    function  SumBlock(Data: PDouble; Count: Integer): Double; override;
  {$ENDIF}
  end;

////////////////////////////////////////////////////////////////////////////////
implementation
////////////////////////////////////////////////////////////////////////////////

{$POINTERMATH ON}
{$IF DEFINED(MSWINDOWS) AND DEFINED(CPUX64)}

// Win64 calling convention:
//   Integer/pointer params : RCX, RDX, R8, R9  (left to right)
//   Double params          : XMM0, XMM1, XMM2, XMM3 (same positional slots)
// Each helper processes PairCount pairs of doubles (2 doubles per XMM register).
// The Pascal wrappers handle the odd tail element with a scalar operation.

// ---- vector op vector -------------------------------------------------------

Procedure SSE2VAddV(PairCount: Integer; L, R, Res: PDouble);
// RCX=PairCount  RDX=L  R8=R  R9=Res
asm
  TEST  ECX, ECX
  JLE   @done
@loop:
  MOVUPD  XMM0, [RDX]
  MOVUPD  XMM1, [R8]
  ADDPD   XMM0, XMM1
  MOVUPD  [R9], XMM0
  ADD     RDX, 16
  ADD     R8,  16
  ADD     R9,  16
  DEC     ECX
  JNZ     @loop
@done:
end;

Procedure SSE2VSubV(PairCount: Integer; L, R, Res: PDouble);
asm
  TEST  ECX, ECX
  JLE   @done
@loop:
  MOVUPD  XMM0, [RDX]
  MOVUPD  XMM1, [R8]
  SUBPD   XMM0, XMM1
  MOVUPD  [R9], XMM0
  ADD     RDX, 16
  ADD     R8,  16
  ADD     R9,  16
  DEC     ECX
  JNZ     @loop
@done:
end;

Procedure SSE2VMulV(PairCount: Integer; L, R, Res: PDouble);
asm
  TEST  ECX, ECX
  JLE   @done
@loop:
  MOVUPD  XMM0, [RDX]
  MOVUPD  XMM1, [R8]
  MULPD   XMM0, XMM1
  MOVUPD  [R9], XMM0
  ADD     RDX, 16
  ADD     R8,  16
  ADD     R9,  16
  DEC     ECX
  JNZ     @loop
@done:
end;

Procedure SSE2VDivV(PairCount: Integer; L, R, Res: PDouble);
asm
  TEST  ECX, ECX
  JLE   @done
@loop:
  MOVUPD  XMM0, [RDX]
  MOVUPD  XMM1, [R8]
  DIVPD   XMM0, XMM1
  MOVUPD  [R9], XMM0
  ADD     RDX, 16
  ADD     R8,  16
  ADD     R9,  16
  DEC     ECX
  JNZ     @loop
@done:
end;

Procedure SSE2VSqrt(PairCount: Integer; V, Res: PDouble);
// RCX=PairCount  RDX=V  R8=Res
asm
  TEST  ECX, ECX
  JLE   @done
@loop:
  MOVUPD  XMM0, [RDX]
  SQRTPD  XMM0, XMM0
  MOVUPD  [R8], XMM0
  ADD     RDX, 16
  ADD     R8,  16
  DEC     ECX
  JNZ     @loop
@done:
end;

// ---- scalar op vector -------------------------------------------------------
// Scalar arrives in XMM0 (position 0 is float).
// PairCount in RDX, V in R8, Res in R9.

Procedure SSE2SAddV(Scalar: Double; PairCount: Integer; V, Res: PDouble);
// XMM0=Scalar  RDX=PairCount  R8=V  R9=Res
asm
  TEST    EDX, EDX
  JLE     @done
  SHUFPD  XMM0, XMM0, 0     // broadcast: XMM0 = [s, s]
@loop:
  MOVUPD  XMM1, [R8]
  ADDPD   XMM1, XMM0
  MOVUPD  [R9], XMM1
  ADD     R8,  16
  ADD     R9,  16
  DEC     EDX
  JNZ     @loop
@done:
end;

Procedure SSE2SSubV(Scalar: Double; PairCount: Integer; V, Res: PDouble);
asm
  TEST    EDX, EDX
  JLE     @done
  SHUFPD  XMM0, XMM0, 0
@loop:
  MOVUPD  XMM1, [R8]
  SUBPD   XMM1, XMM0
  MOVUPD  [R9], XMM1
  ADD     R8,  16
  ADD     R9,  16
  DEC     EDX
  JNZ     @loop
@done:
end;

Procedure SSE2SMulV(Scalar: Double; PairCount: Integer; V, Res: PDouble);
asm
  TEST    EDX, EDX
  JLE     @done
  SHUFPD  XMM0, XMM0, 0
@loop:
  MOVUPD  XMM1, [R8]
  MULPD   XMM1, XMM0
  MOVUPD  [R9], XMM1
  ADD     R8,  16
  ADD     R9,  16
  DEC     EDX
  JNZ     @loop
@done:
end;

// ---- horizontal sum ---------------------------------------------------------

function SSE2SumV(PairCount: Integer; V: PDouble): Double;
// RCX=PairCount  RDX=V  -> result in XMM0
//
// Four accumulators (XMM0-XMM3) break the ADDPD latency chain: with a
// single accumulator each add waits ~4 cycles for the previous one;
// four independent chains keep the add ports fed from L1.
asm
  XORPD    XMM0, XMM0
  XORPD    XMM1, XMM1
  XORPD    XMM2, XMM2
  XORPD    XMM3, XMM3
  // 4-pair groups (8 doubles per iteration)
  MOV      EAX, ECX
  SHR      EAX, 2
  JZ       @rest
@loop4:
  MOVUPD   XMM4, [RDX]
  ADDPD    XMM0, XMM4
  MOVUPD   XMM5, [RDX+16]
  ADDPD    XMM1, XMM5
  MOVUPD   XMM4, [RDX+32]
  ADDPD    XMM2, XMM4
  MOVUPD   XMM5, [RDX+48]
  ADDPD    XMM3, XMM5
  ADD      RDX, 64
  DEC      EAX
  JNZ      @loop4
@rest:
  // Remaining 0-3 pairs
  AND      ECX, 3
  JZ       @reduce
@loop1:
  MOVUPD   XMM4, [RDX]
  ADDPD    XMM0, XMM4
  ADD      RDX, 16
  DEC      ECX
  JNZ      @loop1
@reduce:
  ADDPD    XMM0, XMM1
  ADDPD    XMM2, XMM3
  ADDPD    XMM0, XMM2
  // fold high lane into low: XMM0[0] = XMM0[0] + XMM0[1]
  MOVAPD   XMM1, XMM0
  UNPCKHPD XMM1, XMM1    // XMM1[0] = XMM0[1]
  ADDSD    XMM0, XMM1
end;

// ---- method wrappers --------------------------------------------------------

Procedure TVectorMathSSE2.Add(const Scalar: Double; const Vector, Result: TArray<Double>);
var Len: Integer;
begin
  Len := Length(Vector);
  if Len >= 2 then SSE2SAddV(Scalar, Len shr 1, @Vector[0], @Result[0]);
  if (Len and 1) = 1 then Result[Len-1] := Vector[Len-1] + Scalar;
end;

Procedure TVectorMathSSE2.Add(const Left, Right, Result: TArray<Double>);
var Len: Integer;
begin
  Len := Length(Left);
  if Len >= 2 then SSE2VAddV(Len shr 1, @Left[0], @Right[0], @Result[0]);
  if (Len and 1) = 1 then Result[Len-1] := Left[Len-1] + Right[Len-1];
end;

Procedure TVectorMathSSE2.Subtract(const Scalar: Double; const Vector, Result: TArray<Double>);
var Len: Integer;
begin
  Len := Length(Vector);
  if Len >= 2 then SSE2SSubV(Scalar, Len shr 1, @Vector[0], @Result[0]);
  if (Len and 1) = 1 then Result[Len-1] := Vector[Len-1] - Scalar;
end;

Procedure TVectorMathSSE2.Subtract(const Left, Right, Result: TArray<Double>);
var Len: Integer;
begin
  Len := Length(Left);
  if Len >= 2 then SSE2VSubV(Len shr 1, @Left[0], @Right[0], @Result[0]);
  if (Len and 1) = 1 then Result[Len-1] := Left[Len-1] - Right[Len-1];
end;

Procedure TVectorMathSSE2.Multiply(const Scalar: Double; const Vector, Result: TArray<Double>);
var Len: Integer;
begin
  Len := Length(Vector);
  if Len >= 2 then SSE2SMulV(Scalar, Len shr 1, @Vector[0], @Result[0]);
  if (Len and 1) = 1 then Result[Len-1] := Vector[Len-1] * Scalar;
end;

Procedure TVectorMathSSE2.Multiply(const Left, Right, Result: TArray<Double>);
var Len: Integer;
begin
  Len := Length(Left);
  if Len >= 2 then SSE2VMulV(Len shr 1, @Left[0], @Right[0], @Result[0]);
  if (Len and 1) = 1 then Result[Len-1] := Left[Len-1] * Right[Len-1];
end;

Procedure TVectorMathSSE2.Divide(const Scalar: Double; const Vector, Result: TArray<Double>);
var Len: Integer;
begin
  Len := Length(Vector);
  if Len >= 2 then SSE2SMulV(1.0 / Scalar, Len shr 1, @Vector[0], @Result[0]);
  // Tail multiplies by the reciprocal like the SIMD body and the scalar base
  // class, so all elements (and all engines) round identically.
  if (Len and 1) = 1 then Result[Len-1] := Vector[Len-1] * (1.0 / Scalar);
end;

Procedure TVectorMathSSE2.Divide(const Left, Right, Result: TArray<Double>);
var Len: Integer;
begin
  Len := Length(Left);
  if Len >= 2 then SSE2VDivV(Len shr 1, @Left[0], @Right[0], @Result[0]);
  if (Len and 1) = 1 then Result[Len-1] := Left[Len-1] / Right[Len-1];
end;

Procedure TVectorMathSSE2.Sqrt(const Vector, Result: TArray<Double>);
var Len: Integer;
begin
  Len := Length(Vector);
  if Len >= 2 then SSE2VSqrt(Len shr 1, @Vector[0], @Result[0]);
  if (Len and 1) = 1 then Result[Len-1] := System.Sqrt(Vector[Len-1]);
end;

function TVectorMathSSE2.Sum(const Vector: TArray<Double>): Double;
var Len: Integer;
begin
  Len := Length(Vector);
  if Len >= 2 then
    Result := SSE2SumV(Len shr 1, @Vector[0])
  else
    Result := 0.0;
  if (Len and 1) = 1 then Result := Result + Vector[Len - 1];
end;

function TVectorMathSSE2.SumBlock(Data: PDouble; Count: Integer): Double;
begin
  if Count >= 2 then Result := SSE2SumV(Count shr 1, Data)
  else Result := 0.0;
  if (Count and 1) = 1 then Result := Result + Data[Count - 1];
end;

{$ENDIF}

end.
