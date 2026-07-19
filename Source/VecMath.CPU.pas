unit VecMath.CPU;

////////////////////////////////////////////////////////////////////////////////
//
// Author: Jaap Baak (initial implementation developed with Claude Fable 5)
// https://github.com/transportmodelling/VecMath
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
interface
////////////////////////////////////////////////////////////////////////////////

type
  TCpuFeatures = record
    HasSSE2: Boolean;
    HasAVX:  Boolean;
    HasAVX2: Boolean;
  end;

function GetCpuFeatures: TCpuFeatures;

////////////////////////////////////////////////////////////////////////////////
implementation
////////////////////////////////////////////////////////////////////////////////

// Inline asm is only supported by the Windows compilers, so feature detection
// is guarded on MSWINDOWS+CPUX64 like the SIMD units.  On all other platforms
// GetCpuFeatures reports no SIMD support and CreateOptimal falls back to the
// pure-Pascal TVectorMath implementations.
{$IF DEFINED(MSWINDOWS) AND DEFINED(CPUX64)}

type
  TCPUIDResult = record
    EAX, EBX, ECX, EDX: Cardinal;
  end;

Procedure ExecuteCPUID(Leaf: Cardinal; out Res: TCPUIDResult);
asm
  PUSH  RBX
  MOV   R8,  RDX        // save output pointer - RDX is clobbered by CPUID
  MOV   EAX, ECX        // Leaf -> EAX (first param is in RCX on Win64)
  XOR   ECX, ECX        // sub-leaf = 0
  CPUID
  MOV   [R8],    EAX
  MOV   [R8+4],  EBX
  MOV   [R8+8],  ECX
  MOV   [R8+12], EDX
  POP   RBX
end;

function GetXCR0Low: Cardinal;
asm
  XOR ECX, ECX           // XCR register index 0
  XGETBV                 // result lo32 -> EAX (returned), hi32 -> EDX (ignored)
end;

{$ENDIF}

function GetCpuFeatures: TCpuFeatures;
{$IF DEFINED(MSWINDOWS) AND DEFINED(CPUX64)}
const
  EDX_SSE2    = 1 shl 26;
  ECX_OSXSAVE = 1 shl 27;  // OS has enabled XSAVE - required before XGETBV
  ECX_AVX     = 1 shl 28;
  EBX_AVX2    = 1 shl 5;
  XCR0_YMM   = $6;          // bits 1+2: OS saves XMM and YMM state
var
  R:    TCPUIDResult;
  XCR0: Cardinal;
{$ENDIF}
begin
  Result := Default(TCpuFeatures);
  {$IF DEFINED(MSWINDOWS) AND DEFINED(CPUX64)}
  ExecuteCPUID(1, R);
  Result.HasSSE2 := (R.EDX and EDX_SSE2) <> 0;

  if ((R.ECX and ECX_OSXSAVE) <> 0) and ((R.ECX and ECX_AVX) <> 0) then
  begin
    XCR0 := GetXCR0Low;
    Result.HasAVX := (XCR0 and XCR0_YMM) = XCR0_YMM;
  end;

  if Result.HasAVX then
  begin
    ExecuteCPUID(7, R);
    Result.HasAVX2 := (R.EBX and EBX_AVX2) <> 0;
  end;
  {$ENDIF}
end;

end.
