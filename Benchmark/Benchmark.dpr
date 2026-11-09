program Benchmark;

////////////////////////////////////////////////////////////////////////////////
//
// Author: Jaap Baak (initial implementation developed with Claude Fable 5)
// https://github.com/transportmodelling/VecMath
//
////////////////////////////////////////////////////////////////////////////////

{$APPTYPE CONSOLE}

{$R *.res}

uses
  System.SysUtils,
  System.Diagnostics,
  VecMath.CPU in '..\Source\VecMath.CPU.pas',
  VecMath in '..\Source\VecMath.pas',
  VecMath.SSE2 in '..\Source\VecMath.SSE2.pas',
  VecMath.AVX in '..\Source\VecMath.AVX.pas';

type
  TBenchProc    = reference to Procedure;

const
  WARMUP_ITERS = 5;
  MIN_ITERS    = 3;
  MIN_MS       = 50;

Function BenchmarkUs(const Op: TBenchProc): Double;
var
  SW: TStopwatch;
  I, Iters: Integer;
begin
  for I := 1 to WARMUP_ITERS do Op;

  Iters := 0;
  SW := TStopwatch.StartNew;
  repeat
    Op;
    Inc(Iters);
  until (Iters >= MIN_ITERS) and (SW.ElapsedMilliseconds >= MIN_MS);
  SW.Stop;

  Result := (SW.Elapsed.TotalMilliseconds * 1000.0) / Iters;
end;

Procedure WarmupCPU;
// Spin for 500 ms to bring the CPU to its sustained Turbo Boost frequency
// before any measurements are taken.  The accumulator is printed so the
// compiler cannot eliminate the loop.
var
  SW: TStopwatch;
  X:  Double;
  I:  Integer;
begin
  Writeln;
  Write('Warming up CPU... ');
  Flush(Output);
  X := 1.0;
  SW := TStopwatch.StartNew;
  repeat
    for I := 1 to 10000 do
      X := X * 1.0000001 + 0.0000001;
  until SW.ElapsedMilliseconds >= 500;
  Writeln('done (sink=', X:0:4, ')');
  Flush(Output);
end;

Procedure FillSafe(const A: TArray<Double>; Lo, Hi: Double);
var I: Integer;
begin
  for I := 0 to High(A) do
    A[I] := Lo + (Hi - Lo) * (I mod 1009) / 1008.0;
end;

Procedure PrintHeader(const Size: Integer);
begin
  Writeln;
  Writeln('=== Vector size: ', Size, ' elements ===');
  Writeln(Format('%-22s %12s %12s %12s %12s %10s %10s %10s',
                 ['Operation', 'Scalar(us)', 'SSE2(us)', 'AVX(us)', 'AVX2(us)',
                  'SSE2 gain', 'AVX gain', 'AVX2 gain']));
  Writeln(StringOfChar('-', 102));
  Flush(Output);
end;

Procedure PrintRow(const Name: string; ScUs, SSE2Us, AVXUs, AVX2Us: Double);
begin
  Writeln(Format('%-22s %12.2f %12.2f %12.2f %12.2f %9.2fx %9.2fx %9.2fx',
                 [Name, ScUs, SSE2Us, AVXUs, AVX2Us,
                  ScUs / SSE2Us, ScUs / AVXUs, ScUs / AVX2Us]));
  Flush(Output);
end;

// Native (System.Exp / System.Ln) row - shows time in Scalar column,
// dashes for SIMD columns, and the ratio vs AVX2 in the last gain column.
Procedure PrintNativeRow(const Name: string; NatUs, AVX2Us: Double);
begin
  Writeln(Format('%-22s %12.2f %12s %12s %12s %10s %10s %9.2fx*',
                 [Name, NatUs, '-', '-', '-', '-', '-', NatUs / AVX2Us]));
  Flush(Output);
end;

Procedure NativeExp(const Src, Dst: TArray<Double>);
var S, D: PDouble; N: Integer;
begin
  N := Length(Src); S := Pointer(Src); D := Pointer(Dst);
  while N > 0 do begin D^ := Exp(S^); Inc(S); Inc(D); Dec(N); end;
end;

Procedure NativeLn(const Src, Dst: TArray<Double>);
var S, D: PDouble; N: Integer;
begin
  N := Length(Src); S := Pointer(Src); D := Pointer(Dst);
  while N > 0 do begin D^ := Ln(S^); Inc(S); Inc(D); Dec(N); end;
end;

var
  // The scratch buffers of the engines, reused over the array sizes. The
  // benchmark closures use them as variables of the program: an anonymous
  // method cannot capture a var parameter.
  ScScratch, S2Scratch, AvScratch, Av2Scratch: TScratchBuffer;

Procedure RunBenchmarks(Size: Integer;
                        Sc: TVectorMath; S2: TVectorMathSSE2; Av: TVectorMathAVX; Av2: TVectorMathAVX2);
var
  A, B, C: TArray<Double>;
  ScUs, S2Us, AvUs, Av2Us: Double;
  ExpSum, NatExpUs, NatLnUs: Double;
begin
  SetLength(A, Size);
  SetLength(B, Size);
  SetLength(C, Size);
  FillSafe(A, 0.1, 0.9);
  FillSafe(B, 0.1, 0.9);

  PrintHeader(Size);

  // Add vector + scalar
  ScUs  := BenchmarkUs(Procedure begin Sc.Add(2.0, A, C) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Add(2.0, A, C) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Add(2.0, A, C) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Add(2.0, A, C) end);
  PrintRow('Add(scalar, V)', ScUs, S2Us, AvUs, Av2Us);

  // Add vector + vector
  ScUs  := BenchmarkUs(Procedure begin Sc.Add(A, B, C) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Add(A, B, C) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Add(A, B, C) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Add(A, B, C) end);
  PrintRow('Add(V, V)', ScUs, S2Us, AvUs, Av2Us);

  // Subtract vector - scalar
  ScUs  := BenchmarkUs(Procedure begin Sc.Subtract(0.05, A, C) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Subtract(0.05, A, C) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Subtract(0.05, A, C) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Subtract(0.05, A, C) end);
  PrintRow('Subtract(scalar, V)', ScUs, S2Us, AvUs, Av2Us);

  // Subtract vector - vector
  ScUs  := BenchmarkUs(Procedure begin Sc.Subtract(A, B, C) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Subtract(A, B, C) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Subtract(A, B, C) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Subtract(A, B, C) end);
  PrintRow('Subtract(V, V)', ScUs, S2Us, AvUs, Av2Us);

  // Multiply vector * scalar
  ScUs  := BenchmarkUs(Procedure begin Sc.Multiply(3.0, A, C) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Multiply(3.0, A, C) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Multiply(3.0, A, C) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Multiply(3.0, A, C) end);
  PrintRow('Multiply(scalar, V)', ScUs, S2Us, AvUs, Av2Us);

  // Multiply vector * vector
  ScUs  := BenchmarkUs(Procedure begin Sc.Multiply(A, B, C) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Multiply(A, B, C) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Multiply(A, B, C) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Multiply(A, B, C) end);
  PrintRow('Multiply(V, V)', ScUs, S2Us, AvUs, Av2Us);

  // Divide vector / scalar
  ScUs  := BenchmarkUs(Procedure begin Sc.Divide(3.0, A, C) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Divide(3.0, A, C) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Divide(3.0, A, C) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Divide(3.0, A, C) end);
  PrintRow('Divide(scalar, V)', ScUs, S2Us, AvUs, Av2Us);

  // Divide vector / vector
  ScUs  := BenchmarkUs(Procedure begin Sc.Divide(A, B, C) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Divide(A, B, C) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Divide(A, B, C) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Divide(A, B, C) end);
  PrintRow('Divide(V, V)', ScUs, S2Us, AvUs, Av2Us);

  // Sqrt
  ScUs  := BenchmarkUs(Procedure begin Sc.Sqrt(A, C) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Sqrt(A, C) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Sqrt(A, C) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Sqrt(A, C) end);
  PrintRow('Sqrt', ScUs, S2Us, AvUs, Av2Us);

  // Sum (reduction - result is a scalar)
  ScUs  := BenchmarkUs(Procedure begin Sc.Sum(A) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Sum(A) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Sum(A) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Sum(A) end);
  PrintRow('Sum', ScUs, S2Us, AvUs, Av2Us);

  // Exp (polynomial via base class)
  ScUs  := BenchmarkUs(Procedure begin Sc.Exp(A, C) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Exp(A, C) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Exp(A, C) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Exp(A, C) end);
  NatExpUs := BenchmarkUs(Procedure begin NativeExp(A, C) end);
  PrintNativeRow('Exp (System.Exp)', NatExpUs, Av2Us);
  PrintRow('Exp', ScUs, S2Us, AvUs, Av2Us);

  // Exp with reused scratch buffer
  ScUs  := BenchmarkUs(Procedure begin Sc.Exp(A, C, ScScratch) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Exp(A, C, S2Scratch) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Exp(A, C, AvScratch) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Exp(A, C, Av2Scratch) end);
  PrintRow('Exp (scratch)', ScUs, S2Us, AvUs, Av2Us);

  // Fused Exp + Sum
  ScUs  := BenchmarkUs(Procedure begin Sc.Exp(A, C, ExpSum, ScScratch) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Exp(A, C, ExpSum, S2Scratch) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Exp(A, C, ExpSum, AvScratch) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Exp(A, C, ExpSum, Av2Scratch) end);
  PrintRow('Exp+Sum (fused)', ScUs, S2Us, AvUs, Av2Us);

  // Log (polynomial via base class)
  ScUs  := BenchmarkUs(Procedure begin Sc.Log(A, C) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Log(A, C) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Log(A, C) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Log(A, C) end);
  NatLnUs := BenchmarkUs(Procedure begin NativeLn(A, C) end);
  PrintNativeRow('Log (System.Ln)', NatLnUs, Av2Us);
  PrintRow('Log', ScUs, S2Us, AvUs, Av2Us);

  // Log with reused scratch buffer
  ScUs  := BenchmarkUs(Procedure begin Sc.Log(A, C, ScScratch) end);
  S2Us  := BenchmarkUs(Procedure begin S2.Log(A, C, S2Scratch) end);
  AvUs  := BenchmarkUs(Procedure begin Av.Log(A, C, AvScratch) end);
  Av2Us := BenchmarkUs(Procedure begin Av2.Log(A, C, Av2Scratch) end);
  PrintRow('Log (scratch)', ScUs, S2Us, AvUs, Av2Us);

  Writeln;
  Writeln('* last gain column = native / AVX2: how many times slower System.Exp/Ln is vs AVX2 polynomial');
  Flush(Output);
end;

var
  F:          TCpuFeatures;
  Sc:         TVectorMath;
  S2:         TVectorMathSSE2;
  Av:         TVectorMathAVX;
  Av2:        TVectorMathAVX2;

begin
  try
    F := GetCpuFeatures;
    Writeln('CPU features:');
    Writeln('  SSE2 : ', F.HasSSE2);
    Writeln('  AVX  : ', F.HasAVX);
    Writeln('  AVX2 : ', F.HasAVX2);
    Writeln('  Selected engine: ', Engine.ClassName);
    Flush(Output);

    WarmupCPU;

    Sc   := TVectorMath.Create;
    S2   := TVectorMathSSE2.Create;
    Av   := TVectorMathAVX.Create;
    Av2  := TVectorMathAVX2.Create;
    try
      RunBenchmarks(        1_000, Sc, S2, Av, Av2);
      RunBenchmarks(       10_000, Sc, S2, Av, Av2);
      RunBenchmarks(      100_000, Sc, S2, Av, Av2);
      RunBenchmarks(    1_000_000, Sc, S2, Av, Av2);
    finally
      Sc.Free;
      S2.Free;
      Av.Free;
      Av2.Free;
    end;

    Writeln;
    Write('Done. Press Enter...');
    Readln;
  except
    on E: Exception do
      Writeln(E.ClassName, ': ', E.Message);
  end;
end.
