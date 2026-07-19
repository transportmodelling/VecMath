program TestVecMath;

////////////////////////////////////////////////////////////////////////////////
//
// Author: Jaap Baak (initial implementation developed with Claude Fable 5)
// https://github.com/transportmodelling/VecMath
//
////////////////////////////////////////////////////////////////////////////////

{$APPTYPE CONSOLE}
{$STRONGLINKTYPES ON}

uses
  System.SysUtils,
  DUnitX.Loggers.Console,
  DUnitX.Loggers.Xml.JUnit,
  DUnitX.TestFramework,
  VecMath.CPU in '..\Source\VecMath.CPU.pas',
  VecMath in '..\Source\VecMath.pas',
  VecMath.SSE2 in '..\Source\VecMath.SSE2.pas',
  VecMath.AVX in '..\Source\VecMath.AVX.pas',
  VecMath.Tests in 'VecMath.Tests.pas';

begin
  try
    // Initialization
    FormatSettings.DecimalSeparator := '.';
    ReportMemoryLeaksOnShutdown := True;
    TDUnitX.CheckCommandLine;
    TDUnitX.Options.XMLOutputFile := '.\TestVecMath.xml';
    // Create the test Runner
    var Runner := TDUnitX.CreateRunner;
    Runner.UseRTTI := True;
    Runner.FailsOnNoAsserts := False;
    // Create loggers
    var ConsoleLogger := TDUnitXConsoleLogger.Create(false);
    var JunitLogger := TDUnitXXMLJUnitFileLogger.Create(TDUnitX.Options.XMLOutputFile);
    Runner.AddLogger(ConsoleLogger);
    Runner.AddLogger(JunitLogger);
    //Run tests
    var Results := Runner.Execute;
    if not Results.AllPassed then System.ExitCode := EXIT_ERRORS;
  except
    on E: Exception do
      System.Writeln(E.ClassName, ': ', E.Message);
  end;
end.
