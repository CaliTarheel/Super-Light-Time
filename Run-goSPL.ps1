<#
.SYNOPSIS
Run a Deep Time goSPL export in the dedicated DeepTime-goSPL WSL distribution.
.DESCRIPTION
Select input.yml for landscape evolution or forcing-check.yml for forcing review.
Keep the selected YAML file and its adjacent input folder together after extracting
the export ZIP. The input folder contains the mesh and all interval forcing files.
The launcher changes to the YAML's directory so these relative paths resolve.
It does not install software or change your WSL default distribution.
.PARAMETER InputFile
An exported YAML configuration, or a directory containing input.yml. When omitted,
a Windows file picker opens. Run-goSPL.cmd supports double-clicking and drag-and-drop.
.PARAMETER Processes
Number of MPI workers, from 1 to 8. The laptop default is 2.
.PARAMETER DryRun
Resolve the configuration and print the exact executable and argument array without
starting goSPL. Path conversion still requires the DeepTime-goSPL WSL distribution.
.EXAMPLE
.\Run-goSPL.ps1 -InputFile "C:\Worlds\My planet\input.yml" -DryRun
.EXAMPLE
.\Run-goSPL.ps1 -InputFile "C:\Worlds\My planet" -Processes 2
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$InputFile,
    [ValidateRange(1, 8)]
    [int]$Processes = 2,
    [switch]$DryRun
)

function Select-GoSPLConfiguration {
    Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    try {
        $dialog.Title = 'Select input.yml or forcing-check.yml from an extracted goSPL export'
        $dialog.Filter = 'goSPL configuration (*.yml;*.yaml)|*.yml;*.yaml'
        $dialog.CheckFileExists = $true
        $dialog.Multiselect = $false
        $dialog.InitialDirectory = $PSScriptRoot
        if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { return $null }
        return $dialog.FileName
    }
    finally { $dialog.Dispose() }
}

function Resolve-GoSPLConfiguration {
    param([Parameter(Mandatory = $true)][string]$Path)
    $item = Get-Item -LiteralPath $Path -ErrorAction Stop
    if ($item.PSProvider.Name -ne 'FileSystem') { throw 'Choose a YAML file on disk.' }
    if ($item.PSIsContainer) {
        $item = Get-Item -LiteralPath (Join-Path $item.FullName 'input.yml') -ErrorAction Stop
    }
    if ($item.PSIsContainer -or $item.Extension -notin @('.yml', '.yaml')) {
        throw 'Choose input.yml or forcing-check.yml, or their containing export directory.'
    }
    $inputDirectory = Join-Path $item.DirectoryName 'input'
    if (-not (Test-Path -LiteralPath $inputDirectory -PathType Container)) {
        throw "The adjacent input folder is missing. Extract the export ZIP and keep the YAML file beside its input folder (mesh and forcing files)."
    }
    return $item.FullName
}

function Get-GoSPLLaunchPlan {
    param(
        [Parameter(Mandatory = $true)][string]$Configuration,
        [ValidateRange(1, 8)][int]$Processes = 2
    )
    $wslExecutable = (Get-Command wsl.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
    $conversionArguments = @('--distribution', 'DeepTime-goSPL', '--user', 'gospl',
        '--exec', 'wslpath', '-a', '-u', $Configuration)
    # --exec invokes the binary directly; paths are arguments, never shell text.
    $conversionOutput = @(& $wslExecutable @conversionArguments 2>&1)
    $conversionExit = $LASTEXITCODE
    if ($conversionExit -ne 0) {
        $details = ($conversionOutput | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
        throw "WSL could not resolve the selected configuration (exit $conversionExit). Check that DeepTime-goSPL is installed and the file is accessible. $details"
    }
    $conversionLines = @($conversionOutput | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ })
    $linuxPaths = @($conversionLines | Where-Object { $_.StartsWith('/') })
    $startupMessages = @($conversionLines | Where-Object { -not $_.StartsWith('/') })
    if ($startupMessages.Count) { Write-Warning ($startupMessages -join [Environment]::NewLine) }
    if ($linuxPaths.Count -ne 1) { throw 'WSL must return exactly one absolute Linux configuration path. Select a configuration in an extracted export folder.' }
    $linuxConfiguration = $linuxPaths[0]
    $separator = $linuxConfiguration.LastIndexOf('/')
    if (-not $linuxConfiguration.StartsWith('/') -or $separator -lt 0 -or $separator -eq $linuxConfiguration.Length - 1) {
        throw 'WSL did not return an absolute Linux configuration path.'
    }
    $linuxDirectory = $linuxConfiguration.Substring(0, $separator)
    if (-not $linuxDirectory) { $linuxDirectory = '/' }
    $filename = $linuxConfiguration.Substring($separator + 1)
    $launchArguments = @('--distribution', 'DeepTime-goSPL', '--user', 'gospl',
        '--cd', $linuxDirectory, '--exec', 'env',
        'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OMPI_MCA_btl=self,vader,tcp',
        'OMPI_MCA_btl_vader_single_copy_mechanism=none',
        '/home/gospl/.local/bin/micromamba', 'run', '-p', '/home/gospl/micromamba/envs/gospl',
        'mpirun', '-np', [string]$Processes, 'gospl', '-i', $filename, '-v')
    return [pscustomobject]@{
        Configuration = $Configuration
        LinuxConfiguration = $linuxConfiguration
        WorkingDirectory = $linuxDirectory
        Processes = $Processes
        Executable = $wslExecutable
        Arguments = $launchArguments
    }
}

function Invoke-GoSPLLauncher {
    param([string]$InputFile, [ValidateRange(1, 8)][int]$Processes = 2, [switch]$DryRun)
    try {
        if ([string]::IsNullOrWhiteSpace($InputFile)) {
            $InputFile = Select-GoSPLConfiguration
            if (-not $InputFile) { Write-Host 'No configuration selected. Nothing was started.'; return 0 }
        }
        $configuration = Resolve-GoSPLConfiguration -Path $InputFile
        $plan = Get-GoSPLLaunchPlan -Configuration $configuration -Processes $Processes
        Write-Host "Configuration: $($plan.Configuration)"
        Write-Host "Working directory in WSL: $($plan.WorkingDirectory)"
        Write-Host "MPI workers: $($plan.Processes) (one numerical-library thread per worker)"
        if ($DryRun) {
            Write-Host 'Dry run: no model was started. Exact executable and separate arguments:'
            Write-Host ($plan | Select-Object Executable, Arguments | ConvertTo-Json -Depth 3)
            return 0
        }
        Write-Host 'Keep the YAML file and its adjacent input folder together. goSPL writes output beneath this configuration directory.'
        Write-Host 'Running goSPL. Press Ctrl+C to stop the model.'
        $wslExecutable = $plan.Executable
        $launchArguments = $plan.Arguments
        # Stream native output to the interactive terminal; preserve its exit code.
        & $wslExecutable @launchArguments | Out-Host
        $modelExit = $LASTEXITCODE
        if ($modelExit -ne 0) {
            Write-Host "goSPL stopped with exit code $modelExit. The model's details are above." -ForegroundColor Yellow
        }
        else { Write-Host 'goSPL finished successfully.' }
        return $modelExit
    }
    catch {
        Write-Host "Unable to run goSPL: $($_.Exception.Message)" -ForegroundColor Red
        return 2
    }
}

# Dot-sourcing exposes the small functions for tests without opening a dialog.
if ($MyInvocation.InvocationName -ne '.') {
    exit (Invoke-GoSPLLauncher @PSBoundParameters)
}
