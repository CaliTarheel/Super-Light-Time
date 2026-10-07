param(
    [string]$PythonPath,
    [string]$WheelDirectory,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$environmentRoot = Join-Path $projectRoot '.venv'
$environmentPython = Join-Path $environmentRoot 'Scripts\python.exe'
$requirementsPath = Join-Path $projectRoot 'requirements-portable.txt'

# Python is always called as an executable plus arguments, never evaluated as
# shell text. A path containing spaces is safe in -PythonPath or -WheelDirectory.
function Get-PythonInfo([string]$Executable) {
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $null }
    $probe = "import json,platform,struct,sys; print(json.dumps(dict(executable=sys.executable,version=list(sys.version_info[:3]),bits=struct.calcsize('P')*8,implementation=platform.python_implementation(),prefix=sys.prefix,base_prefix=sys.base_prefix)))"
    try {
        $result = & $Executable -I -c $probe 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        return ($result | ConvertFrom-Json)
    } catch { return $null }
}

function Test-CompatiblePython($Info) {
    return ($null -ne $Info -and $Info.version[0] -eq 3 -and $Info.version[1] -eq 12 -and $Info.bits -eq 64 -and $Info.implementation -eq 'CPython')
}

try {
    if (-not (Test-Path -LiteralPath $requirementsPath -PathType Leaf)) {
        throw 'requirements-portable.txt is missing. Extract the entire ZIP before running Setup.cmd.'
    }
    $required = @{}
    foreach ($line in Get-Content -LiteralPath $requirementsPath) {
        if ($line -match '^([A-Za-z0-9_-]+)==([0-9][A-Za-z0-9.+-]*)\s*$') { $required[$Matches[1]] = $Matches[2] }
        elseif ($line.Trim() -and -not $line.Trim().StartsWith('#')) { throw "Unexpected dependency declaration: $line" }
    }
    if ($required.Count -eq 0) { throw 'The pinned dependency list is empty.' }

    if (Test-Path -LiteralPath $environmentRoot) {
        $info = Get-PythonInfo $environmentPython
        if (-not (Test-CompatiblePython $info) -or $info.prefix -eq $info.base_prefix -or [IO.Path]::GetFullPath($info.prefix).TrimEnd('\') -ne [IO.Path]::GetFullPath($environmentRoot).TrimEnd('\')) {
            throw 'An incompatible or incomplete .venv already exists. Nothing was overwritten. Keep it by renaming that folder, then run Setup.cmd again.'
        }
        $installed = & $environmentPython -I -c "import importlib.metadata,json; print(json.dumps({d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()}))"
        if ($LASTEXITCODE -ne 0) { throw 'The existing .venv cannot read its installed packages. It was left unchanged.' }
        $versions = $installed | ConvertFrom-Json
        foreach ($package in $required.Keys) {
            $property = $versions.PSObject.Properties[$package.ToLowerInvariant()]
            if ($null -ne $property -and $property.Value -ne $required[$package]) {
                throw "The existing .venv has $package $($property.Value), but this handoff requires $($required[$package]). It was left unchanged. Rename .venv to keep it, then run Setup.cmd again."
            }
        }
        Write-Host "Using the existing local Python $($info.version -join '.') environment."
    } else {
        $candidates = @()
        if ($PythonPath) {
            $candidates += $PythonPath
        } else {
            $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
            if ($launcher) {
                try {
                    $launched = & $launcher.Source -3.12 -I -c 'import sys; print(sys.executable)' 2>$null
                    if ($LASTEXITCODE -eq 0 -and $launched) { $candidates += [string]$launched }
                } catch { }
            }
            foreach ($name in @('python.exe', 'python3.exe')) {
                $command = Get-Command $name -ErrorAction SilentlyContinue
                if ($command) { $candidates += $command.Source }
            }
            if ($env:LOCALAPPDATA) { $candidates += Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe' }
        }
        $runtime = $null
        foreach ($candidate in $candidates | Select-Object -Unique) {
            $candidateInfo = Get-PythonInfo $candidate
            if (Test-CompatiblePython $candidateInfo) { $runtime = $candidateInfo.executable; $info = $candidateInfo; break }
        }
        if (-not $runtime) {
            throw 'Install 64-bit CPython 3.12 from https://www.python.org/downloads/windows/ (include pip and the Python launcher), then run Setup.cmd again. An existing install can be selected with Setup.cmd -PythonPath "C:\path\to\python.exe". No files were changed.'
        }
        Write-Host "Found Python $($info.version -join '.') at $runtime"
        if ($CheckOnly) {
            Write-Host 'Prerequisites found. CheckOnly did not create an environment or install packages.'
            exit 0
        }
        Write-Host 'Creating .venv inside this app folder...'
        & $runtime -I -m venv $environmentRoot
        if ($LASTEXITCODE -ne 0) { throw 'Python could not create .venv. Ensure Python includes venv and pip, and extract the app to a writable folder. Any partial .venv was retained for inspection.' }
    }

    if ($CheckOnly) {
        & $environmentPython -I -c "import numpy,PIL,h5py; print('NumPy',numpy.__version__,'| Pillow',PIL.__version__,'| h5py',h5py.__version__)"
        if ($LASTEXITCODE -ne 0) { throw 'The compatible .venv is missing a required package. Run Setup.cmd without -CheckOnly to finish installation.' }
        Write-Host 'The portable environment is ready. CheckOnly made no changes.'
        exit 0
    }

    $installArguments = @('-I', '-m', 'pip', '--isolated', 'install', '--disable-pip-version-check', '--only-binary=:all:', '--requirement', $requirementsPath)
    if (-not $WheelDirectory -and (Test-Path -LiteralPath (Join-Path $projectRoot 'wheels') -PathType Container)) {
        $WheelDirectory = Join-Path $projectRoot 'wheels'
    }
    if ($WheelDirectory) {
        if (-not (Test-Path -LiteralPath $WheelDirectory -PathType Container)) { throw "Wheel directory does not exist: $WheelDirectory" }
        $installArguments += @('--no-index', '--find-links', (Resolve-Path -LiteralPath $WheelDirectory).Path)
        Write-Host 'Installing pinned packages from the local wheel directory...'
    } else {
        Write-Host 'Installing pinned packages from PyPI. This first setup needs an internet connection...'
    }
    & $environmentPython @installArguments
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed. Read the pip error above, then rerun Setup.cmd after fixing the connection or wheel directory. This app installs only into its own .venv.' }

    & $environmentPython -I -c "import io,numpy as np,PIL,h5py; from PIL import Image; b=io.BytesIO(); a=np.arange(6,dtype=np.float32); f=h5py.File(b,'w'); f.create_dataset('elev',data=a); assert np.array_equal(f['elev'][:],a); f.close(); Image.new('RGB',(2,2)).save(io.BytesIO(),format='PNG'); print('Verified NumPy',np.__version__,'| Pillow',PIL.__version__,'| h5py',h5py.__version__)"
    if ($LASTEXITCODE -ne 0) { throw 'Installed packages failed the array, image, or HDF5 check. The app environment was retained for inspection.' }
    Write-Host ''
    Write-Host 'Dependencies are installed. Start.cmd opens the interface; see README.md for production binding requirements.'
    Write-Host 'goSPL result import is ready; running the goSPL solver requires its separate Linux/WSL setup.'
} catch {
    Write-Host ''
    Write-Host ('Setup could not finish: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
