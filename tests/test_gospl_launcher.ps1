# No WSL installation or model run: native calls are replaced by an argument spy.
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
. (Join-Path $workspace 'Run-goSPL.ps1')
$testParent = [IO.Path]::GetFullPath((Join-Path $workspace 'tmp'))
$testRoot = Join-Path $testParent ('gospl-launcher-test-' + [guid]::NewGuid().ToString('N'))
$script:calls = @()
$script:convertedPath = '/mnt/c/Worlds/O''Brien planet/input.yml'
$script:conversionExit = 0
$script:startupWarning = ''
$script:extraPath = ''
$script:multipleWslMatches = $false
$script:modelExit = 0
$script:checks = 0

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Test-Check([string]$Name, [scriptblock]$Action) {
    & $Action
    $script:checks++
    Write-Host "PASS $Name"
}
function Get-Command {
    param([string]$Name, [string]$CommandType, [object]$ErrorAction)
    if ($Name -ne 'wsl.exe') { throw "Unexpected command lookup: $Name" }
    [pscustomobject]@{ Source = 'Invoke-TestWsl' }
    if ($script:multipleWslMatches) { [pscustomobject]@{ Source = 'Unused-WindowsApps-Wsl' } }
}
function Invoke-TestWsl {
    $script:calls += ,@($args)
    if ($args -contains 'wslpath') {
        $global:LASTEXITCODE = $script:conversionExit
        if ($script:conversionExit -eq 0) {
            if ($script:startupWarning) { Write-Output $script:startupWarning }
            if ($script:extraPath) { Write-Output $script:extraPath }
            return $script:convertedPath
        }
        return 'Test distro is unavailable'
    }
    $global:LASTEXITCODE = $script:modelExit
    Write-Output 'Test model output'
}

try {
    $exportDirectory = Join-Path $testRoot "O'Brien planet [draft]"
    New-Item -ItemType Directory -Path (Join-Path $exportDirectory 'input') -Force | Out-Null
    $yaml = Join-Path $exportDirectory 'input.yml'
    $reviewYaml = Join-Path $exportDirectory 'forcing-check.yml'
    [IO.File]::WriteAllText($yaml, '{}')
    [IO.File]::WriteAllText($reviewYaml, '{}')

    Test-Check 'directory and literal YAML paths select the same input without wildcard expansion' {
        Assert-True ((Resolve-GoSPLConfiguration $exportDirectory) -eq $yaml) 'Directory did not select input.yml.'
        Assert-True ((Resolve-GoSPLConfiguration $yaml) -eq $yaml) 'Literal path was changed.'
        Assert-True ((Resolve-GoSPLConfiguration $reviewYaml) -eq $reviewYaml) 'Forcing review file was changed.'
    }
    Test-Check 'spaces, apostrophes and brackets are each passed as one wslpath argument' {
        $script:calls = @()
        $plan = Get-GoSPLLaunchPlan -Configuration $yaml
        $expected = @('--distribution','DeepTime-goSPL','--user','gospl','--exec','wslpath','-a','-u',$yaml)
        Assert-True (($script:calls[0] | ConvertTo-Json -Compress) -eq ($expected | ConvertTo-Json -Compress)) 'Path conversion arguments changed or split.'
        Assert-True ($plan.WorkingDirectory -eq "/mnt/c/Worlds/O'Brien planet") 'Incorrect Linux working directory.'
    }
    Test-Check 'Windows native argument serialization preserves spaces and apostrophes' {
        $echoScript = Join-Path $testRoot 'echo arguments.ps1'
        [IO.File]::WriteAllText($echoScript, 'ConvertTo-Json -InputObject @($args) -Compress')
        $nativeArguments = @('--distribution', 'DeepTime-goSPL', '--exec', 'wslpath', '-a', '-u', $yaml,
            '--cd', "/mnt/c/Worlds/O'Brien planet", '-i', 'forcing-check.yml')
        $echoed = & powershell.exe -NoProfile -File $echoScript @nativeArguments
        Assert-True ($LASTEXITCODE -eq 0) 'Native argument echo failed.'
        $roundTrip = ConvertFrom-Json $echoed
        Assert-True (($roundTrip | ConvertTo-Json -Compress) -eq ($nativeArguments | ConvertTo-Json -Compress)) 'Windows native command-line serialization changed an argument.'
    }
    Test-Check 'multiple installed wsl.exe matches resolve to one executable' {
        $script:multipleWslMatches = $true
        $plan = Get-GoSPLLaunchPlan -Configuration $yaml
        Assert-True ($plan.Executable -eq 'Invoke-TestWsl') 'Multiple executable paths were combined.'
        Assert-True ($plan.LinuxConfiguration -eq $script:convertedPath) 'Path conversion did not use the first executable.'
        $script:multipleWslMatches = $false
    }
    Test-Check 'benign WSL startup warnings are displayed without blocking the resolved path' {
        $script:startupWarning = "wsl: Unknown key 'wsl2.pageReporting' in the existing .wslconfig"
        $plan = Get-GoSPLLaunchPlan -Configuration $yaml -WarningVariable warnings
        Assert-True ($plan.LinuxConfiguration -eq $script:convertedPath) 'Startup warning prevented path resolution.'
        Assert-True (($warnings -join ' ') -match 'pageReporting') 'Startup warning was hidden.'
        $script:startupWarning = ''
    }
    Test-Check 'more than one absolute Linux path is rejected' {
        $script:extraPath = '/mnt/c/other/input.yml'; $rejected = $false
        try { Get-GoSPLLaunchPlan -Configuration $yaml | Out-Null } catch { $rejected = $true }
        Assert-True $rejected 'Ambiguous absolute paths were accepted.'
        $script:extraPath = ''
    }
    Test-Check 'launch uses the named distro, exact environment, two workers and single-thread limits' {
        $plan = Get-GoSPLLaunchPlan -Configuration $yaml
        $expected = @('--distribution','DeepTime-goSPL','--user','gospl','--cd',"/mnt/c/Worlds/O'Brien planet",'--exec','env',
            'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','OMPI_MCA_btl=self,vader,tcp',
            'OMPI_MCA_btl_vader_single_copy_mechanism=none',
            '/home/gospl/.local/bin/micromamba','run','-p','/home/gospl/micromamba/envs/gospl',
            'mpirun','-np','2','gospl','-i','input.yml','-v')
        Assert-True (($plan.Arguments | ConvertTo-Json -Compress) -eq ($expected | ConvertTo-Json -Compress)) 'Launch differs from the dedicated installation contract.'
        Assert-True (-not ($plan.Arguments -contains '-c')) 'Launcher must not interpolate a shell command.'
    }
    Test-Check 'a forcing-check filename stays intact and worker count can be changed to eight' {
        $script:convertedPath = '/mnt/c/Worlds/O''Brien planet/forcing-check.yml'
        $plan = Get-GoSPLLaunchPlan -Configuration $reviewYaml -Processes 8
        Assert-True ($plan.Arguments[-4] -eq 'gospl') 'goSPL executable is missing.'
        Assert-True ($plan.Arguments[-2] -eq 'forcing-check.yml') 'Selected review configuration was changed.'
        Assert-True ($plan.Arguments[19] -eq '8') 'Requested worker count was not passed.'
        $script:convertedPath = '/mnt/c/Worlds/O''Brien planet/input.yml'
    }
    Test-Check 'dry run converts paths but never invokes the model' {
        $script:calls = @()
        $result = Invoke-GoSPLLauncher -InputFile $exportDirectory -DryRun
        Assert-True ($result -eq 0) 'Dry run should succeed.'
        Assert-True ($script:calls.Count -eq 1 -and $script:calls[0] -contains 'wslpath') 'Dry run invoked a model.'
    }
    Test-Check 'model launch preserves native success and failure codes' {
        $script:calls = @(); $script:modelExit = 0
        Assert-True ((Invoke-GoSPLLauncher -InputFile $yaml) -eq 0) 'Success code was changed.'
        Assert-True ($script:calls.Count -eq 2) 'Expected conversion and one model invocation.'
        $script:modelExit = 17
        Assert-True ((Invoke-GoSPLLauncher -InputFile $yaml) -eq 17) 'Native failure code was changed.'
        $script:modelExit = 0
    }
    Test-Check 'failed WSL conversion gives a setup error without launching' {
        $script:calls = @(); $script:conversionExit = 1
        Assert-True ((Invoke-GoSPLLauncher -InputFile $yaml) -eq 2) 'Setup error should return 2.'
        Assert-True ($script:calls.Count -eq 1) 'Model launched after failed path conversion.'
        $script:conversionExit = 0
    }
    Test-Check 'missing adjacent input folder is rejected before any native call' {
        $looseFile = Join-Path $testRoot 'loose.yml'; [IO.File]::WriteAllText($looseFile, '{}')
        $script:calls = @()
        Assert-True ((Invoke-GoSPLLauncher -InputFile $looseFile) -eq 2) 'Loose configuration should fail.'
        Assert-True ($script:calls.Count -eq 0) 'Native call made for an incomplete export.'
    }
    Test-Check 'worker limits reject zero and nine before a native call' {
        foreach ($workers in @(0, 9)) {
            $rejected = $false
            try { Get-GoSPLLaunchPlan -Configuration $yaml -Processes $workers | Out-Null } catch { $rejected = $true }
            Assert-True $rejected "Worker count $workers was accepted."
        }
    }
    Test-Check 'cancelling the file picker succeeds without a model or environment change' {
        function Select-GoSPLConfiguration { return $null }
        $script:calls = @()
        Assert-True ((Invoke-GoSPLLauncher) -eq 0) 'Cancelling should be a clean exit.'
        Assert-True ($script:calls.Count -eq 0) 'Cancelled selection invoked WSL.'
    }
    Write-Host "$script:checks goSPL launcher checks passed"
}
finally {
    # Only the unique directory created by this test can be removed.
    $resolved = [IO.Path]::GetFullPath($testRoot)
    $allowedPrefix = $testParent.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if ($resolved.StartsWith($allowedPrefix, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path $resolved -Leaf).StartsWith('gospl-launcher-test-') -and
        (Test-Path -LiteralPath $resolved)) {
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}
